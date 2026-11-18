# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Yinchao Ma
# @Email   : imyc@mail.ustc.edu.cn
# --------------------------------------------------------
#
# ★ 이 파일은 source/projects/plugin/models/dcn_fpn.py 를 그대로 복사한 뒤
#   학습용 한글 주석만 추가한 사본입니다. 실제 실행에는 쓰이지 않습니다
#   (원본은 source/ 쪽 — git으로 버전관리되는 진짜 코드).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init  # ConvModule = conv+BN+활성화를 한번에 묶어주는 mmcv 헬퍼

from .dense_heads import LanePointsConv   # ★ LFA(Lane-aware Feature Aggregator) 본체
from mmcv.runner import auto_fp16
from mmdet.models import NECKS
from mmdet.models.utils import SinePositionalEncoding  # transformer류에서 쓰는 "위치 정보" 인코딩


def build_position_encoding(hidden_dim, shape):
    # self-attention에 넣을 위치 임베딩을 만든다.
    # attention 자체는 "순서"를 모르는 연산이라(모든 위치를 동등하게 섞어버림),
    # "이 픽셀이 화면의 어디쯤인지" 정보를 별도로 더해줘야 한다 -- 그 역할.
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = SinePositionalEncoding(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs


class AttentionLayer(nn.Module):
    """ Position attention module"""
    # ★ Self-Attention 블록. "이미지 전체의 모든 픽셀이 서로를 참고"하게 만드는 모듈.
    # 차선처럼 화면을 길게 가로지르는 구조는, 가까운 픽셀끼리만 보는 CNN(conv)으로는
    # "저 멀리 떨어진 두 점이 사실 같은 선이다"라는 관계를 놓치기 쉬워서 추가됨.

    def __init__(self, in_dim, out_dim, ratio=4, stride=1):  # 우리 설정: (512, 64, 4, 1)
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        norm_cfg = dict(type='BN', requires_grad=True)
        act_cfg = dict(type='ReLU')

        # 입력 채널(512, C5)을 먼저 64채널로 줄이는 conv (뒤 연산량을 줄이기 위함)
        self.pre_conv = ConvModule(
            in_dim, out_dim, kernel_size=3, stride=stride, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)

        # attention의 핵심 3요소: Query(질문) / Key(비교 대상) / Value(실제 가져올 내용)
        # ratio=4 라서 query/key는 채널을 1/4로 더 줄여서 계산량을 아낀다 (64 -> 16)
        self.query_conv = nn.Conv2d(out_dim, out_dim // ratio, kernel_size=1)  # 64,16,1
        self.key_conv = nn.Conv2d(out_dim, out_dim // ratio, kernel_size=1)    # 64,16,1
        self.value_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)           # 64,64,1

        # attention 결과를 한 번 더 다듬는 conv
        self.final_conv = ConvModule(
            out_dim, out_dim, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.softmax = nn.Softmax(dim=-1)
        # gamma: "attention 결과를 얼마나 반영할지" 세기를 학습으로 정하는 스칼라.
        # 0으로 초기화해서, 학습 초반엔 attention 영향 없이(원본 그대로) 시작하고
        # 학습이 진행되며 필요한 만큼만 점점 섞이게 만드는 안전장치.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pos=None):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x = self.pre_conv(x)                       # (B,512,H,W) -> (B,64,H,W)
        m_batchsize, _, height, width = x.size()
        if pos is not None:
            x += pos                                # 위치 정보 더하기

        # (B,64,H,W) -> (B, H*W, 16)  "이 위치에서 뭘 찾고 있나"
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # (B,64,H,W) -> (B, 16, H*W)  "각 위치가 뭘 갖고 있나"
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # Query·Key를 행렬곱 -> (B, H*W, H*W) : 모든 위치 쌍의 "얼마나 관련있는지" 점수
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)            # 점수를 0~1 확률처럼 정규화
        attention = attention.permute(0, 2, 1)

        # (B,64,H,W) -> (B,64,H*W) "각 위치의 실제 특징값"
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # attention 가중치대로 value를 섞어 모음 -> "다른 위치 정보를 참고해서 합성된 특징"
        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, -1, height, width)

        proj_value = proj_value.view(m_batchsize, -1, height, width)  # (사용 안 됨, 죽은 코드)

        # gamma로 세기 조절해서 원본(x)에 더하기 (residual) -> 다시 conv로 다듬기
        out_feat = self.gamma * out + x
        out_feat = self.final_conv(out_feat)
        return out_feat


class TransConvEncoderModule(nn.Module):
    # AttentionLayer를 여러 겹(attn_in_dims/out_dims 리스트 길이만큼) 쌓아주는 래퍼.
    # 우리 설정은 attn_in_dims=[512,64], attn_out_dims=[64,64] 이므로 2겹:
    #   1번째: 512채널 입력받아 64채널로, 2번째: 64채널을 64채널로 (정제만)
    def __init__(self,
            attn_in_dims,
            attn_out_dims,
            strides,
            ratios,
            pos_shape=None):
        super(TransConvEncoderModule, self).__init__()
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))

        if pos_shape is not None:
            # pos_shape가 있으면(우리 설정이 이쪽) 위치 임베딩을 미리 계산해서
            # 레이어마다 forward 때 직접 넘겨줌 -> ModuleList로 보관(각자 따로 호출해야 하므로)
            self.attn_layers = nn.ModuleList(attn_layers)
        else:
            # 없으면 그냥 순서대로 이어붙여서 실행 (nn.Sequential)
            self.attn_layers = nn.Sequential(*attn_layers)

        self.pos_shape = pos_shape
        self.pos_embeds = []
        if pos_shape is not None:
            for dim in attn_out_dims:
                pos_embed = build_position_encoding(dim, pos_shape).cuda()
                self.pos_embeds.append(pos_embed)

    def forward(self, src):
        if self.pos_shape is None:
            src = self.attn_layers(src)
        else:
            # 레이어마다 자기 전용 위치 임베딩을 넣어주며 순서대로 통과
            for layer, pos in zip(self.attn_layers, self.pos_embeds):
                src = layer(src, pos.to(src.device))
        return src


@NECKS.register_module()  # ★ 이 데코레이터로 "DeformFPN"이라는 이름이 레지스트리에 등록됨
                           #   -> config에서 neck=dict(type='DeformFPN', ...)로 쓸 수 있게 됨
class DeformFPN(nn.Module):

    def __init__(self,
                 in_channels,          # [128, 256, 512]  (C3,C4,C5 채널수)
                 out_channels,         # 64  (neck 내부 통일 채널수, = fpn_out_dim)
                 start_level=0,
                 end_level=-1,
                 dcn_point_num=[9,7,5,3],         # 우리 설정은 [7, 5, 3] (레벨별 LFA가 볼 인접점 수)
                 deconv_layer=[True, True, True, True],  # 우리 설정은 [True, False, False]
                                                            # -> "어느 레벨에 LFA를 적용할지" on/off
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 use_latern=False,
                 use_res=False,        # LFA 결과를 원본에 "더할지"(True) "교체할지"(False)
                 deconv_before=False,  # LFA를 top-down 합치기 "전에"(True) 할지 "후에"(False) 할지
                 trans_cfg=None,       # self-attention 설정 딕셔너리 (None이면 attention 자체를 안 씀)
                 dconv_cfg=None,
                 trans_idx=-1,         # self-attention을 적용할 레벨 인덱스 (-1 = 가장 깊은 레벨)
                 trans_mode="replace", # attention 결과로 원래 레벨을 "교체"할지
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dcn_only_cls=False):  # ★ True면 "LFA는 confidence map(분류) 쪽에만 쓴다"는 뜻
        super(DeformFPN, self).__init__()
        self.deconv_layer = deconv_layer
        self.deconv_before = deconv_before
        self.use_res = use_res
        self.trans_cfg = trans_cfg
        self.trans_mode = trans_mode
        self.dcn_only_cls = dcn_only_cls

        if self.trans_cfg is not None:
            # 마지막 레벨(C5)의 채널수를, attention을 거치고 나면 될 채널수(64)로 미리 바꿔둠.
            # 안 바꾸면 바로 아래에서 lateral_conv를 만들 때 입력 채널이 틀어짐.
            in_channels[-1] = trans_cfg['attn_out_dims'][-1]  # [128,256,512] -> [128,256,64]
            self.trans_idx = trans_idx
            self.trans_head = TransConvEncoderModule(**trans_cfg)  # self-attention 모듈 생성

        num_outs = len(in_channels)              # 3
        upsample_cfg = {'mode': 'nearest'}        # top-down 합칠 때 쓸 업샘플 방식(최근접 보간)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels            # [128,256,64]
        self.out_channels = out_channels          # 64
        self.num_ins = len(in_channels)           # 3
        self.num_outs = num_outs                  # 3
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins  # 3
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level  # 0
        self.end_level = end_level      # -1
        self.add_extra_convs = add_extra_convs  # False (우리 설정)
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()   # 레벨별 1x1 conv (채널 통일용)
        self.fpn_convs = nn.ModuleList()       # 레벨별 3x3 conv (최종 다듬기용)

        # ★ 레벨 3개(i=0,1,2) 각각에 대해 1x1 conv + 3x3 conv 한 쌍씩 만들어둠
        for i in range(self.start_level, self.backbone_end_level):  # 0, 1, 2
            l_conv = ConvModule(
                in_channels[i], out_channels, 1,          # 1x1 conv: 128/256/64 -> 64
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg, inplace=False)
            fpn_conv = ConvModule(
                out_channels, out_channels, 3, padding=1,  # 3x3 conv: 64 -> 64
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # 레벨을 더 늘리고 싶을 때 쓰는 옵션. add_extra_convs=False(우리 설정)라서
        # 아래 블록은 실제로는 실행되지 않는다 -- 읽을 때 건너뛰어도 됨.
        extra_levels = num_outs - self.backbone_end_level + self.start_level  # 0
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels, out_channels, 3, stride=2, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # ★ LFA(LanePointsConv) 모듈도 레벨 3개 전부에 대해 만들어둠.
        #   주의: "만들어둔다"와 "실제로 쓰인다"는 다르다 -- forward를 보면
        #   deconv_layer=[True,False,False] 때문에 def_convs[0]만 실제로 호출되고
        #   def_convs[1], def_convs[2]는 생성만 되고 forward에서 한 번도 안 불린다
        #   (아래 forward의 top-down 루프 주석 참고).
        self.def_convs = nn.ModuleList()
        if dconv_cfg is None:
            feat_channels = 256       # LFA 내부 conv 채널수
            stacked_convs = 3         # LFA 내부 conv를 몇 겹 쌓을지
        else:
            feat_channels = dconv_cfg['feat_channels']
            stacked_convs = dconv_cfg['stacked_convs']
        for i in range(self.start_level, self.backbone_end_level):  # 0, 1, 2
            def_conv = LanePointsConv(
                in_channels=out_channels,          # 64
                feat_channels=feat_channels,       # 256
                point_feat_channels=out_channels,  # 64
                num_points=dcn_point_num[i],       # 레벨별 인접점 개수: 7, 5, 3
                use_latern=use_latern,
                stacked_convs=stacked_convs)
            self.def_convs.append(def_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    # ↓ 원본 저자가 주석 처리(''' ... ''')해서 꺼둔 중복 버전 -- 바로 아래 진짜 init_weights와
    #   내용이 똑같다. 실행되는 코드는 아니고, 그냥 지우지 않고 남겨둔 죽은 코드.
    '''
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    '''

    def init_weights(self):
        # ★ 실제로 호출되는 버전. 모든 Conv2d를 Xavier 방식으로 초기화 (학습 시작 전 한 번 호출됨)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()  # 이 함수 안 연산은 자동으로 fp16(절반 정밀도)로 돌려도 된다는 mmcv 표시
    def forward(self, inputs):
        # inputs: 백본(ResNet)이 이미 계산해서 건네준 결과. 이 함수 "안"에는 백본 연산이 없다 --
        # 백본 호출은 이 클래스 바깥(SingleStageDetector.extract_feat)에서 먼저 끝나 있다.

        inputs = list(inputs)
        if len(inputs) > len(self.in_channels):
            del inputs[0]   # 백본이 4개(C2,C3,C4,C5) 주면, 가장 저수준인 C2를 버림
        assert len(inputs) == len(self.in_channels)
        # 여기까지: inputs = [C3(128ch,92x160), C4(256ch,46x80), C5(512ch,23x40)]

        # ── 1단계: self-attention (가장 깊은 레벨=C5에만 적용) ──────────────
        if self.trans_cfg is not None:
            trans_feat = self.trans_head(inputs[self.trans_idx])  # trans_idx=-1 -> C5
            if self.trans_mode == 'replace':
                inputs[self.trans_idx] = trans_feat   # C5 자리를 attention 결과로 통째로 교체
        # 여기까지: inputs = [C3(128ch), C4(256ch), AttnC5(64ch)]

        # ── 2단계: 레벨별 1x1 conv로 채널 통일 (+ deconv_before 옵션) ───────
        laterals = [None]*self.num_ins
        deform_points = [None]*self.num_ins

        for i, (lateral_conv, deform_conv) in enumerate(zip(self.lateral_convs, self.def_convs)):
            mid_feat = lateral_conv(inputs[i + self.start_level])
            laterals[i] = mid_feat
            # deconv_before=False(우리 설정)라서 이 블록은 실행 안 됨 -- 건너뛰어도 됨
            if self.deconv_before and self.deconv_layer[i]:
                d_feat, points = deform_conv(mid_feat)
                deform_points[i] = points
                if self.use_res:
                    laterals[i] = d_feat + mid_feat
                else:
                    laterals[i] = d_feat
        # 여기까지: laterals = [64ch@92x160, 64ch@46x80, 64ch@23x40]  (아직 레벨끼리 안 섞임)

        # ── 3단계: top-down 합치기(진짜 "FPN") + LFA ────────────────────────
        used_backbone_levels = len(laterals)  # 3
        for i in range(used_backbone_levels - 1, 0, -1):   # i = 2, 1 순서로 내려옴
            prev_shape = laterals[i - 1].shape[2:]
            # ★ 핵심 연산: 깊은 레벨을 업샘플해서 얕은 레벨에 "더한다" (concat 아님, add)
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            if i - 1 == 0:
                # 가장 얕은(최종) 레벨에 도달했을 때, LFA 적용 "전" 상태를 사본으로 떠둠.
                # -> 이게 head에서 confidence map(LFA O) vs offset map(LFA X, 이 aux_feat 사용)
                #    으로 갈라지는 바로 그 지점.
                aux_feat = laterals[0]

            # deconv_layer=[True, False, False] 이므로 i-1=0 일 때만 조건이 참이 됨
            # -> LFA(def_convs[0])는 레벨0에만 적용되고, def_convs[1]/[2]는 여기서도 안 불림
            if self.deconv_layer[i-1] and not self.deconv_before:
                d_feat, points = self.def_convs[i-1](laterals[i - 1])
                if self.use_res:
                    laterals[i - 1] = d_feat + laterals[i - 1]   # 더하기(residual)
                else:
                    laterals[i - 1] = d_feat                      # 교체 (우리 설정: use_res=False)
                deform_points[i-1] = points
        # 여기까지: laterals[0]은 "C3+C4+C5 다 섞이고 LFA까지 적용된" 상태
        #          laterals[1]은 "C4+C5만 섞인" 상태 (LFA 없음)
        #          laterals[2]는 "attention만 적용된 C5" 상태 그대로 (아무것도 안 더해짐)

        # ── 4단계: 레벨별 3x3 conv로 마무리 ─────────────────────────────────
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # ── 5단계: 결과 포장해서 반환 ────────────────────────────────────────
        output = dict(
            features=tuple(outs),        # head로 넘어갈 최종 특징 3개 (LFA 적용된 레벨0 포함)
            aux_feat=None,
            deform_points=tuple(deform_points),  # 학습 시 aux loss 계산에만 쓰임(추론 땐 안 씀)
        )
        if self.dcn_only_cls:
            # True(우리 설정)면 LFA 적용 "전" 레벨0 사본(aux_feat)도 같이 내보냄
            # -> head에서 confidence map은 features[0](LFA 적용본), offset map은
            #    이 aux_feat(LFA 미적용본)을 쓰게 됨
            output.update(aux_feat=aux_feat)
        return output
