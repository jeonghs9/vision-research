"""si_vietnam_drone 2클래스 — GANet-medium(ResNet34), P3 제거 + C5 SA 를 APT 토큰으로 계산 (SA 유지).

res34_s16_full_2cls_P3del.py와 전부 동일, neck 에 APT(Adaptive Patch Transformer, arXiv 2510.18091) 토큰 병합(apt_cfg)만 추가 (dcn_fpn.py APT 부분).
C5 한 칸(입력 32x32) = 기본 패치, 입력 64x64 픽셀 엔트로피 < apt_threshold 인 2x2 칸은 큰 패치 토큰 1개로 합침.
"""
from projects.cfgs.drone.res34_s16_full_2cls_P3del import *  # noqa

apt_threshold = 6.5  # 64x64 엔트로피 문턱값: C5 토큰 감소 train 27% / val 15% (논문 검출 실험 ~30% 에 맞춤)
work_dir = './work_dirs/drone/res34_s16_full_2cls_P3del_APT'
model['neck']['apt_cfg'] = dict(threshold=apt_threshold, base_patch=32, bins=512,
                                mean=img_norm_cfg['mean'], std=img_norm_cfg['std'])
model['neck']['trans_cfg']['use_apt'] = True
for _k in ('train', 'val', 'test'):
    data[_k]['work_dir'] = work_dir
del _k
