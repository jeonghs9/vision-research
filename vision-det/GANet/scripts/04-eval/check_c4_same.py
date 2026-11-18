# 무학습 P3 제거 vs 기존: backbone C4/C5, C5+SA, P4(DCN 전) 가 같은지 수치 확인
import sys; sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch, torch.nn.functional as F
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa
cfg = mmcv.Config.fromfile("projects/cfgs/drone/res34_s8_full_2cls.py")
cfg.model.train_cfg = None; cfg.model.backbone.init_cfg = None
m = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(m, "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth", map_location="cpu")
m = m.cuda().eval(); neck = m.neck
W, H = cfg.img_scale
rgb = cv2.cvtColor(cv2.resize(cv2.imread("/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"), (W, H)), cv2.COLOR_BGR2RGB)
t = torch.from_numpy(((rgb - np.array(cfg.img_norm_cfg["mean"])) / np.array(cfg.img_norm_cfg["std"])).astype(np.float32).transpose(2, 0, 1))[None].cuda()
print("backbone out_indices:", m.backbone.out_indices, "neck in_channels:", neck.in_channels)
got = {}
neck.trans_head.register_forward_hook(lambda mod, i, o: got.setdefault("SA_orig", o))
with torch.no_grad():
    outs = m.backbone(t)
    print("backbone 출력 shape:", [tuple(o.shape) for o in outs])
    C3, C4, C5 = outs[1:]
    m.neck(outs)                                           # 기존 경로 (P3 포함)
    lat4 = neck.lateral_convs[1](C4)
    p4_noP3 = lat4 + F.interpolate(neck.lateral_convs[2](neck.trans_head(C5)), size=lat4.shape[2:], **neck.upsample_cfg)
    # 기존 경로 안의 P4 를 다시 계산 (DeformFPN.forward 와 같은 순서)
    lats = [neck.lateral_convs[0](C3), lat4, neck.lateral_convs[2](got["SA_orig"])]
    p4_orig = lats[1] + F.interpolate(lats[2], size=lats[1].shape[2:], **neck.upsample_cfg)
    print("P4(DCN 전) 기존 vs P3 제거 최대 차이:", float((p4_orig - p4_noP3).abs().max()))
    print("C5+SA 기존 vs P3 제거 최대 차이:", float((got["SA_orig"] - neck.trans_head(C5)).abs().max()))
