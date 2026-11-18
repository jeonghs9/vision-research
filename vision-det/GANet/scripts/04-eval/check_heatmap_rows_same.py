# 기존(P3) vs 무학습 P3 제거: 히트맵 1줄(활성 크기)과 2줄(실선 기여도) 이 단계별로 같은지 수치 비교
import sys; sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch, torch.nn.functional as F
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa
cfg = mmcv.Config.fromfile("projects/cfgs/drone/res34_s8_full_2cls.py")
cfg.model.train_cfg = None; cfg.model.backbone.init_cfg = None
m = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(m, "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth", map_location="cpu")
m = m.cuda().eval(); neck = m.neck; thr = m.bbox_head.kpt_thr
W, H = cfg.img_scale
rgb = cv2.cvtColor(cv2.resize(cv2.imread("/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"), (W, H)), cv2.COLOR_BGR2RGB)
t = torch.from_numpy(((rgb - np.array(cfg.img_norm_cfg["mean"])) / np.array(cfg.img_norm_cfg["std"])).astype(np.float32).transpose(2, 0, 1))[None].cuda()
orig_forward = neck.forward
def run(noP3):
    f = {}
    def keep(n, x): x.retain_grad(); f[n] = x
    def fwd(inputs):
        C3, C4, C5 = list(inputs)[1:]
        keep("C3", C3); keep("C4", C4); keep("C5", C5)
        ac5 = neck.trans_head(C5); keep("C5+SA", ac5)
        p4 = neck.lateral_convs[1](C4) + F.interpolate(neck.lateral_convs[2](ac5), size=C4.shape[2:], **neck.upsample_cfg)
        keep("P4", p4)
        if noP3:
            d, pts = neck.def_convs[0](p4)
            return dict(features=(neck.fpn_convs[0](d),), aux_feat=p4, deform_points=(pts,))
        p3 = neck.lateral_convs[0](C3) + F.interpolate(p4, size=C3.shape[2:], **neck.upsample_cfg)
        d, pts = neck.def_convs[0](p3)
        return dict(features=(neck.fpn_convs[0](d),), aux_feat=p3, deform_points=(pts,))
    neck.forward = fwd
    hm = m.bbox_head.forward(m.extract_feat(t))["kpts_hm"][0]
    names = ["C3", "C4", "C5", "C5+SA", "P4"]
    A = [f[n] for n in names]
    g = torch.autograd.grad(hm[0][hm[0] > thr].sum(), A)
    act = {n: a[0].detach().norm(dim=0).cpu().numpy() for n, a in zip(names, A)}
    cam = {n: torch.relu((gg[0].mean(dim=(1, 2))[:, None, None] * a[0]).sum(0)).detach().cpu().numpy() for n, a, gg in zip(names, A, g)}
    return act, cam, hm
with torch.enable_grad():
    a0, c0, hm0 = run(False)
    # 확인: 직접 짠 P3 경로가 GANet 원래 forward 와 같은 conf 를 내는지
    neck.forward = orig_forward
    hm_ref = m.bbox_head.forward(m.extract_feat(t))["kpts_hm"][0]
    print("직접 짠 P3 경로 vs GANet 원래 forward conf 최대 차이:", float((hm0 - hm_ref).abs().max()))
    a1, c1, _ = run(True)
print(f"{'단계':6s} {'1줄 활성 최대차':>14s} {'2줄 실선기여도 상관':>16s}")
for n in a0:
    corr = np.corrcoef(c0[n].ravel(), c1[n].ravel())[0, 1]
    print(f"{n:6s} {np.abs(a0[n] - a1[n]).max():14.6f} {corr:16.3f}")
