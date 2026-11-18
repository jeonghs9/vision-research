import sys, time; sys.path.insert(0, "./")
import mmcv, torch
from mmdet.models import build_detector
from projects.plugin import *  # noqa
for c in ["res34_s8_full_2cls", "res34_s8_full_2cls_noSA", "res34_s16_full_2cls_P3del", "res34_s16_full_2cls_P3del_noSA"]:
    cfg = mmcv.Config.fromfile(f"projects/cfgs/drone/{c}.py"); cfg.model.train_cfg = None; cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model, test_cfg=cfg.get("test_cfg")).cuda().eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    W, H = cfg.img_scale; x = torch.randn(1, 3, H, W).cuda()
    with torch.no_grad():
        for _ in range(20): m.bbox_head.forward(m.extract_feat(x))
        torch.cuda.synchronize(); t = time.time()
        for _ in range(100): out = m.bbox_head.forward(m.extract_feat(x))
        torch.cuda.synchronize()
    print(f"{c}: params {n:.2f}M, 네트워크 {1000*(time.time()-t)/100:.1f} ms, conf map {tuple(out['kpts_hm'].shape[2:])}")
