"""P3del 구조에서 C5 attention 을 SA / DA(head4·8, pos add·query) / 없음 으로 바꿨을 때 params·속도·역전파 확인."""
import sys, time, copy
sys.path.insert(0, "./")
import mmcv, torch
from mmdet.models import build_detector
from projects.plugin import *  # noqa
base = mmcv.Config.fromfile("projects/cfgs/drone/res34_s16_full_2cls_P3del.py")
VARS = [("SA (P3del)", dict()), ("noSA", None),
        ("DA head4 add", dict(attn_type='deform', num_heads=4, pos_mode='add')),
        ("DA head8 add", dict(attn_type='deform', num_heads=8, pos_mode='add')),
        ("DA head8 query", dict(attn_type='deform', num_heads=8, pos_mode='query')),
        ("DA head8 pos없음", dict(attn_type='deform', num_heads=8, pos_shape=None))]
W, H = base.img_scale
x = torch.randn(1, 3, H, W).cuda()
for tag, extra in VARS:
    cfg = copy.deepcopy(base)
    cfg.model.backbone.init_cfg = None
    if extra is None:
        cfg.model.neck.trans_cfg = None
    else:
        cfg.model.neck.trans_cfg.update(extra)
    m = build_detector(cfg.model).cuda()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    na = sum(p.numel() for p in m.neck.trans_head.parameters()) / 1e6 if extra is not None else 0
    m.train()
    out = m.neck(m.backbone(x))
    sum(o.float().mean() for o in out if torch.is_tensor(o)).backward() if any(torch.is_tensor(o) for o in out) else None
    m.eval()
    with torch.no_grad():
        for _ in range(20): m.extract_feat(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(100): m.extract_feat(x)
        torch.cuda.synchronize()
    print(f"{tag:16s} 전체 {n:.3f}M, C5 attention 블록 {na*1e3:.1f}K, backbone+neck {(time.time()-t)*10:.2f}ms, 역전파 OK")
    del m; torch.cuda.empty_cache()
