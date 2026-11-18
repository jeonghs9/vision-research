"""GANet 2클래스 실험 11종 params / 네트워크 시간(backbone+neck+head forward, 후처리 제외) — 같은 GPU·같은 이미지·같은 방법.
실행: cd GANet/source && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python /home/hsjeong/workspace/vision-det/GANet/scripts/05-bench/all_speed_bench.py"""
import sys, time; sys.path.insert(0, './')
import torch
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
import projects.plugin  # noqa
def cfg(p):
    for k in [k for k in sys.modules if k.startswith('projects.cfgs')]: del sys.modules[k]
    return Config.fromfile(f'projects/cfgs/drone/{p}.py')
c0 = cfg('res34_s16_full_2cls_P3del'); d = c0.data.val.copy(); d['pipeline'] = c0.val_pipeline
x = build_dataset(d)[0]['img']; x = (x[0] if isinstance(x, list) else x).data.unsqueeze(0).cuda()
for e, p in [('003', 'res18_s8_full_2cls'), ('004', 'res34_s8_full_2cls'), ('005', 'res18_s16_full_2cls_P3del'),
             ('006', 'res34_s16_full_2cls_P3del'), ('007', 'res34_s8_full_2cls_noSA'), ('008', 'res34_s16_full_2cls_P3del_noSA'),
             ('009', 'res34_s16_full_2cls_P3del_DAq'), ('010', 'res34_s16_full_2cls_P3del_DAadd'), ('011', 'res34_s16_full_2cls_P3del_APT'),
             ('012', 'res34_s16_full_2cls_P3del_DAq_APT'), ('013', 'res34_s16_full_2cls_P3del_noSA_APT')]:
    c = cfg(p); c.model.backbone.init_cfg = None
    m = build_detector(c.model).cuda().eval()
    n = sum(v.numel() for v in m.parameters()) / 1e6
    f = lambda: m.bbox_head.forward(m.extract_feat(x))
    ts = []
    with torch.no_grad():
        for _ in range(30): f()
        for _ in range(5):
            torch.cuda.synchronize(); t = time.time()
            for _ in range(100): f()
            torch.cuda.synchronize(); ts.append((time.time() - t) * 10)
    print(f'{e} {p:38s} {n:.2f}M {min(ts):.2f}ms', flush=True)
    del m; torch.cuda.empty_cache()
