"""기존 3종(P3del+SA / +DA / −SA) vs APT 3종: backbone+neck params, batch1 속도(같은 val 이미지)."""
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
for p in ['res34_s16_full_2cls_P3del', 'res34_s16_full_2cls_P3del_APT', 'res34_s16_full_2cls_P3del_DAq',
          'res34_s16_full_2cls_P3del_DAq_APT', 'res34_s16_full_2cls_P3del_noSA', 'res34_s16_full_2cls_P3del_noSA_APT']:
    m = build_detector(cfg(p).model).cuda().eval()
    n = sum(v.numel() for k, v in m.named_parameters() if not k.startswith('bbox_head')) / 1e6
    ts = []
    with torch.no_grad():
        for _ in range(20): m.extract_feat(x)
        for _ in range(5):
            torch.cuda.synchronize(); t = time.time()
            for _ in range(100): m.extract_feat(x)
            torch.cuda.synchronize(); ts.append((time.time() - t) * 10)
    print(f'{p:40s} {n:.3f}M  {min(ts):.2f}ms')
