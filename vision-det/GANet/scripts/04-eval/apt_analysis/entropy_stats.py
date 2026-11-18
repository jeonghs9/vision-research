"""학습 이미지(val 파이프라인 = 무작위 증강 없음)에서 64x64 패치 엔트로피 분포와 문턱값별 병합 비율.
실행: cd GANet/source && PYTHONNOUSERSITE=1 python /home/hsjeong/workspace/vision-det/GANet/scripts/04-eval/apt_analysis/entropy_stats.py"""
import sys; sys.path.insert(0, './')
import numpy as np, torch
from mmcv import Config
from mmdet.datasets import build_dataset
import projects.plugin  # noqa
from projects.plugin.models.dcn_fpn import apt_patch_entropy

cfg = Config.fromfile('projects/cfgs/drone/res34_s16_full_2cls_P3del.py')
tr = cfg.data.train.copy(); tr['pipeline'] = cfg.val_pipeline
ds = build_dataset(tr)
rng = np.random.RandomState(0)
ids = rng.choice(len(ds), min(400, len(ds)), replace=False)
m, s = cfg.img_norm_cfg['mean'], cfg.img_norm_cfg['std']
ents = []
for i in ids:
    d = ds[i]['img']; d = d[0] if isinstance(d, list) else d; img = d.data.unsqueeze(0).cuda()
    e = apt_patch_entropy(img, 64, m, s)[0]
    ents.append(e[e < 1e5].flatten().cpu())
e = torch.cat(ents).numpy()
print('train images', len(ds), 'sampled', len(ids), 'img', tuple(img.shape))
print('64x64 entropy percentiles 5/25/50/75/95:', np.percentile(e, [5, 25, 50, 75, 95]).round(2))
n_fine = 23 * 40
n_groups_full = 11 * 20  # 마지막(홀수) 줄 제외한 2x2 묶음
for t in [2.0, 3.0, 4.0, 4.5, 5.0, 5.75, 6.0]:
    f = (e < t).mean()
    merged = f * n_groups_full
    tokens = n_fine - 4 * merged + merged
    print(f'tau={t:5.2f}  병합 묶음 {f*100:5.1f}%  토큰 {tokens:6.1f}/920  감소 {100*(1-tokens/n_fine):5.1f}%')
