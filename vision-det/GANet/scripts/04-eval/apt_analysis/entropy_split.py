"""train / val 각각 64x64 엔트로피 문턱값별 병합 비율 (val 파이프라인, 무작위 증강 없음)."""
import sys; sys.path.insert(0, './')
import numpy as np, torch
from mmcv import Config
from mmdet.datasets import build_dataset
import projects.plugin  # noqa
from projects.plugin.models.dcn_fpn import apt_patch_entropy
cfg = Config.fromfile('projects/cfgs/drone/res34_s16_full_2cls_P3del.py')
m, s = cfg.img_norm_cfg['mean'], cfg.img_norm_cfg['std']
for split in ['train', 'val']:
    d = cfg.data[split].copy(); d['pipeline'] = cfg.val_pipeline; ds = build_dataset(d)
    per_img = {t: [] for t in [4.0, 5.0, 5.75, 6.0, 6.5, 7.0]}
    allv = []
    for i in range(len(ds)):
        x = ds[i]['img']; x = x[0] if isinstance(x, list) else x
        e = apt_patch_entropy(x.data.unsqueeze(0).cuda(), 64, m, s)[0][:11]  # 마지막(pad) 줄 제외
        allv.append(e.flatten().cpu())
        for t in per_img: per_img[t].append((e < t).float().mean().item())
    a = torch.cat(allv).numpy()
    print(f'[{split}] {len(ds)}장  엔트로피 분위 5/25/50/75/95 {np.percentile(a,[5,25,50,75,95]).round(2)}')
    for t, v in per_img.items():
        v = np.array(v); red = 3 * v * 220 / 920
        print(f'   tau={t:4.2f} 병합 묶음 평균 {v.mean()*100:5.1f}% (이미지별 중앙 {np.median(v)*100:5.1f}%, 0인 이미지 {(v==0).mean()*100:4.0f}%)  토큰 감소 평균 {red.mean()*100:4.1f}%')
