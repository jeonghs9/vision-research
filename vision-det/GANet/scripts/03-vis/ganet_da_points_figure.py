"""DA(Deformable Attention)가 C5 한 칸에서 '몇 곳을 보는지' 설명 그림.

(a) num_points=4 학습 시작 때 위치 (mmcv MultiScaleDeformableAttention.init_weights 그대로: head h 방향 θ=h·45°, point i 는 i+1 칸)
(b) num_points=8 학습 시작 때 위치
(c) 학습된 test/009(DAq, ep150) 1층이 val 이미지 한 장의 차선 위 칸에서 실제로 보는 32곳 (점 크기 = 가중치)

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_da_points_figure.py
"""
import math
import sys
sys.path.insert(0, './')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
import projects.plugin  # noqa

G = '/home/hsjeong/workspace/vision-det/GANet'
OUT = f'{G}/output/07-cls2-P3제거특징맵분석/figures/DA-1. 샘플링점_설명_point4vs8.png'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
HEADS = 8
COLORS = plt.cm.tab10(np.arange(HEADS))
H, W = 23, 40  # C5 격자


def init_offsets(num_points):
    """mmcv init_weights 와 같은 초기 offset (칸 단위). 반환 heads x points x 2 (x, y)."""
    th = torch.arange(HEADS, dtype=torch.float32) * (2 * math.pi / HEADS)
    g = torch.stack([th.cos(), th.sin()], -1)
    g = g / g.abs().max(-1, keepdim=True)[0]
    return np.stack([g.numpy() * (i + 1) for i in range(num_points)], 1)


def draw_init(ax, num_points, R=9):
    off = init_offsets(num_points)
    for v in np.arange(-R - 0.5, R + 1):
        ax.axhline(v, color='0.85', lw=0.6, zorder=0)
        ax.axvline(v, color='0.85', lw=0.6, zorder=0)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, color='k', zorder=3))
    for h in range(HEADS):
        pts = off[h]
        ax.plot([0, pts[-1, 0]], [0, pts[-1, 1]], color=COLORS[h], lw=1.2, alpha=0.5, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=70, color=COLORS[h], edgecolor='k', lw=0.5, zorder=4)
        for i, (x, y) in enumerate(pts):
            if num_points <= 4 or i in (0, num_points - 1):
                ax.text(x, y, str(i + 1), fontsize=6.5, ha='center', va='center', color='w', zorder=5, fontweight='bold')
        lx, ly = pts[-1] * (1 + 0.9 / np.abs(pts[-1]).max())
        ax.text(lx, ly, f'head{h + 1}', fontsize=8, ha='center', va='center', color=COLORS[h], fontweight='bold')
    ax.set_xlim(-R - 0.5, R + 0.5)
    ax.set_ylim(R + 0.5, -R - 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    n = HEADS * num_points
    ax.set_title(f'num_points={num_points}: 학습 시작 때 보는 곳\n{HEADS} head × {num_points} point = {n}곳 '
                 f'(가장 먼 곳 {num_points}칸 = 입력 {num_points * 32}px)', fontsize=11)


def learned_points():
    for k in [k for k in sys.modules if k.startswith('projects.cfgs')]:
        del sys.modules[k]
    cfg = Config.fromfile('projects/cfgs/drone/res34_s16_full_2cls_P3del_DAq.py')
    model = build_detector(cfg.model).cuda().eval()
    sd = torch.load(f'{G}/test/009-res34-2cls-1280-P3del-DAq-150ep/epoch_150.pth', map_location='cpu')['state_dict']
    print('strict load', model.load_state_dict(sd, strict=True))
    d = cfg.data.val.copy()
    d['pipeline'] = cfg.val_pipeline
    item = build_dataset(d)[0]
    img = item['img']
    img = (img[0] if isinstance(img, list) else img).data.unsqueeze(0).cuda()
    meta = item['img_metas']
    meta = (meta[0] if isinstance(meta, list) else meta).data
    attn = model.neck.trans_head.attn_layers[0].attn
    cap = {}

    def hook(m, args, kwargs=None):
        cap['query'] = args[0]

    h = attn.register_forward_pre_hook(hook)
    # pre-hook 은 위치 인자만 받으므로 query_pos 는 같은 방식으로 다시 만든다
    with torch.no_grad():
        model.extract_feat(img)
    h.remove()
    pos = model.neck.trans_head.pos_embeds[0].cuda().flatten(2).transpose(1, 2)
    q = cap['query'] + pos
    with torch.no_grad():
        off = attn.sampling_offsets(q).view(1, H * W, HEADS, 1, 4, 2)[0, :, :, 0]  # N x heads x P x 2 (칸 단위)
        aw = attn.attention_weights(q).view(1, H * W, HEADS, 4).softmax(-1)[0]
    x = img[0].permute(1, 2, 0).cpu().numpy() * np.array(cfg.img_norm_cfg['std']) + np.array(cfg.img_norm_cfg['mean'])
    x = np.clip(x, 0, 255).astype(np.uint8)
    # 차선 위 칸: 첫 GT 차선(입력 좌표)의 가운데 점
    gt = np.array(meta['gt_points'][0]).reshape(-1, 2)
    px, py = gt[len(gt) // 2]
    cx, cy = int(px // 32), int(py // 32)
    n = cy * W + cx
    return x, (cx, cy), off[n].cpu().numpy(), aw[n].cpu().numpy(), gt


def draw_learned(ax, data):
    x, (cx, cy), off, aw, gt = data
    ax.imshow(x)
    for v in range(0, 1281, 32):
        ax.axvline(v, color='w', lw=0.25, alpha=0.4)
    for v in range(0, 737, 32):
        ax.axhline(v, color='w', lw=0.25, alpha=0.4)
    ax.add_patch(plt.Rectangle((cx * 32, cy * 32), 32, 32, fill=False, ec='yellow', lw=2.5, zorder=5))
    c = np.array([(cx + 0.5) * 32, (cy + 0.5) * 32])
    for hh in range(HEADS):
        p = c + off[hh] * 32  # offset 는 칸 단위 -> 입력 픽셀
        for i in range(4):
            ax.plot([c[0], p[i, 0]], [c[1], p[i, 1]], color=COLORS[hh], lw=0.6, alpha=0.6)
        ax.scatter(p[:, 0], p[:, 1], s=20 + 400 * aw[hh], color=COLORS[hh], edgecolor='k', lw=0.6, zorder=6)
    allp = c + off.reshape(-1, 2) * 32
    lo = np.minimum(allp.min(0), c - 64) - 48
    hi = np.maximum(allp.max(0), c + 64) + 48
    ax.set_xlim(max(lo[0], 0), min(hi[0], 1280))
    ax.set_ylim(min(hi[1], 736), max(lo[1], 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('학습 후(test/009, 1층): 노란 칸이 실제로 보는 32곳\n색 = head, 점 크기 = 그 head 안에서의 가중치', fontsize=11)


fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
draw_init(axes[0], 4)
draw_init(axes[1], 8, R=10)
draw_learned(axes[2], learned_points())
for ax, t in zip(axes, ['(a)', '(b)', '(c)']):
    ax.text(0.01, 0.99, t, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print('saved', OUT)
