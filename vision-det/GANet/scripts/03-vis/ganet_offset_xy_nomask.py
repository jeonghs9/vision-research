#!/usr/bin/env python3
"""1-4-2 오프셋맵을 conf 마스킹 없이 전체 칸(160x92) 그대로 그림. 검정 윤곽 = conf>0.4 칸(1-4-2에서 칠해진 곳).

실행 (GANet/source 에서):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_offset_xy_nomask.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_offset_xy_explained.py").read().split('\nif __name__ == "__main__":')[0])
OUTV = "/home/hsjeong/workspace/vision-det/GANet/test/tmp/view"
model, cfg = build()
W, H = tuple(cfg.img_scale)
mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
std = np.array(cfg.img_norm_cfg["std"], np.float32)
thr = cfg.model.head.kpt_thr
for prefix, path in JOBS:
    rs = cv2.resize(cv2.imread(path), (W, H))
    rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        hd = model.bbox_head.forward(model.extract_feat(x))
    conf = hd["kpts_hm"][0].cpu().numpy().max(0)
    off = hd["pts_offset"][0].cpu().numpy()
    mask = conf > thr
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.repeat((gray * 0.5 + 255 * 0.5).astype(np.uint8)[..., None], 3, axis=2)
    for nm, a in (("x", off[0]), ("y", off[1])):
        print(f"[프레임{prefix}] {nm}: 전체 범위 {a.min():.1f}~{a.max():.1f}칸, "
              f"conf>{thr} 칸 범위 {a[mask].min():.1f}~{a[mask].max():.1f}칸, 그 밖 칸 범위 {a[~mask].min():.1f}~{a[~mask].max():.1f}칸")

    fig, axes = plt.subplots(1, 3, figsize=(30, 6.6), gridspec_kw=dict(wspace=0.12))
    fig.suptitle(f"{prefix}-4-3. 오프셋맵 전체 칸 그대로 (conf로 거르지 않음, 검정 윤곽 = conf>{thr} 칸 = 1-4-2에서 칠해진 곳)", fontsize=17)
    axes[0].imshow(rgb); axes[0].axis("off"); axes[0].set_title("원본", fontsize=14)
    for ax, a, title, lab in ((axes[1], off[0], "x: 시작점이 옆으로 몇 칸?\n파랑 = 왼쪽   흰색 = 바로 아래   빨강 = 오른쪽", ("왼쪽", "오른쪽")),
                              (axes[2], off[1], "y: 시작점이 아래로 몇 칸?\n빨강 = 아래   흰색 = 여기   파랑 = 위", ("위", "아래"))):
        vmax = float(np.abs(a).max())
        ax.imshow(bg)
        im = ax.imshow(up(a), cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.9)
        ax.contour(up(mask.astype(np.float32)), levels=[0.5], colors="black", linewidths=0.8)
        ax.axis("off"); ax.set_title(title, fontsize=14)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, ticks=[-vmax, 0, vmax])
        cb.ax.set_yticklabels([f"{lab[0]} {vmax:.0f}칸", "0", f"{lab[1]} {vmax:.0f}칸"], fontsize=11)
    fig.savefig(f"{OUTV}/offnomask{prefix}.png", dpi=110, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
