"""차선 후보점 conf 표시: GANet과 같은 NMS(가로 3칸 최댓값)로 뽑은 점 중 conf>=0.1 전부.
노랑 = conf>0.4 (실제 키포인트), 빨강 = 0.1~0.4 (탈락, conf 값 표기). 실선 / 점선 두 칸.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_candidate_conf_points.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_sa_inference_cover.py").read().split("\nfor name, path in JOBS:")[0])

LOW = 0.1
OUT = "/home/hsjeong/workspace/vision-det/GANet/output/06-cls2-특징맵BEV무학습분석/figures"
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    with torch.no_grad():
        hmt = model.bbox_head.forward(model.extract_feat(t))["kpts_hm"]
        hmax = F.max_pool2d(hmt, (1, 3), stride=(1, 1), padding=(0, 1))   # ganet_head.py:141 과 동일
        nms = (hmt * (hmax == hmt).float())[0].cpu().numpy()
    bg = (rgb.astype(np.float32) / 255 * 0.55).clip(0, 1)
    ay, ax_ = np.where(nms.max(0) >= LOW)                       # 점 있는 영역만 잘라서 크게
    x0, x1 = max(ax_.min() * 8 - 40, 0), min(ax_.max() * 8 + 48, W)
    y0, y1 = max(ay.min() * 8 - 40, 0), min(ay.max() * 8 + 48, H)
    pw = 15
    fig, axes = plt.subplots(1, 2, figsize=(2 * pw, pw * (y1 - y0) / (x1 - x0) + 1.2), gridspec_kw=dict(wspace=0.03))
    for c in range(2):
        ax = axes[c]
        ax.imshow(bg)
        ys, xs = np.where(nms[c] >= LOW)
        v = nms[c, ys, xs]
        hi, lo = v > kpt_thr, v <= kpt_thr
        px, py = xs * 8 + 4, ys * 8 + 4
        ax.scatter(px[hi], py[hi], s=22, c="#ffd400", edgecolors="black", linewidths=0.4, zorder=3)
        ax.scatter(px[lo], py[lo], s=22, c="#ff2020", edgecolors="black", linewidths=0.4, zorder=4)
        for x, y, val in zip(px[lo], py[lo], v[lo]):
            ax.text(x + 5, y - 2, f"{val:.2f}", fontsize=7.5, color="#ff5050", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.08", fc="black", ec="none", alpha=0.55))
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{CLS_NAME[c]} conf — 노랑(>{kpt_thr}) {int(hi.sum())}점, 빨강({LOW}~{kpt_thr}) {int(lo.sum())}점", fontsize=16)
        print(f"[프레임{name}] {CLS_NAME[c]}: 노랑 {int(hi.sum())}, 빨강 {int(lo.sum())}")
    fig.subplots_adjust(top=1 - 0.75 / fig.get_figheight())
    fig.suptitle(f"{name}-11. 차선 후보점 conf (GANet NMS 가로 3칸 최댓값, conf {LOW} 이상) — 노랑 = {kpt_thr} 초과(키포인트), 빨강 = {kpt_thr} 이하(탈락, 값 표기)",
                 fontsize=18)
    fig.savefig(f"{OUT}/{name}-11. 후보점_conf_0.4기준.png", dpi=130, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
