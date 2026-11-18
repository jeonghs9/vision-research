"""res34 기존(P3) vs res34 P3del 학습 모델 — 원본 입력 / BEV 입력 최종 검출을 한 장에 (2단계 7번 그림).
줄: 원본·기존 / 원본·P3del / BEV·기존 / BEV·P3del,  칸: 실선 conf, 점선 conf, 최종 검출.
모델·검출은 ganet_p3del_trained_compare.py(3번 그림), BEV 사다리꼴은 ganet_p3del_trained_compare_bev.py(6번 그림)와 같다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_orig_vs_bev_detection.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_trained_compare.py").read().split("\nos.makedirs(OUTDIR")[0])

os.makedirs(OUTDIR, exist_ok=True)
_src = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])
_dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
M_bev = cv2.getPerspectiveTransform(_src, _dst)
for name, path in JOBS:
    orig = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    bev = cv2.warpPerspective(orig, M_bev, (W, H))
    res = []
    for itag, rgb in [("원본", orig), ("BEV", bev)]:
        t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
        for tag, m, _ in models:
            hm, rows, lanes = detect(m, t)
            res.append((f"{itag} 입력\n{tag}", rgb, hm, rows, lanes))
    print(f"[프레임{name}]")
    for tag, _, hm, rows, _ in res:
        print(f"   {tag.replace(chr(10), ' '):34s}: " + ", ".join(
            f"{CLS_NAME[c]} 선 {len(rows[c][2])}개 {rows[c][2]} 버려진 점 {rows[c][3]}" for c in range(2)))

    n = len(res)
    fig, axes = plt.subplots(n, 3, figsize=(21, 4.2 * n), gridspec_kw=dict(hspace=0.1, wspace=0.03))
    for r, (tag, rgb, hm, _, lanes) in enumerate(res):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
        for c in range(2):
            ax = axes[r, c]
            ax.imshow(bg)
            ax.imshow(cv2.resize(hm[c], (W, H), interpolation=cv2.INTER_NEAREST), cmap="jet", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{CLS_NAME[c]} conf", fontsize=15)
        axes[r, 0].set_ylabel(tag, fontsize=15)
        ax = axes[r, 2]
        ax.imshow(rgb)
        for c, pts in lanes:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=CLS_COLOR[c], lw=3)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
    fig.subplots_adjust(top=1 - 1.2 / (4.2 * n))
    fig.suptitle(f"{name}-7. res34 기존 vs P3del 학습 모델의 최종 검출 결과 — 원본 입력(위 2줄) / BEV 입력(아래 2줄)", fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, 0:2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-7. 최종검출결과_원본vsBEV입력_P3del학습.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
