"""1단계-③ 두 경로 합치기: 기존 경로(AC5 -> C4 -> C3) 출력 + 전체 BEV 경로(BAC5 -> BC4 -> BC3, 되돌린 것) 출력 (학습 없이).

head 가 쓰는 features[0], aux_feat 두 개에 대해 (ganet_head.py:103-108):
  더하기: P3 = P3_기존 + unwarp(P3_BEV)
  평균  : P3 = (P3_기존 + unwarp(P3_BEV)) / 2      # 학습 때와 값 크기를 맞춘 버전
  conf만 더하기: features[0] 만 더하기, aux_feat(offset 쪽)은 기존 그대로
  BEV 화면 밖이라 BEV 값이 없는 칸은 P3_기존 만 씀.
전체 BEV 경로 계산은 ganet_bev_full_path.py 의 full_bev_neck 그대로.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_dual_fuse.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_full_path.py").read().split("\n# 재구성 검증")[0])

FUSE = dict(H=None, w=None)


def fuse_hook(m, inp, out):
    if FUSE["H"] is None:
        return None
    Hm = FUSE["H"]
    p3, aux = full_bev_neck(inp[0], Hm)
    Hinv = np.linalg.inv(Hm)
    p3b, valid = warp_feat(p3, Hinv, 8)
    auxb, _ = warp_feat(aux, Hinv, 8)
    v = torch.from_numpy(valid).to(p3.device)[None, None]
    a = 1.0 if FUSE["w"] in ("sum", "sumconf") else 0.5
    o = dict(out)
    f0 = out["features"][0]
    o["features"] = (torch.where(v, a * f0 + a * p3b if a == 0.5 else f0 + p3b, f0),) + tuple(out["features"][1:])
    if FUSE["w"] == "sumconf":
        return o
    o["aux_feat"] = torch.where(v, a * out["aux_feat"] + a * auxb if a == 0.5 else out["aux_feat"] + auxb, out["aux_feat"])
    return o


neck.register_forward_hook(fuse_hook)

MODES = [("sum", "더하기\n기존 + 전체BEV"), ("mean", "평균\n(기존 + 전체BEV)/2"), ("sumconf", "conf만 더하기\n(offset은 기존)")]
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    STATE["H"] = None; FULL["H"] = None; FUSE["H"] = None
    hm0, pred0 = run(t)
    kpts = np.concatenate([sd[:, 0] for _, _, sd in pred0 if len(sd)]) * S + S / 2
    Hm, src, g, y0, inside = bev_homography_cover(lanes_px(pred0), kpts)
    res = [(hm0, *summarize(pred0), "기존")]
    for w, tag in MODES:
        FUSE["H"] = Hm; FUSE["w"] = w
        hm, pred = run(t)
        FUSE["H"] = None
        res.append((hm, *summarize(pred), tag))
    print(f"[프레임{name}] 키포인트 BEV 안 {inside*100:.0f}%")
    for c in range(2):
        for hm, rows, _, tag in res:
            r = rows[c]
            print(f"   {CLS_NAME[c]} {tag.splitlines()[0]:4s}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, 선 {len(r[2])}개 {r[2]}, "
                  f"버려진 점 {r[3]}, 0.4 넘는 칸 {int((hm[c] > kpt_thr).sum())}, |conf 변화| 최대 {np.abs(hm[c] - hm0[c]).max():.2f}")

    fig, axes = plt.subplots(4, 3, figsize=(21, 16.8), gridspec_kw=dict(hspace=0.1, wspace=0.03))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
    for r, (hm, _, lanes, tag) in enumerate(res):
        for c in range(2):
            ax = axes[r, c]
            ax.imshow(bg)
            ax.imshow(cv2.resize(hm[c], (W, H), interpolation=cv2.INTER_NEAREST), cmap="jet", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{CLS_NAME[c]} conf", fontsize=15)
        ax = axes[r, 2]
        ax.imshow(rgb)
        for c, pts in lanes:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=CLS_COLOR[c], lw=3)
        if r > 0:
            ax.add_patch(plt.Polygon(src, fill=False, ec="#00e5ff", lw=1.5, ls="--"))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
        axes[r, 0].set_ylabel(tag, fontsize=16)
    fig.subplots_adjust(top=1 - 1.7 / 16.8)
    fig.suptitle(f"{name}-7. 두 경로 합치기 (학습 없이 추론만 변경) vs 기존\n"
                 "경로: [AC5 → +C4 → +C3] 와 [BAC5 → +BC4 → +BC3 → 원본 시점으로 되돌리기] 를 head 직전에 더하기 / 평균 / conf 쪽만 더하기", fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, :2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-7. 두경로합치기_무학습추론.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
