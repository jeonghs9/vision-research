"""1단계-④ 두 경로를 '학습 가중치 없는 방식'으로 합치기 (concat 뒤 새 conv/attention 층 대신, 학습 없이).

P = 기존 경로 P3 (AC5 -> +C4 -> +C3),  B = 전체 BEV 경로 P3 (BAC5 -> +BC4 -> +BC3, 원본 시점으로 되돌린 것)
합치는 건 conf 쪽(features[0]) 만. offset 쪽(aux_feat)은 기존 그대로 (ganet_bev_dual_fuse.py 에서 offset 을 건드리면 선이 깨짐).
  ① 칸별 max     : F = max(P, B)  (채널마다 더 큰 값)
  ② conf 기준 선택: 각 경로 conf 로짓 s_P, s_B (keypts_head, 클래스 중 최대) -> w = softmax([s_P, s_B]),  F = w_P·P + w_B·B
  ③ 가중치 없는 cross-attention: Q = P, K = V = B (BEV 화면 밖 칸 제외), A = softmax(QᵀK/√64),  F = P + V·Aᵀ
  ④ 출력에서 합치기: conf = max(head(P), head(B)) -- 특징이 아니라 conf 맵끼리 칸별 max
BEV 화면 밖이라 B 값이 없는 칸은 P 만 씀.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_dual_paramfree.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_full_path.py").read().split("\n# 재구성 검증")[0])

kh = model.bbox_head.keypts_head
PF = dict(H=None, mode=None, B=None, v=None, busy=False)


def logit_max(f):
    PF["busy"] = True
    s = kh(f)["hm"].max(1, keepdim=True)[0]
    PF["busy"] = False
    return s


def pf_neck_hook(m, inp, out):
    if PF["H"] is None:
        return None
    p3, _ = full_bev_neck(inp[0], PF["H"])
    B, valid = warp_feat(p3, np.linalg.inv(PF["H"]), 8)
    v = torch.from_numpy(valid).to(B.device)[None, None]
    P = out["features"][0]
    mode = PF["mode"]
    if mode == "late":
        PF["B"], PF["v"] = B, v
        return None
    if mode == "max":
        Fm = torch.max(P, B)
    elif mode == "gate":
        w = torch.softmax(torch.cat([logit_max(P), logit_max(B)], 1), 1)
        Fm = w[:, :1] * P + w[:, 1:] * B
    elif mode == "xattn":
        b, c, h, wd = P.shape
        q = P.view(b, c, -1).permute(0, 2, 1)
        k = B.view(b, c, -1)
        e = torch.bmm(q, k) / c ** 0.5
        e = e.masked_fill(~v.view(1, 1, -1), float("-inf"))
        a = torch.softmax(e, -1)
        Fm = P + torch.bmm(k, a.permute(0, 2, 1)).view(b, c, h, wd)
    o = dict(out)
    o["features"] = (torch.where(v, Fm, P),) + tuple(out["features"][1:])
    return o


def late_hook(m, inp, out):
    if PF["mode"] != "late" or PF["H"] is None or PF["busy"]:
        return None
    PF["busy"] = True
    hb = kh(PF["B"])["hm"]
    PF["busy"] = False
    return dict(out, hm=torch.where(PF["v"], torch.max(out["hm"], hb), out["hm"]))


neck.register_forward_hook(pf_neck_hook)
kh.register_forward_hook(late_hook)

MODES = [("max", "① 칸별 max\nmax(P, B)"), ("gate", "② conf 기준 선택\nsoftmax(conf)로 P·B 섞기"),
         ("xattn", "③ 가중치 없는 cross-attn\nP + attn(Q=P, K·V=B)"), ("late", "④ 출력에서 합치기\nmax(conf_P, conf_B)")]
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    STATE["H"] = None; FULL["H"] = None; PF["H"] = None
    hm0, pred0 = run(t)
    kpts = np.concatenate([sd[:, 0] for _, _, sd in pred0 if len(sd)]) * S + S / 2
    Hm, src, g, y0, inside = bev_homography_cover(lanes_px(pred0), kpts)
    res = [(hm0, *summarize(pred0), "기존 (P)")]
    for mode, tag in MODES:
        PF["H"] = Hm; PF["mode"] = mode
        hm, pred = run(t)
        PF["H"] = None
        res.append((hm, *summarize(pred), tag))
    print(f"[프레임{name}] 키포인트 BEV 안 {inside*100:.0f}%")
    for c in range(2):
        for hm, rows, _, tag in res:
            r = rows[c]
            print(f"   {CLS_NAME[c]} {tag.splitlines()[0]:22s}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, 선 {len(r[2])}개 {r[2]}, "
                  f"버려진 점 {r[3]}, 0.4 넘는 칸 {int((hm[c] > kpt_thr).sum())}, |conf 변화| 최대 {np.abs(hm[c] - hm0[c]).max():.2f}")

    n = len(res)
    fig, axes = plt.subplots(n, 3, figsize=(21, 4.2 * n), gridspec_kw=dict(hspace=0.1, wspace=0.03))
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
        axes[r, 0].set_ylabel(tag, fontsize=15)
    fig.subplots_adjust(top=1 - 1.7 / (4.2 * n))
    fig.suptitle(f"{name}-8. 두 경로를 학습 가중치 없이 합치기 (학습 없이 추론만 변경) vs 기존\n"
                 "경로: P=[AC5 → +C4 → +C3], B=[BAC5 → +BC4 → +BC3 → 원본 시점으로 되돌리기] → conf 쪽만 합침(offset은 기존) → head",
                 fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, :2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-8. 두경로_가중치없이합치기_무학습추론.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
