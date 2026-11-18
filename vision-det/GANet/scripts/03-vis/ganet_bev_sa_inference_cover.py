"""1단계-A (사다리꼴을 도로 전체로): 학습 없이 추론 때만 'C5 -> BEV -> SA -> 원래 시점으로 되돌림 -> 기존 FPN' 으로 바꿔서 결과 비교.

1차 추론(기존)으로 검출한 선에서 프레임별 BEV 변환을 정하고(ganet_bev_feature_view.py 와 같은 방식),
2차 추론에서 trans_head 출력만 바꿔치기한다 (forward hook). 소스 코드는 건드리지 않는다.
  AC5' = unwarp( SA( warp(C5) ) ),  되돌렸을 때 BEV 화면 밖이라 값이 없는 칸은 기존 AC5 로 채움.
비교: 클래스별 키포인트 수(conf>0.4, NMS 후), 시작점 후보 수, 선 수/점 수, 버려진 점 수 + conf 맵/최종 검출 그림.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_sa_inference_cover.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_feature_view.py").read().split("os.makedirs(OUTDIR")[0])

kpt_thr = model.bbox_head.kpt_thr
CLS_NAME = ["실선", "점선"]
CLS_COLOR = ["#00dc00", "#dc0000"]  # ganet.py CLASS_COLORS 와 같은 색
STATE = dict(H=None, busy=False, filled=None)


def bev_homography_cover(lanes, kpts):
    """ganet_bev_feature_view 의 사다리꼴(같은 두 직선)을 쓰되, BEV 에서 차선 간격 g(화면 폭 비율)와
    사다리꼴 윗변 위치 y0 를 조절해서 모든 키포인트(버려진 점 포함)가 BEV 화면 안에 들어오는 설정 중 g 가 가장 큰 것."""
    _, src = bev_homography(lanes)
    best = None
    for g in [0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
        for y0 in [0.0, 0.2, 0.4]:
            dst = np.float32([[0.5 * W - g * W / 2, y0 * H], [0.5 * W + g * W / 2, y0 * H],
                              [0.5 * W + g * W / 2, H], [0.5 * W - g * W / 2, H]])
            Hm = cv2.getPerspectiveTransform(src, dst)
            q = cv2.perspectiveTransform(kpts[None].astype(np.float32), Hm)[0]
            inside = np.mean((q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H))
            if best is None or inside > best[0] + 1e-9:
                best = (inside, g, y0, Hm)
            if inside >= 0.98:
                return Hm, src, g, y0, inside
    return best[3], src, best[1], best[2], best[0]


GAMMA_LOG = []
for _l in neck.trans_head.attn_layers:
    def _g(m, inp, out):  # out 은 final_conv 전 값이 아니므로, 직접 다시 계산해서 γ·out / x 비율을 잰다
        x = m.pre_conv(inp[0])
        if len(inp) > 1 and inp[1] is not None:
            x = x + inp[1]
        b, _, h, w = x.shape
        q = m.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        k = m.key_conv(x).view(b, -1, h * w)
        a = m.softmax(torch.bmm(q, k)).permute(0, 2, 1)
        o = torch.bmm(m.value_conv(x).view(b, -1, h * w), a).view(b, -1, h, w)
        GAMMA_LOG.append(float((m.gamma * o).norm() / x.norm()))
    _l.register_forward_hook(_g)


def bev_sa_hook(m, inp, out):
    if STATE["H"] is None or STATE["busy"]:
        return None
    STATE["busy"] = True
    bc5, _ = warp_feat(inp[0], STATE["H"], 32)
    bac5 = m(bc5)
    STATE["busy"] = False
    back, valid = warp_feat(bac5, np.linalg.inv(STATE["H"]), 32)
    v = torch.from_numpy(valid).to(out.device)[None, None]
    STATE["filled"] = 1 - valid.mean()
    return torch.where(v, back, out)


neck.trans_head.register_forward_hook(bev_sa_hook)


def run(t):
    feats.clear()
    with torch.no_grad():
        x = model.extract_feat(t)
        hd = model.bbox_head.forward(x)
        pred = model.bbox_head.forward_test(x)
    return hd["kpts_hm"][0].cpu().numpy(), pred


def summarize(pred):
    rows, lanes = [], []
    for c, (_, cand, seeds) in enumerate(pred):
        n = len(seeds)
        if n == 0:
            rows.append((0, len(cand), [], 0)); continue
        groups, _ = group_points_fast(seeds, cand, pp.cluster_thr, pp.cluster_by_center_thr)
        kept = [g for g in groups if len(g) > pp.points_thr]
        used = sum(len(g) for g in groups)
        rows.append((n, len(cand), [len(g) for g in kept], n - used))
        for g in kept:
            pts = np.array([p[1] for p in g], float) * S + S / 2
            lanes.append((c, pts[np.argsort(pts[:, 1])]))
    return rows, lanes


for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    STATE["H"] = None
    GAMMA_LOG.clear()
    hm0, pred0 = run(t)
    kpts = np.concatenate([sd[:, 0] for _, _, sd in pred0 if len(sd)]) * S + S / 2
    Hm, src, g, y0, inside = bev_homography_cover(lanes_px(pred0), kpts)
    print(f"[프레임{name}] BEV 차선간격 {g:.2f}W, 윗변 y0={y0:.1f}H, 키포인트 BEV 안 {inside*100:.0f}%, "
          f"attention 기여 |γ·out|/|x| (1층, 2층) = {GAMMA_LOG[0]:.3f}, {GAMMA_LOG[1]:.3f}")
    GAMMA_LOG.clear()
    STATE["H"] = Hm
    hm1, pred1 = run(t)
    STATE["H"] = None
    r0, l0 = summarize(pred0)
    r1, l1 = summarize(pred1)
    print(f"   되돌린 AC5 중 기존 값으로 채운 칸 {STATE['filled']*100:.0f}%")
    for c in range(2):
        for tag, r in [("기존    ", r0[c]), ("BEV-SA  ", r1[c])]:
            print(f"   {CLS_NAME[c]} {tag}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, 선 {len(r[2])}개 {r[2]}, 버려진 점 {r[3]}")
        d = hm1[c] - hm0[c]
        print(f"   {CLS_NAME[c]} conf 변화: 평균 {d.mean():+.3f}, |변화| 최대 {np.abs(d).max():.2f}, "
              f"0.4 넘는 칸 {int((hm0[c] > kpt_thr).sum())} -> {int((hm1[c] > kpt_thr).sum())}")

    fig, axes = plt.subplots(2, 3, figsize=(21, 8.4), gridspec_kw=dict(hspace=0.1, wspace=0.03))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
    for r, (hm, lanes, tag) in enumerate([(hm0, l0, "기존"), (hm1, l1, "BEV-SA (학습 없음)")]):
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
        if r == 1:
            ax.add_patch(plt.Polygon(src, fill=False, ec="#00e5ff", lw=1.5, ls="--"))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
        axes[r, 0].set_ylabel(tag, fontsize=16)
    fig.suptitle(f"{name}-4. BEV에서 attention, 도로 전체 (학습 없이 추론만 변경) vs 기존\n"
                 "경로: C5를 BEV로 펴기(BC5, 도로 전체 범위) → attention(BAC5) → 원본 시점으로 되돌리기 → +C4 → +C3 → head", fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, :2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-4. BEV어텐션_무학습추론_도로전체.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
