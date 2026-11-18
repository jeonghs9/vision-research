"""1단계-① cross-attention: Q = 원본 시점 C5, K·V = BEV 로 편 BC5 (학습 없이, 지금 가중치 그대로).

각 attention 층 (dcn_fpn.py AttentionLayer 의 학습된 부품을 그대로 사용, 계산 순서도 동일):
  xq  = pre_conv(입력) + pos            # 원본 시점
  xkv = pre_conv(warp(입력)) + pos       # BEV (pos 는 같은 23x40 격자 임베딩)
  A   = softmax(Q(xq)ᵀ K(xkv))           # BEV 화면 밖 칸은 key 에서 제외
  out = V(xkv)·Aᵀ,  y = final_conv(γ·out + xq)     -> 원본 시점 그대로라 되돌리기 없음
2층은 1층 출력 y 를 다시 BEV 로 펴서 K·V 로 쓴다.
BEV 변환은 ganet_bev_sa_inference_cover.py 의 '도로 전체' 변환(A)과 같다.
검증: 항등 변환을 넣으면 원래 AC5 와 같아야 한다 (재구성 확인).

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_cross_attention.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_sa_inference_cover.py").read().split("\nfor name, path in JOBS:")[0])

th = neck.trans_head
CROSS = dict(H=None, busy=False)


def cross_layer(m, xin, pos, Hm):
    xq = m.pre_conv(xin) + pos
    xb, valid = warp_feat(xin, Hm, 32)
    xkv = m.pre_conv(xb) + pos
    b, _, h, w = xq.shape
    q = m.query_conv(xq).view(b, -1, h * w).permute(0, 2, 1)
    k = m.key_conv(xkv).view(b, -1, h * w)
    energy = torch.bmm(q, k)
    vmask = torch.from_numpy(valid.ravel()).to(energy.device)
    energy = energy.masked_fill(~vmask[None, None, :], float("-inf"))
    a = m.softmax(energy).permute(0, 2, 1)
    out = torch.bmm(m.value_conv(xkv).view(b, -1, h * w), a).view(b, -1, h, w)
    return m.final_conv(m.gamma * out + xq)


def cross_forward(c5, Hm):
    x = c5
    for layer, pos in zip(th.attn_layers, th.pos_embeds):
        x = cross_layer(layer, x, pos.to(x.device), Hm)
    return x


def cross_hook(m, inp, out):
    if CROSS["H"] is None or CROSS["busy"]:
        return None
    CROSS["busy"] = True
    y = cross_forward(inp[0], CROSS["H"])
    CROSS["busy"] = False
    return y


th.register_forward_hook(cross_hook)

# 재구성 검증: 항등 변환이면 원래 self-attention 과 같아야 함
_t = torch.from_numpy(((cv2.resize(cv2.cvtColor(cv2.imread(JOBS[0][1]), cv2.COLOR_BGR2RGB), (W, H)).astype(np.float32)
                        - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
feats.clear()
with torch.no_grad():
    model.extract_feat(_t)
    ref = feats["AC5"]
    chk = cross_forward(feats["C5"], np.eye(3))
print(f"[검증] 항등 변환 cross vs 원래 AC5: 최대 차이 {float((chk - ref).abs().max()):.2e} (AC5 크기 {float(ref.abs().max()):.1f})")

rows_all = {}
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    STATE["H"] = None; CROSS["H"] = None; GAMMA_LOG.clear()
    hm0, pred0 = run(t)
    kpts = np.concatenate([sd[:, 0] for _, _, sd in pred0 if len(sd)]) * S + S / 2
    Hm, src, g, y0, inside = bev_homography_cover(lanes_px(pred0), kpts)
    STATE["H"] = Hm
    hmA, predA = run(t)
    STATE["H"] = None
    CROSS["H"] = Hm
    hmX, predX = run(t)
    CROSS["H"] = None
    r0, l0 = summarize(pred0)
    rA, _ = summarize(predA)
    rX, lX = summarize(predX)
    print(f"[프레임{name}] BEV 차선간격 {g:.2f}W, 윗변 y0={y0:.1f}H, 키포인트 BEV 안 {inside*100:.0f}%")
    for c in range(2):
        for tag, r in [("기존        ", r0[c]), ("A BEV-SA    ", rA[c]), ("① cross-attn", rX[c])]:
            print(f"   {CLS_NAME[c]} {tag}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, 선 {len(r[2])}개 {r[2]}, 버려진 점 {r[3]}")
        d = hmX[c] - hm0[c]
        print(f"   {CLS_NAME[c]} cross conf 변화: |변화| 최대 {np.abs(d).max():.2f}, "
              f"0.4 넘는 칸 {int((hm0[c] > kpt_thr).sum())} -> {int((hmX[c] > kpt_thr).sum())}")

    fig, axes = plt.subplots(2, 3, figsize=(21, 8.4), gridspec_kw=dict(hspace=0.1, wspace=0.03))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
    for r, (hm, lanes, tag) in enumerate([(hm0, l0, "기존"), (hmX, lX, "cross-attention\nQ=C5, K·V=BC5")]):
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
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
        axes[r, 0].set_ylabel(tag, fontsize=16)
    fig.suptitle(f"{name}-5. cross-attention (Q=원본 C5, K·V=BEV C5, 학습 없이 추론만 변경) vs 기존\n"
                 "경로: attention의 Q는 원본 C5, K·V는 BEV로 편 BC5 → AC5(원본 시점, 되돌리기 없음) → +C4 → +C3 → head", fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, :2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-5. 크로스어텐션_무학습추론.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
