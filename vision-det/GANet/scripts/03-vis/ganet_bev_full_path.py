"""1단계-② 전체 BEV 경로: BAC5 -> BC4 -> BC3 을 전부 BEV 좌표에서 합치고, head 직전에 한 번만 원본 시점으로 되돌림 (학습 없이).

neck(dcn_fpn.py DeformFPN.forward) 계산 순서를 그대로 따라가되 입력만 BEV 로 바꾼다:
  BC3, BC4, BC5 = warp(C3, s=8), warp(C4, s=16), warp(C5, s=32)     # 같은 도로 BEV 변환(A)
  BAC5 = trans_head(BC5)                                            # BEV 에서 self-attention
  P5 = lat(BAC5);  P4 = lat(BC4) + up(P5);  P3 = lat(BC3) + up(P4)  # 전부 BEV 좌표
  aux_feat = P3 (DCN 전),  features[0] = fpn_conv(P3 + DCN(P3))      # BEV 좌표
  head 에 넣기 전: features[0], aux_feat 만 원본 시점으로 되돌림 (head 가 쓰는 건 이 둘뿐, ganet_head.py:103-108)
  되돌렸을 때 BEV 화면 밖이라 값이 없는 칸은 기존 값으로 채움.
검증: 항등 변환을 넣으면 원래 neck 출력과 같아야 한다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_full_path.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_sa_inference_cover.py").read().split("\nfor name, path in JOBS:")[0])

FULL = dict(H=None, filled=None)
STRIDES = [8, 16, 32]


def full_bev_neck(inputs, Hm):
    inputs = list(inputs)
    if len(inputs) > len(neck.in_channels):
        del inputs[0]
    inputs = [warp_feat(f, Hm, s)[0] for f, s in zip(inputs, STRIDES)]  # BC3, BC4, BC5
    inputs[neck.trans_idx] = neck.trans_head(inputs[neck.trans_idx])  # BAC5
    laterals = [lc(inputs[i + neck.start_level]) for i, lc in enumerate(neck.lateral_convs)]
    for i in range(len(laterals) - 1, 0, -1):
        laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], **neck.upsample_cfg)
        if i - 1 == 0:
            aux = laterals[0]
        if neck.deconv_layer[i - 1] and not neck.deconv_before:
            d_feat, _ = neck.def_convs[i - 1](laterals[i - 1])
            laterals[i - 1] = d_feat + laterals[i - 1] if neck.use_res else d_feat
    return neck.fpn_convs[0](laterals[0]), aux


def full_hook(m, inp, out):
    if FULL["H"] is None:
        return None
    Hm = FULL["H"]
    p3, aux = full_bev_neck(inp[0], Hm)
    Hinv = np.linalg.inv(Hm)
    p3b, valid = warp_feat(p3, Hinv, 8)
    auxb, _ = warp_feat(aux, Hinv, 8)
    v = torch.from_numpy(valid).to(p3.device)[None, None]
    FULL["filled"] = 1 - valid.mean()
    o = dict(out)
    o["features"] = (torch.where(v, p3b, out["features"][0]),) + tuple(out["features"][1:])
    o["aux_feat"] = torch.where(v, auxb, out["aux_feat"])
    return o


neck.register_forward_hook(full_hook)

# 재구성 검증: 항등 변환이면 원래 neck 출력과 같아야 함
_t = torch.from_numpy(((cv2.resize(cv2.cvtColor(cv2.imread(JOBS[0][1]), cv2.COLOR_BGR2RGB), (W, H)).astype(np.float32)
                        - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
with torch.no_grad():
    bb = model.backbone(_t)
    ref = neck(bb)
    p3, aux = full_bev_neck(bb, np.eye(3))
print(f"[검증] 항등 변환 전체BEV vs 원래 neck: features[0] 최대 차이 {float((p3 - ref['features'][0]).abs().max()):.2e}, "
      f"aux_feat 최대 차이 {float((aux - ref['aux_feat']).abs().max()):.2e} (P3 크기 {float(ref['features'][0].abs().max()):.1f})")

for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    STATE["H"] = None; FULL["H"] = None
    hm0, pred0 = run(t)
    kpts = np.concatenate([sd[:, 0] for _, _, sd in pred0 if len(sd)]) * S + S / 2
    Hm, src, g, y0, inside = bev_homography_cover(lanes_px(pred0), kpts)
    FULL["H"] = Hm
    hmF, predF = run(t)
    FULL["H"] = None
    r0, l0 = summarize(pred0)
    rF, lF = summarize(predF)
    print(f"[프레임{name}] BEV 차선간격 {g:.2f}W, 윗변 y0={y0:.1f}H, 키포인트 BEV 안 {inside*100:.0f}%, "
          f"되돌린 P3 중 기존 값으로 채운 칸 {FULL['filled']*100:.0f}%")
    for c in range(2):
        for tag, r in [("기존       ", r0[c]), ("② 전체 BEV ", rF[c])]:
            print(f"   {CLS_NAME[c]} {tag}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, 선 {len(r[2])}개 {r[2]}, 버려진 점 {r[3]}")
        d = hmF[c] - hm0[c]
        print(f"   {CLS_NAME[c]} conf 변화: |변화| 최대 {np.abs(d).max():.2f}, "
              f"0.4 넘는 칸 {int((hm0[c] > kpt_thr).sum())} -> {int((hmF[c] > kpt_thr).sum())}")

    fig, axes = plt.subplots(2, 3, figsize=(21, 8.4), gridspec_kw=dict(hspace=0.1, wspace=0.03))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
    for r, (hm, lanes, tag) in enumerate([(hm0, l0, "기존"), (hmF, lF, "전체 BEV 경로\n(학습 없음)")]):
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
    fig.suptitle(f"{name}-6. 전체 BEV 경로 (학습 없이 추론만 변경) vs 기존\n"
                 "경로: C3·C4·C5를 BEV로 펴기 → BC5에 attention(BAC5) → BAC5+BC4 → +BC3 → 원본 시점으로 되돌리기 → head",
                 fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, :2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-6. 전체BEV경로_무학습추론.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
