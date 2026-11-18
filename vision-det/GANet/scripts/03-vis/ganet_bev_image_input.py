"""1단계-⑤ BEV '이미지'를 backbone 에 따로 넣어서 BAC5 를 만들기 (학습 없이). 사다리꼴은 검출 결과 없이 고정 비율.

사다리꼴(입력 1280x736 기준, 모든 이미지 공통): 윗변 y=0.45H 에서 가운데 40% 폭, 아랫변 y=H 에서 화면 전체 폭 -> 화면 전체 직사각형으로 폄.
  BEV 이미지 = warpPerspective(이미지, H)  -> backbone -> BC3', BC4', BC5'   (특징맵을 늘린 게 아니라 BEV 이미지에서 새로 뽑은 특징)
  a: BAC5 = trans_head(BC5') -> 원본 시점으로 되돌림 -> AC5 자리에 넣고 +C4 -> +C3 -> head  (원본 이미지의 C4, C3)
  b: neck(BC3', BC4', BC5') 전체를 BEV 에서 계산 (= BAC5 -> +BC4 -> +BC3) -> head 가 쓰는 features[0], aux_feat 를 되돌림 -> head
  b': b 에서 conf 쪽(features[0])만 BEV 것을 쓰고 offset 쪽(aux_feat)은 기존
      (BEV 이미지에서 나온 offset 은 BEV 기준 방향·거리라서 칸 위치만 되돌려도 원본에서는 방향이 틀림)
되돌렸을 때 사다리꼴 밖이라 값이 없는 칸은 기존 값.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_image_input.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_sa_inference_cover.py").read().split("\nfor name, path in JOBS:")[0])

SRC = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])
DST = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
HM = cv2.getPerspectiveTransform(SRC, DST)
IMG = dict(mode=None, bb=None, busy=False)


def to_tensor(rgb):
    return torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)


def img_trans_hook(m, inp, out):  # a
    if IMG["mode"] != "a" or IMG["busy"]:
        return None
    IMG["busy"] = True
    bac5 = m(IMG["bb"][3])
    IMG["busy"] = False
    back, valid = warp_feat(bac5, np.linalg.inv(HM), 32)
    return torch.where(torch.from_numpy(valid).to(out.device)[None, None], back, out)


def img_neck_hook(m, inp, out):  # b, b'
    if IMG["mode"] not in ("b", "b2") or IMG["busy"]:
        return None
    IMG["busy"] = True
    ob = m(IMG["bb"])
    IMG["busy"] = False
    Hinv = np.linalg.inv(HM)
    p3, valid = warp_feat(ob["features"][0], Hinv, 8)
    v = torch.from_numpy(valid).to(p3.device)[None, None]
    o = dict(out)
    o["features"] = (torch.where(v, p3, out["features"][0]),) + tuple(out["features"][1:])
    if IMG["mode"] == "b":
        aux, _ = warp_feat(ob["aux_feat"], Hinv, 8)
        o["aux_feat"] = torch.where(v, aux, out["aux_feat"])
    return o


neck.trans_head.register_forward_hook(img_trans_hook)
neck.register_forward_hook(img_neck_hook)

MODES = [("a", "a: BAC5 → +C4 → +C3"), ("b", "b: BAC5 → +BC4 → +BC3"), ("b2", "b': b의 conf + 기존 offset")]
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    bev = cv2.warpPerspective(rgb, HM, (W, H))
    t, tb = to_tensor(rgb), to_tensor(bev)
    STATE["H"] = None; IMG["mode"] = None
    hm0, pred0 = run(t)
    with torch.no_grad():
        IMG["bb"] = model.backbone(tb)
    res = [(hm0, *summarize(pred0), "기존")]
    for mode, tag in MODES:
        IMG["mode"] = mode
        hm, pred = run(t)
        IMG["mode"] = None
        res.append((hm, *summarize(pred), tag))
    print(f"[프레임{name}]")
    for c in range(2):
        for hm, rows, _, tag in res:
            r = rows[c]
            print(f"   {CLS_NAME[c]} {tag:24s}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, 선 {len(r[2])}개 {r[2]}, "
                  f"버려진 점 {r[3]}, 0.4 넘는 칸 {int((hm[c] > kpt_thr).sum())}, |conf 변화| 최대 {np.abs(hm[c] - hm0[c]).max():.2f}")

    n = len(res)
    fig, axes = plt.subplots(n, 4, figsize=(28, 4.2 * n), gridspec_kw=dict(hspace=0.1, wspace=0.03))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
    for r, (hm, _, lanes, tag) in enumerate(res):
        ax = axes[r, 0]
        if r == 0:
            ax.imshow(rgb)
            ax.add_patch(plt.Polygon(SRC, fill=False, ec="#00e5ff", lw=2, ls="--"))
            ax.set_title("입력 이미지 (하늘색 = 사다리꼴)", fontsize=15)
        else:
            ax.imshow(bev)
            if r == 1:
                ax.set_title("BEV 이미지 (backbone에 따로 넣음)", fontsize=15)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(tag, fontsize=15)
        for c in range(2):
            ax = axes[r, c + 1]
            ax.imshow(bg)
            ax.imshow(cv2.resize(hm[c], (W, H), interpolation=cv2.INTER_NEAREST), cmap="jet", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{CLS_NAME[c]} conf", fontsize=15)
        ax = axes[r, 3]
        ax.imshow(rgb)
        for c, pts in lanes:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=CLS_COLOR[c], lw=3)
        if r > 0:
            ax.add_patch(plt.Polygon(SRC, fill=False, ec="#00e5ff", lw=1.5, ls="--"))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
    fig.subplots_adjust(top=1 - 1.7 / (4.2 * n))
    fig.suptitle(f"{name}-9. BEV 이미지 입력 시 경로별 검출 결과 (고정 비율 사다리꼴, 학습 없이 추론만 변경) vs 기존\n"
                 "경로: 이미지 → BEV 이미지 → backbone → BC3·BC4·BC5 → attention(BAC5) → [a: 되돌려 +C4 +C3] / [b: +BC4 +BC3 후 되돌림] → head",
                 fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, 1:3], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-9. 경로별_검출결과_BEV이미지입력.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
