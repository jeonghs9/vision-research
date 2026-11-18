"""P3 단계를 제거한 GANet의 최종 검출 결과 (학습 없이 추론만 변경) vs 기존.

P3 제거 경로: C4 -> lateral, C5 -> attention(AC5) -> lateral -> 업샘플해 더함 = P4
  -> (기존 P3 자리 그대로) DCN(def_convs[0]) -> fpn_convs[0] -> head(conf, offset).  C3는 쓰지 않음, 출력 1/16.
줄: 기존(1/8) / P3 제거(offset 값 그대로 1/16 칸 단위로 해석) / P3 제거 + offset ½
  (head 는 1/8 칸 단위로 offset 을 배웠으므로, 1/16 칸 단위로 환산하려면 ½ 이 맞을 수 있어 둘 다 봄)
후처리 문턱값(root_thr=1, cluster_by_center_thr=5, points_thr=10)은 칸 단위 그대로.
GANet 소스는 건드리지 않고 neck.forward 와 offset 헤드 출력만 이 스크립트 안에서 바꿔 끼운다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_noP3_detection.py
"""
exec(open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_sa_inference_cover.py").read().split("\nfor name, path in JOBS:")[0])
import os
import torch.nn.functional as F

OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "07-cls2-P3제거특징맵분석/figures")
MODE = dict(noP3=False, half=False)
orig_forward = neck.forward


def forward_switch(inputs):
    if not MODE["noP3"]:
        return orig_forward(inputs)
    C3, C4, C5 = list(inputs)[1:]
    p4 = neck.lateral_convs[1](C4)
    p4 = p4 + F.interpolate(neck.lateral_convs[2](neck.trans_head(C5)), size=p4.shape[2:], **neck.upsample_cfg)
    d_feat, points = neck.def_convs[0](p4)  # 기존 P3 자리의 DCN(1x7) 그대로
    return dict(features=(neck.fpn_convs[0](d_feat),), aux_feat=p4, deform_points=(points,))
neck.forward = forward_switch


def half_hook(m, inp, out):
    if MODE["half"]:
        return {k: v * 0.5 for k, v in out.items()}
    return None
model.bbox_head.offset_head.register_forward_hook(half_hook)
model.bbox_head.reg_head.register_forward_hook(half_hook)

ROWS = [(False, False, 8, "기존 (P3, 1/8)"),
        (True, False, 16, "P3 제거 (P4, 1/16)\noffset 그대로"),
        (True, True, 16, "P3 제거 (P4, 1/16)\noffset ½")]
os.makedirs(OUTDIR, exist_ok=True)
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    res = []
    for noP3, half, s, tag in ROWS:
        MODE.update(noP3=noP3, half=half)
        S = s  # summarize() 가 점 좌표를 픽셀로 바꿀 때 쓰는 stride
        hm, pred = run(t)
        rows, lanes = summarize(pred)
        res.append((hm, rows, lanes, tag))
    MODE.update(noP3=False, half=False)
    print(f"[프레임{name}]")
    for c in range(2):
        for hm, rows, _, tag in res:
            r = rows[c]
            print(f"   {CLS_NAME[c]} {tag.replace(chr(10), ' '):34s}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, "
                  f"선 {len(r[2])}개 {r[2]}, 버려진 점 {r[3]}, 0.4 넘는 칸 {int((hm[c] > kpt_thr).sum())}")

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
        axes[r, 0].set_ylabel(tag, fontsize=15)
        ax = axes[r, 2]
        ax.imshow(rgb)
        for c, pts in lanes:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=CLS_COLOR[c], lw=3)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
    fig.subplots_adjust(top=1 - 1.2 / (4.2 * n))
    fig.suptitle(f"{name}-2. P3 제거 시 최종 검출 결과 (P4가 최종 출력, 학습 없이 추론만 변경) vs 기존", fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, 0:2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-2. 최종검출결과_P3제거.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
