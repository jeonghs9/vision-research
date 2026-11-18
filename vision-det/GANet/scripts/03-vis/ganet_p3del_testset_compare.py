"""test셋(163장, = val과 같은 목록) 전체에서 기존 2cls vs P3del 모델 4개 결과를 한 장씩 비교 그림으로 저장.

그림(이미지마다 1장, 원본 stem 그대로): 2x3 격자
  [정답        | res18 기존 | res18 P3del]
  [(빈칸)      | res34 기존 | res34 P3del]
칸 제목 = 맞춘 선 / 예측 선 / 정답 선 수, 맞춘 선 중 클래스(실선·점선)까지 맞은 수.
매칭은 score_checkpoint.py 와 같음(평균 |x오차| 40px 이하, 원본 3840 기준, 그리디).
이미지별 수치는 비교표.csv, 차이 큰 순 목록은 README.md 에 씀.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_testset_compare.py
"""
import csv
import importlib.util
import os
import sys
sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_dp, compat_cfg, get_device
from projects.plugin import *  # noqa

G = "/home/hsjeong/workspace/vision-det/GANet"
spec = importlib.util.spec_from_file_location("sc", f"{G}/scripts/04-eval/score_checkpoint.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

OUT = f"{G}/output/04-cls2-P3del비교-test결과"
MODELS = [
    ("res18 기존", "res18_s8_full_2cls", f"{G}/models/drone-ganet-res18-2cls-v1.pth"),
    ("res18 P3del", "res18_s16_full_2cls_P3del", f"{G}/test/005-res18-2cls-1280-P3del-150ep/epoch_150.pth"),
    ("res34 기존", "res34_s8_full_2cls", f"{G}/models/drone-ganet-res34-2cls-v1.pth"),
    ("res34 P3del", "res34_s16_full_2cls_P3del", f"{G}/test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth"),
]
CLS_COLOR = {1: "#00dc00", 2: "#dc0000"}  # 1=실선 초록, 2=점선 빨강 (ganet.py CLASS_COLORS 와 같은 색)
SHOW_W = 1280


def match_pairs(preds, gts):
    """score_checkpoint.match_lanes 와 같은 그리디 매칭, 매칭쌍 (예측 idx, 정답 idx) 반환."""
    if not preds or not gts:
        return []
    pr = [sc.resample(p) for p in preds]
    gt = [sc.resample(g) for g in gts]
    pairs = sorted((sc.mean_dist(p, g), i, j) for i, p in enumerate(pr) for j, g in enumerate(gt))
    used_p, used_g, out = set(), set(), []
    for d, i, j in pairs:
        if d > sc.MATCH_THR:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); out.append((i, j))
    return out


def run_model(cfg_name, ckpt):
    cfg = compat_cfg(mmcv.Config.fromfile(f"projects/cfgs/drone/{cfg_name}.py"))
    cfg.gpu_ids = [0]
    cfg.device = get_device()
    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    ds.open_check(); ds.set_all_scenes()
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)
    cfg.model.train_cfg = None
    cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt, map_location="cpu")
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids).eval()
    out = {}
    with torch.no_grad():
        for data in loader:
            r = model(return_loss=False, rescale=True, **data)[0]
            lanes, _, _, cls = r["final_dict_list"]
            meta = r["img_metas"]
            gts = [[(l[2 * i], l[2 * i + 1]) for i in range(len(l) // 2)] for l in meta.get("gt_points_for_show", [])]
            out[meta["filename"]] = dict(lanes=lanes, cls=list(cls), gts=gts, gt_cls=list(meta.get("id_classes", [])))
    return out


res = [run_model(c, k) for _, c, k in MODELS]
files = list(res[0].keys())
os.makedirs(f"{OUT}/images", exist_ok=True)
rows = []
for f in files:
    stem = os.path.splitext(os.path.basename(f))[0]
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    s = SHOW_W / img.shape[1]
    img = cv2.resize(img, (SHOW_W, int(img.shape[0] * s)))
    gts, gt_cls = res[0][f]["gts"], res[0][f]["gt_cls"]
    stats = []
    for m in range(4):
        d = res[m][f]
        pairs = match_pairs(d["lanes"], gts)
        cls_ok = sum(1 for i, j in pairs if i < len(d["cls"]) and j < len(gt_cls) and d["cls"][i] == gt_cls[j])
        stats.append((len(pairs), len(d["lanes"]), len(gts), cls_ok))

    fig, axes = plt.subplots(2, 3, figsize=(24, 9.6), gridspec_kw=dict(hspace=0.12, wspace=0.03))
    def draw(ax, lanes, cls, title):
        ax.imshow(img)
        for k, l in enumerate(lanes):
            p = np.asarray(l, float) * s
            ax.plot(p[:, 0], p[:, 1], "-", color=CLS_COLOR.get(cls[k] if k < len(cls) else 1, "#ffd400"), lw=3)
        ax.set_title(title, fontsize=15)
        ax.set_xticks([]); ax.set_yticks([])
    draw(axes[0, 0], gts, gt_cls, f"정답 ({len(gts)}개)")
    axes[1, 0].axis("off")
    for m, (r, c) in enumerate([(0, 1), (0, 2), (1, 1), (1, 2)]):
        mp, npred, ngt, ok = stats[m]
        d = res[m][f]
        draw(axes[r, c], d["lanes"], d["cls"], f"{MODELS[m][0]}: 맞춤 {mp} / 예측 {npred} / 정답 {ngt} · 클래스 일치 {ok}")
    fig.suptitle(f"{stem}  기존 vs P3del (초록 실선 · 빨강 점선)", fontsize=17)
    fig.savefig(f"{OUT}/images/{stem}.png", dpi=70, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    row = dict(이미지=stem, 정답=len(gts))
    for m, (mp, npred, _, ok) in enumerate(stats):
        tag = MODELS[m][0].replace(" ", "_")
        row[f"{tag}_맞춤"] = mp; row[f"{tag}_예측"] = npred; row[f"{tag}_클래스일치"] = ok
    rows.append(row)

with open(f"{OUT}/비교표.csv", "w", newline="", encoding="utf-8-sig") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)


def score(row, tag):
    """이미지별 점수 = 맞춘 선 - 틀린 예측 선 (높을수록 좋음)."""
    mp, npred = row[f"{tag}_맞춤"], row[f"{tag}_예측"]
    return mp - (npred - mp)


summary = []
for bb in ["res18", "res34"]:
    deltas = [(score(r, f"{bb}_P3del") - score(r, f"{bb}_기존"), r) for r in rows]
    better = sorted([d for d in deltas if d[0] > 0], key=lambda t: -t[0])
    worse = sorted([d for d in deltas if d[0] < 0], key=lambda t: t[0])
    same = len(deltas) - len(better) - len(worse)
    tot = {k: sum(r[f"{bb}_{k2}_{k3}"] for r in rows) for k, (k2, k3) in
           dict(기맞=("기존", "맞춤"), 기예=("기존", "예측"), 기클=("기존", "클래스일치"),
                P맞=("P3del", "맞춤"), P예=("P3del", "예측"), P클=("P3del", "클래스일치")).items()}
    summary.append((bb, better, worse, same, tot))
    print(f"{bb}: P3del 좋아짐 {len(better)}장, 같음 {same}장, 나빠짐 {len(worse)}장 | 합계 {tot}")

gt_total = sum(r["정답"] for r in rows)
L = ["# 기존 2cls vs P3del — test셋 이미지별 비교", "",
     "> 2026-11-10. 관련: [docs/01-model-review/12. P3 제거 구조 학습 (P3del).md]"
     "(../../docs/01-model-review/12.%20P3%20제거%20구조%20학습%20%28P3del%29.md)", "",
     f"- test셋: `culane_lane_ganet/list/test.txt` {len(rows)}장. **val.txt와 같은 목록입니다**(따로 떼어 둔 test셋 없음).",
     "  epoch 선택도 이 163장으로 했으므로 낙관적인 수치입니다.",
     "- 모델: 기존 = `models/drone-ganet-res{18,34}-2cls-v1.pth`(res18 ep120, res34 ep130), "
     "P3del = `test/005·006` epoch_150",
     "- 매칭: `scripts/04-eval/score_checkpoint.py`와 같음(평균 x 오차 40px 이하, 원본 3840 기준). "
     "클래스 일치 = 매칭된 선 중 실선/점선까지 맞은 수",
     "- 그림: `images/{원본 stem}.png` — 왼쪽 위 정답, 오른쪽 2x2 = res18 기존·P3del(윗줄), res34 기존·P3del(아랫줄)",
     "- 이미지별 전체 수치: `비교표.csv`", "",
     "실행:", "", "```bash", "cd /home/hsjeong/workspace/vision-det/GANet/source",
     "PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \\",
     "  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_testset_compare.py",
     "```", "", "## 합계", "",
     f"정답 선 {gt_total}개.", "",
     "| 모델 | 맞춘 선 | 예측 선 | 틀린 예측 | 클래스 일치(맞춘 선 중) |", "|---|---|---|---|---|"]
for bb, _, _, _, t in summary:
    L.append(f"| {bb} 기존 | {t['기맞']} | {t['기예']} | {t['기예'] - t['기맞']} | {t['기클']} ({t['기클'] / max(t['기맞'], 1):.1%}) |")
    L.append(f"| {bb} P3del | {t['P맞']} | {t['P예']} | {t['P예'] - t['P맞']} | {t['P클']} ({t['P클'] / max(t['P맞'], 1):.1%}) |")
L += ["", "## 차이 큰 이미지", "",
      "이미지별 점수 = 맞춘 선 − 틀린 예측 선. 차이 = P3del 점수 − 기존 점수(+면 P3del이 나음).", ""]
for bb, better, worse, same, _ in summary:
    L += [f"### {bb}: P3del 좋아짐 {len(better)}장 · 같음 {same}장 · 나빠짐 {len(worse)}장", ""]
    for name, lst in [("P3del이 나아진 이미지 (상위 10)", better[:10]), ("P3del이 나빠진 이미지 (상위 10)", worse[:10])]:
        L += [f"**{name}**", "", "| 이미지 | 차이 | 정답 | 기존 맞춤/예측 | P3del 맞춤/예측 |", "|---|---|---|---|---|"]
        for dlt, r in lst:
            L.append(f"| `{r['이미지']}` | {dlt:+d} | {r['정답']} | {r[f'{bb}_기존_맞춤']}/{r[f'{bb}_기존_예측']} "
                     f"| {r[f'{bb}_P3del_맞춤']}/{r[f'{bb}_P3del_예측']} |")
        L.append("")
open(f"{OUT}/README.md", "w").write("\n".join(L))
