"""전체 2클래스 res34 실험(004, 006~013)의 경로별 과정도 — 9번 그림(ganet_bev_image_path_diagram.py)과 같은 형식.

경로 근거: dcn_fpn.py DeformFPN.forward
  - 입력 첫 단계(C2, P3del은 C3)는 버림 -> 기존 C3·C4·C5 / P3del C4·C5
  - trans_head(SA/DA, APT는 merged mask 같이) 결과가 C5 자리를 대체(trans_mode='replace')
  - top-down 덧셈 후 맨 아래 층에 DCN, 그 직전 특징(aux_feat)이 offset head 로 (dcn_only_cls=True)

--bev: BEV 입력 버전(21·23번 그림과 같은 입력). 원본을 사다리꼴(0.3W,0.45H)-(0.7W,0.45H)-(W,H)-(0,H)로 펴서
  같은 학습 모델에 그대로 넣음. 결과는 BEV 좌표로 남고 되돌리지 않음. APT 엔트로피도 BEV 이미지에서 계산.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_experiment_path_diagram.py
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_experiment_path_diagram.py --bev
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
BEVMODE = "--bev" in sys.argv
OUT = "/home/hsjeong/workspace/vision-det/GANet/output/07-cls2-P3제거특징맵분석/figures/" + \
    ("25. 경로별_과정도_실험별구조_BEV입력.png" if BEVMODE else "24. 경로별_과정도_실험별구조.png")
B = "B" if BEVMODE else ""                       # BEV 이미지에서 나온 특징은 B 붙임(9번 그림과 같음)
XL = -2.4 if BEVMODE else -0.8

ORG, NEW, APT, GONE, HEAD, BACK = "#4C78A8", "#E45756", "#F58518", "#9E9E9E", "#54A24B", "#B279A2"
X = [0, 1.55, 3.1, 4.75, 6.4, 7.95, 9.5, 11.3]   # 입력, backbone, C3·C4·C5, C5 처리, +C4, +C3, DCN, head
YT, YB = 1.0, 0.0
BW, BH = 1.25, 0.42


def box(ax, x, y, text, color, faded=False, w=BW):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - BH / 2), w, BH, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color if not faded else "white", ec=color, lw=2, ls="--" if faded else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=11.5, color="white" if not faded else color, weight="bold")


def arrow(ax, p, q, color, label=None, dashed=False, lpos=0.5, loff=(0, 0.1), ha="center"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                                    ls="--" if dashed else "-", shrinkA=0, shrinkB=0))
    if label:
        ax.text(p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1], label,
                ha=ha, va="bottom", fontsize=10, color=color)


def r(x, w=BW):
    return x + w / 2


def l(x, w=BW):
    return x - w / 2


def setup(ax, tag, desc, score, two=False):
    ax.set_xlim(XL, 12.2); ax.set_ylim(-0.4 if two else 0.1, 1.55)
    ax.axis("off")
    ax.text(XL + 0.05, 1.5, tag, fontsize=15, weight="bold", va="top")
    ax.text(2.05, 1.5, desc, fontsize=12, va="top", color="#333")
    if BEVMODE:
        return
    ax.text(12.15, 1.5, score, fontsize=11.5, va="top", ha="right", color="#333", weight="bold")


def row(ax, p3=True, mid=(f"SA\n→ {B}AC5", ORG, False), apt=False):
    """p3: +C3 단계 유무, mid: (C5 처리 박스 글, 색, 없음 여부), apt: 아래 줄에 병합 mask 경로"""
    y = YT
    if BEVMODE:
        box(ax, X[0] - 1.65, y, "원본\n이미지", ORG)
        box(ax, X[0], y, "BEV\n이미지", BACK)
        arrow(ax, (r(X[0] - 1.65), y), (l(X[0]), y), BACK)
        ax.text(X[0] - 0.82, y - BH / 2 - 0.05, "사다리꼴로\n펴기", ha="center", va="top", fontsize=10, color=BACK)
    else:
        box(ax, X[0], y, "원본\n이미지", ORG)
    box(ax, X[1], y, "backbone", ORG)
    box(ax, X[2], y, f"{B}C3·{B}C4·{B}C5" if p3 else f"{B}C4·{B}C5", ORG)
    box(ax, X[3], y, mid[0], mid[1], faded=mid[2])
    box(ax, X[4], y, f"+{B}C4", ORG)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        arrow(ax, (r(X[a]), y), (l(X[b]), y), ORG, dashed=(mid[2] and b == 4) or (mid[2] and b == 3))
    if p3:
        box(ax, X[5], y, f"+{B}C3\n(DCN 전)", ORG)
        arrow(ax, (r(X[4]), y), (l(X[5]), y), ORG)
        dcn_from, last = X[5], f"{B}P3"
    else:
        box(ax, X[5], y, "+C3 없음", GONE, faded=True)
        dcn_from, last = X[4], f"{B}P4"
    box(ax, X[6], y, f"DCN\n→ {last}", ORG)
    if p3:
        arrow(ax, (r(X[5]), y), (l(X[6]), y), ORG)
    else:  # +C4 에서 +C3 칸을 건너뛰어 DCN 으로
        ax.annotate("", xy=(l(X[6]), y + 0.06), xytext=(r(X[4]), y + 0.06),
                    arrowprops=dict(arrowstyle="-|>", color=ORG, lw=2, connectionstyle="arc3,rad=-0.45"))
        ax.text(X[5], y + BH / 2 + 0.2, f"{B}P4 그대로", ha="center", fontsize=10, color=ORG)
    hc, ho = (y, y - 0.55)
    box(ax, X[7], hc, "head: conf", HEAD, w=1.45)
    box(ax, X[7], ho, "head: offset", HEAD, w=1.45)
    arrow(ax, (r(X[6]), y), (X[7] - 0.725, hc), ORG)
    arrow(ax, (dcn_from, y - BH / 2), (X[7] - 0.725, ho), ORG)
    if apt:
        box(ax, X[1], YB, "칸별 엔트로피\n(32px)", APT)
        if BEVMODE:
            ax.text(X[1], YB - BH / 2 - 0.05, "BEV 이미지에서 계산", ha="center", va="top", fontsize=10, color=APT)
        box(ax, X[2], YB, "병합 mask\n(τ=6.5 미만 2×2)", APT, w=1.45)
        arrow(ax, (X[0], y - BH / 2), (X[0], YB), APT)
        
        arrow(ax, (X[0], YB), (l(X[1]), YB), APT, "회색조", lpos=0.5, loff=(0, 0.03))
        arrow(ax, (r(X[1]), YB), (l(X[2], 1.45), YB), APT)
        arrow(ax, (r(X[2], 1.45), YB), (X[3], y - BH / 2), APT)


fig, axes = plt.subplots(9, 1, figsize=(21.3 if BEVMODE else 19, 30), gridspec_kw=dict(
    hspace=0.08, height_ratios=[1.45] * 6 + [1.95] * 3))
fig.subplots_adjust(top=0.955)

# 1단계: P3 · SA 덜어내기
ax = axes[0]; setup(ax, "004 기존 ★", "GANet 원래 경로. C5 전체에 SA(전역 self-attention). 현재 채택 모델", "평균 F1 .742 · 14.23ms")
row(ax, mid=(f"SA\n→ {B}AC5", ORG, False))
ax = axes[1]; setup(ax, "007 −SA", "기존에서 SA만 뺌. C5가 바로 +C4로", "평균 F1 .752 · 13.90ms")
row(ax, mid=("SA 없음", GONE, True))
ax = axes[2]; setup(ax, "006 P3del", "+C3 단계를 뺌. +C4(=P4, 입력의 1/16)에 DCN, offset은 DCN 전 +C4", "평균 F1 .758 · 10.86ms")
row(ax, p3=False)
ax = axes[3]; setup(ax, "008 P3del−SA", "+C3 단계와 SA를 둘 다 뺌", "평균 F1 .774 · 10.42ms  (1위)")
row(ax, p3=False, mid=("SA 없음", GONE, True))

# 2단계: SA 를 DA 로 교체
ax = axes[4]; setup(ax, "009 P3del+DA (query)", "SA 자리에 DA(각 칸이 32곳만 봄). 위치 인코딩은 '어디 볼지' 예측(query)에만 더함", "평균 F1 .763 · 12.12ms")
row(ax, p3=False, mid=(f"DA (query)\n→ {B}AC5", NEW, False))
ax = axes[5]; setup(ax, "010 P3del+DA (add)", "009와 같은데 위치 인코딩을 특징 자체에 더함(SA와 같은 방식)", "평균 F1 .754 · 12.09ms")
row(ax, p3=False, mid=(f"DA (add)\n→ {B}AC5", NEW, False))

# 3단계: APT 병합
ax = axes[6]; setup(ax, "011 P3del+SA+APT", ("BEV" if BEVMODE else "원본") + " 이미지에서 단순한 곳(엔트로피 낮은 2×2칸)을 큰 토큰 1개로 합친 뒤 SA", "평균 F1 .769 · 13.61ms", two=True)
row(ax, p3=False, mid=(f"SA+APT\n→ {B}AC5", APT, False), apt=True)
ax = axes[7]; setup(ax, "012 P3del+DA+APT", "011과 같은 병합 뒤 DA(query). 합친 토큰은 그 2×2 가운데에서 32곳을 봄", "평균 F1 .762 · 14.56ms", two=True)
row(ax, p3=False, mid=(f"DA+APT\n→ {B}AC5", APT, False), apt=True)
ax = axes[8]; setup(ax, "013 P3del−SA+APT", "attention 없이 병합만: 합친 칸은 큰 토큰 값으로 2×2를 채움", "평균 F1 .765 · 11.69ms", two=True)
row(ax, p3=False, mid=(f"APT 병합만\n→ {B}C5", APT, False), apt=True)

handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (ORG, NEW, APT, HEAD)] + \
          [plt.Rectangle((0, 0), 1, 1, fc="white", ec=GONE, ls="--", lw=2)]
if BEVMODE:
    handles.insert(1, plt.Rectangle((0, 0), 1, 1, fc=BACK))
fig.legend(handles, ["기존 GANet 경로"] + (["BEV 이미지(사다리꼴로 펴기)"] if BEVMODE else []) + [ "SA → DA 교체", "APT 병합 경로", "head", "점선 = 뺀 단계"],
           loc="upper center", ncol=6 if BEVMODE else 5, fontsize=12.5, frameon=False, bbox_to_anchor=(0.5, 0.972))
fig.suptitle("25. BEV 이미지 입력 시 실험별 경로 (res34 2클래스, 004 · 006~013)   모델은 24번과 같음, 결과는 BEV 좌표 그대로"
             if BEVMODE else "24. 실험별 경로 (res34 2클래스, 004 · 006~013)   평균 F1 = ep100~150 평균, 시간 = V100 1장 forward",
             fontsize=19, y=0.985)
fig.savefig(OUT, dpi=90, bbox_inches="tight", pad_inches=0.2)
print(OUT)
