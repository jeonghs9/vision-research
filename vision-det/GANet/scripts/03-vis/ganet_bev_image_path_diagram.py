"""9번 그림(BEV 이미지 입력)의 줄별 경로 그림: 기존 / a / b / b'.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_image_path_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
OUT = "/home/hsjeong/workspace/vision-det/GANet/output/06-cls2-특징맵BEV무학습분석/figures/9. 경로별_과정도_BEV이미지입력.png"

ORG, BEV, BACK, FILL, HEAD = "#4C78A8", "#F58518", "#B279A2", "#9E9E9E", "#54A24B"
X = [0, 1.55, 3.1, 4.65, 6.2, 7.75, 9.3, 10.85, 12.6]   # 입력, backbone, C3·C4·C5, attention, +C4, +C3, DCN, 되돌리기, head
YT, YB = 1.0, 0.0                                      # 위 = 원본 이미지 줄, 아래 = BEV 이미지 줄
BW, BH = 1.25, 0.42


def box(ax, x, y, text, color, faded=False, w=BW):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - BH / 2), w, BH, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color if not faded else "white", ec=color, lw=2,
                                alpha=1, ls="--" if faded else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=11.5, color="white" if not faded else color, weight="bold")


def arrow(ax, p, q, color, label=None, dashed=False, lpos=0.5, loff=(0, 0.1)):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                                    ls="--" if dashed else "-", shrinkA=0, shrinkB=0))
    if label:
        ax.text(p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1], label,
                ha="center", va="bottom", fontsize=10, color=color)


def r(x):   # 박스 오른쪽 끝
    return x + BW / 2


def l(x):
    return x - BW / 2


def chain(ax, y, labels, color, xs, faded=()):
    for i, (x, t) in enumerate(zip(xs, labels)):
        box(ax, x, y, t, color, faded=i in faded)
        if i:
            arrow(ax, (r(xs[i - 1]), y), (l(x), y), color, dashed=i in faded)


def heads(ax, ys=(0.75, 0.25)):
    box(ax, X[8], ys[0], "head: conf", HEAD, w=1.45)
    box(ax, X[8], ys[1], "head: offset", HEAD, w=1.45)
    return ys


def setup(ax, tag, desc, two=True):
    ax.set_xlim(-0.8, 13.5); ax.set_ylim(-0.55 if two else 0.1, 1.55)
    ax.axis("off")
    ax.text(-0.75, 1.5, tag, fontsize=16, weight="bold", va="top")
    ax.text(0.35, 1.5, desc, fontsize=12.5, va="top", color="#333")


LAB = ["{}\n이미지", "backbone", "{}C3·{}C4·{}C5", "attention\n→ {}AC5", "+{}C4", "+{}C3\n(DCN 전)", "DCN\n→ {}P3"]


def labels(p, img):
    return [LAB[0].format(img)] + [LAB[1]] + [t.format(*[p] * t.count("{}")) for t in LAB[2:]]


fig, axes = plt.subplots(4, 1, figsize=(19, 15.5), gridspec_kw=dict(hspace=0.05, height_ratios=[1.45, 2.1, 2.1, 2.1]))

# 기존
ax = axes[0]; setup(ax, "기존", "원본 이미지만 씀. GANet 원래 경로", two=False)
y = 1.0
chain(ax, y, labels("", "원본"), ORG, X[:7])
hy = heads(ax, (1.0, 0.45))
arrow(ax, (r(X[6]), y), (X[8] - 0.725, hy[0]), ORG)
arrow(ax, (X[5], y - BH / 2), (X[8] - 0.725, hy[1]), ORG)

# a
ax = axes[1]; setup(ax, "a", "BEV 이미지에서 만든 BAC5만 원본 시점으로 되돌려 AC5 자리에 넣음. 이후 +C4, +C3는 원본 이미지 것")
chain(ax, YT, labels("", "원본"), ORG, X[:7], faded=(3,))
chain(ax, YB, labels("B", "BEV"), BEV, X[:4])
arrow(ax, (X[0], YT - BH / 2), (X[0], YB + BH / 2), BEV, "사다리꼴로 펴기", lpos=0.5, loff=(0.12, -0.05))
ax.texts[-1].set_ha("left")
box(ax, X[3] + 0.78, 0.5, "되돌리기", BACK, w=1.0)
arrow(ax, (r(X[3]), YB), (X[3] + 0.78, 0.5 - BH / 2), BACK)
arrow(ax, (X[3] + 0.78, 0.5 + BH / 2), (r(X[3]) - 0.1, YT - BH / 2), BACK)
ax.text(X[3], YT + 0.3, "사다리꼴 안 칸은 BAC5로 교체", ha="center", fontsize=10, color=BACK)
hy = heads(ax, (1.0, 0.45))
arrow(ax, (r(X[6]), YT), (X[8] - 0.725, hy[0]), ORG)
arrow(ax, (X[5], YT - BH / 2), (X[8] - 0.725, hy[1]), ORG)

# b, b'
for ax, tag, desc, prime in [
        (axes[2], "b", "neck 전체를 BEV 이미지에서 계산(BAC5 → +BC4 → +BC3) 후 head 직전에 되돌림. conf · offset 둘 다 BEV 것", False),
        (axes[3], "b′", "b와 같은데 offset만 기존(원본 이미지) 것. BEV에서 나온 offset은 방향 · 거리가 BEV 기준이라 원본에서 틀림", True)]:
    setup(ax, tag, desc)
    chain(ax, YT, labels("", "원본"), ORG, X[:7], faded=(1, 2, 3, 4, 5, 6) if not prime else (6,))
    chain(ax, YB, labels("B", "BEV"), BEV, X[:7])
    arrow(ax, (X[0], YT - BH / 2), (X[0], YB + BH / 2), BEV, "사다리꼴로 펴기", loff=(0.12, -0.05))
    ax.texts[-1].set_ha("left")
    box(ax, X[7], YB, "되돌리기", BACK, w=1.0)
    arrow(ax, (r(X[6]), YB), (X[7] - 0.5, YB), BEV)
    hy = heads(ax, (0.75, 0.25))
    arrow(ax, (X[7] + 0.5, YB + 0.08), (X[8] - 0.725, hy[0]), BACK)
    arrow(ax, (r(X[6]), YT), (X[8] - 0.725, hy[0] + 0.12), FILL, "사다리꼴 밖 위치만", dashed=True, lpos=0.35, loff=(0, 0.04))
    if not prime:
        ax.annotate("", xy=(X[7] - 0.15, YB - BH / 2), xytext=(X[5], YB - BH / 2),
                    arrowprops=dict(arrowstyle="-|>", color=BEV, lw=2, connectionstyle="arc3,rad=0.35"))
        arrow(ax, (X[7] + 0.5, YB + 0.03), (X[8] - 0.725, hy[1] - 0.05), BACK)
        arrow(ax, (X[5], YT - BH / 2), (X[8] - 0.725, hy[1] + 0.08), FILL, dashed=True)
    else:
        arrow(ax, (X[5], YT - BH / 2), (X[8] - 0.725, hy[1]), ORG)
        ax.text(X[6] + 0.3, 0.6, "offset은 전부 기존 것", fontsize=10.5, color=ORG, weight="bold")
        ax.text(X[5], YB - 0.36, "BEV의 +BC3(DCN 전)은 offset에 안 보냄", fontsize=10, color=BEV, ha="center")

handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (ORG, BEV, BACK, HEAD)] + \
          [plt.Rectangle((0, 0), 1, 1, fc="white", ec=FILL, ls="--", lw=2)]
fig.legend(handles, ["원본 이미지 경로", "BEV 이미지 경로", "원본 시점으로 되돌리기", "head", "점선 = BEV 이미지에 없는 위치(사다리꼴 밖)만 채움"],
           loc="upper center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("9. BEV 이미지 입력 시 경로별 과정 (n-9 그림의 줄: 기존 · a · b · b′)", fontsize=19, y=0.985)
fig.savefig(OUT, dpi=90, bbox_inches="tight", pad_inches=0.2)
print(OUT)
