import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

# ---- 한글 폰트 ----
_cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk_path)
_cjk_name = fm.FontProperties(fname=_cjk_path).get_name()
plt.rcParams["font.family"] = _cjk_name
plt.rcParams["axes.unicode_minus"] = False

COLOR_EXT = "#9e9e9e"      # 외부 mmdet 프레임워크
COLOR_EXT_FILL = "#f0f0f0"
COLOR_ORIG = "#1f6feb"     # 원본 저자 코드
COLOR_ORIG_FILL = "#e8f1ff"
COLOR_OURS = "#d97706"     # 우리가 수정/작성
COLOR_OURS_FILL = "#fff3e0"

fig, ax = plt.subplots(figsize=(15, 23))
ax.set_xlim(0, 10)
ax.set_ylim(0, 34)
ax.axis("off")

def box(x, y, w, h, text, edge, fill, fontsize=10.5, weight="normal", align="center"):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color="#222222", weight=weight, zorder=3, linespacing=1.5)
    return (x + w / 2, y, x + w / 2, y + h)

def arrow(x1, y1, x2, y2, label=None, color="#333333", lw=1.6, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                         linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label, fontsize=9,
                 color=color, ha="left", va="center", style="italic")

# ---------------- 제목 ----------------
ax.text(5, 33.3, "GANet 모델 아키텍처 — 실제 forward 코드 흐름 (ResNet34 + DeformFPN, drone 2cls 설정 기준)",
        ha="center", va="center", fontsize=15, weight="bold")

# ---------------- 범례 ----------------
lx, ly = 0.3, 31.9
ax.add_patch(FancyBboxPatch((lx, ly), 0.35, 0.35, boxstyle="round,pad=0.03", facecolor=COLOR_EXT_FILL, edgecolor=COLOR_EXT, linewidth=1.5))
ax.text(lx + 0.5, ly + 0.17, "외부 mmdet 프레임워크 코드", fontsize=10, va="center")
ax.add_patch(FancyBboxPatch((lx + 4.0, ly), 0.35, 0.35, boxstyle="round,pad=0.03", facecolor=COLOR_ORIG_FILL, edgecolor=COLOR_ORIG, linewidth=1.5))
ax.text(lx + 4.5, ly + 0.17, "원본 저자(GANet) 코드", fontsize=10, va="center")
ax.add_patch(FancyBboxPatch((lx + 7.4, ly), 0.35, 0.35, boxstyle="round,pad=0.03", facecolor=COLOR_OURS_FILL, edgecolor=COLOR_OURS, linewidth=1.5))
ax.text(lx + 7.9, ly + 0.17, "우리가 작성/수정한 코드", fontsize=10, va="center")

# ---------------- 0. 입력 ----------------
_, _, cx0, cy0 = box(3.5, 30.4, 3, 1.0, "입력 이미지\n1280×736×3", COLOR_EXT, "#ffffff", fontsize=11, weight="bold")

# ---------------- 1. Backbone ----------------
y1 = 26.6
box(1.2, y1, 7.6, 3.4,
    "① Backbone : ResNet34  (외부, mmdet/models/backbones/resnet.py)\n\n"
    "stem(7×7,s2)+maxpool(s2) → layer1(s1)=C2 (64, 184,320)\n"
    "→ layer2(s2)=C3 (128, 92,160)  → layer3(s2)=C4 (256, 46,80)  → layer4(s2)=C5 (512, 23,40)\n\n"
    "호출 위치 : extract_feat() — single_stage.py (외부)   x = self.backbone(img)",
    COLOR_EXT, COLOR_EXT_FILL, fontsize=10.5)
arrow(5, 30.4, 5, y1 + 3.4, label="img")
arrow(5, y1 + 3.4 - 3.4, 5, y1 + 2.9 - 2.9 + 0, color="none")  # no-op spacer removed below

# ---------------- extract_feat 연결 ----------------
arrow(5, y1, 5, y1 - 0.55, label="backbone(img) → (C2,C3,C4,C5)")

# ---------------- 2. Neck (DeformFPN) 큰 틀 ----------------
yneck_top = y1 - 0.75
yneck_bot = yneck_top - 11.4
box(0.5, yneck_bot, 9.0, yneck_top - yneck_bot,
    "", COLOR_ORIG, "#ffffff", fontsize=1)
ax.text(1.0, yneck_top - 0.35, "② Neck : DeformFPN  (원본 저자 코드, dcn_fpn.py)   — self.neck(x)",
        fontsize=11.5, weight="bold", color=COLOR_ORIG)

sub_x, sub_w = 1.1, 7.8
y = yneck_top - 1.0

def subbox(y, h, text, fontsize=9.8):
    box(sub_x, y - h, sub_w, h, text, COLOR_ORIG, COLOR_ORIG_FILL, fontsize=fontsize)
    return y - h

yA_top = y
yA_bot = subbox(y, 0.75, "(A) C2 버림 → inputs = (C3, C4, C5)")
arrow(5, yA_bot, 5, yA_bot - 0.25)

yB_bot = subbox(yA_bot - 0.25, 1.15,
                 "(B) Self-Attention (AttentionLayer)  — fpn_attn_idx=-1 → C5(마지막 레벨)에만 적용\n"
                 "C5 (512,23,40) → AttnC5 (512,23,40)   [gamma·attn + residual, gamma 초기값 0]")
arrow(5, yB_bot, 5, yB_bot - 0.25)

yC_bot = subbox(yB_bot - 0.25, 1.35,
                 "(C) lateral_convs : 1×1 conv, 채널 → 64  (레벨마다 따로, 반복문으로 호출)\n"
                 "L0 = lateral(C3) (64,92,160)   L1 = lateral(C4) (64,46,80)   L2 = lateral(AttnC5) (64,23,40)")
arrow(5, yC_bot, 5, yC_bot - 0.25)

yD_bot = subbox(yC_bot - 0.25, 1.15,
                 "(D) Top-down 융합 — 덧셈(+=), concat 아님\n"
                 "L1 += upsample(L2)  →  L0 += upsample(L1)")
arrow(5, yD_bot, 5, yD_bot - 0.25)

yE_bot = subbox(yD_bot - 0.25, 1.45,
                 "(E) LFA (LanePointsConv, def_convs[0])  — hm_idx=0, 즉 L0에만 적용\n"
                 "aux_feat = LFA 적용 \"전\" L0 (따로 보관)  →  L0 자체는 LFA 결과로 교체 (더하기 아님)")
ax.text(sub_x + sub_w + 0.15, yE_bot + 0.72, "★ aux_feat\n여기서 분기", fontsize=9, color=COLOR_OURS, weight="bold", va="center")
arrow(5, yE_bot, 5, yE_bot - 0.25)

yF_bot = subbox(yE_bot - 0.25, 1.15,
                 "(F) fpn_convs : 3×3 conv (레벨별)\n"
                 "out0 (64,92,160)   out1 (64,46,80)   out2 (64,23,40)")

ax.text(5, yF_bot - 0.45, "return { features: [out0, out1, out2],  aux_feat }",
        ha="center", fontsize=10, weight="bold", color=COLOR_ORIG)

# ---------------- aux_feat 분기선 (E → Head 우측 브랜치) ----------------
# 작은 꺾인 화살표로 aux_feat 이 head까지 직접 전달됨을 표시
aux_y_start = yE_bot + 0.72
ax.plot([sub_x + sub_w + 0.9, sub_x + sub_w + 0.9], [aux_y_start, yneck_bot - 1.3],
        color=COLOR_OURS, linewidth=1.6, linestyle="--", zorder=1)
ax.add_patch(FancyArrowPatch((sub_x + sub_w + 0.9, yneck_bot - 1.3), (sub_x + sub_w + 0.3, yneck_bot - 1.7),
                              arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=COLOR_OURS,
                              connectionstyle="arc3,rad=-0.3", zorder=1))

# ---------------- 3. Head (GANetHeadFast) ----------------
yhead_top = yneck_bot - 0.6
yhead_bot = yhead_top - 3.6
box(0.5, yhead_bot, 9.0, yhead_top - yhead_bot, "", COLOR_OURS, "#ffffff", fontsize=1)
ax.text(1.0, yhead_top - 0.3, "③ Head : GANetHeadFast  (ganet_head.py, 원본 뼈대 + 일부 우리 수정)",
        fontsize=11.5, weight="bold", color=COLOR_OURS)

# 좌측 브랜치: heatmap
box(0.9, yhead_bot + 0.35, 4.0, yhead_top - 0.55 - (yhead_bot + 0.35),
    "f_hm = features[hm_idx=0] = out0\n(LFA 적용된 feature)\n↓\nkeypts_head (CtnetHead)\n→ heatmap (num_classes, 92,160)",
    COLOR_OURS, COLOR_OURS_FILL, fontsize=9.8)

# 우측 브랜치: offset
box(5.1, yhead_bot + 0.35, 4.0, yhead_top - 0.55 - (yhead_bot + 0.35),
    "aux_feat (LFA 적용 \"전\" L0)\n(dcn_only_cls=True 이면 사용)\n↓\noffset_head / reg_head (CtnetHead)\n→ offset map ×2 (각 2채널, 92,160)",
    COLOR_OURS, COLOR_OURS_FILL, fontsize=9.8)

ax.text(5, yhead_bot + (yhead_top - yhead_bot) / 2 + 0.05, "X", fontsize=20, color="#c62828",
        ha="center", va="center", weight="bold", family="DejaVu Sans")
ax.text(5, yhead_bot + 0.18, "두 브랜치는 병렬 — concat/add로 합쳐지지 않음\n(최종 결합은 decode에서 산술적으로만 일어남)",
        ha="center", va="center", fontsize=9, color="#c62828", style="italic")

arrow(5, yhead_bot, 5, yhead_bot - 0.55)

# ---------------- 4. Decode ----------------
ydec_top = yhead_bot - 0.75
ydec_bot = ydec_top - 1.5
box(1.0, ydec_bot, 8.0, ydec_top - ydec_bot,
    "④ Decode : ktdet_decode_fast()  (ganet_head.py:129-175, 우리가 멀티클래스 대응으로 수정)\n\n"
    "heatmap에서 peak(키포인트) 탐색  +  coord_mat + pts_offset  (산술적으로 결합)",
    COLOR_OURS, COLOR_OURS_FILL, fontsize=10)
arrow(5, ydec_bot, 5, ydec_bot - 0.55)

# ---------------- 5. Post-process ----------------
ypp_top = ydec_bot - 0.75
ypp_bot = ypp_top - 1.6
box(1.0, ypp_bot, 8.0, ypp_top - ypp_bot,
    "⑤ Post-process : PostProcessor  (post_process.py, 거의 전부 우리가 작성)\n\n"
    "Savitzky-Golay smoothing (smooth_lane_points)  +  RANSAC 이상치 제거 (remove_lane_outliers)",
    COLOR_OURS, COLOR_OURS_FILL, fontsize=10)
arrow(5, ypp_bot, 5, ypp_bot - 0.55)

# ---------------- 최종 출력 ----------------
box(3.0, ypp_bot - 1.3, 4.0, 0.75, "최종 차선 polyline 좌표", COLOR_OURS, "#ffffff", fontsize=11, weight="bold")

ax.text(5, ypp_bot - 2.3,
        "※ 호출 체인 요약 : GANet.forward_test() → extract_feat()[backbone→neck] → bbox_head.forward_test() → post_processing()\n"
        "   (ganet.py:44-49)              (single_stage.py, 외부)                (ganet_head.py:123-127)          (post_process.py)",
        ha="center", fontsize=9.3, color="#555555")

plt.tight_layout()
out_path = "/tmp/claude-1006/-home-hsjeong-workspace/3c57c6da-a904-44ae-a952-64a93aab33ac/scratchpad/ganet_flow_diagram.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print("saved:", out_path)
