// 24번 그림(실험별 경로 과정도)을 편집 가능한 도형으로 옮긴 2장짜리 pptx.
// 실행: cd /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/pptx && npm install && NODE_PATH=$PWD/node_modules node build.js
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/home/hsjeong/.claude/skills/synced/a7e43bc8-ec8b-40dc-a6d1-044a4c28e08b_a813ceb5-7098-40f3-bfc9-b53a50c7421f/pptx/scripts/apply_theme.js");

const OUT = "/home/hsjeong/workspace/vision-det/GANet/output/07-cls2-P3제거특징맵분석/24. 경로별_과정도_실험별구조.pptx";
const FONT = "Malgun Gothic";
// 24번 그림과 같은 색
const THEME = {
  name: "GANet path", headFontFace: FONT, bodyFontFace: FONT,
  colors: { dk1: "222222", lt1: "FFFFFF", dk2: "444444", lt2: "F2F2F2",
    accent1: "4C78A8", accent2: "E45756", accent3: "F58518", accent4: "54A24B", accent5: "9E9E9E", accent6: "B279A2",
    hlink: "4C78A8", folHlink: "B279A2" },
};
const COL = { ORG: "4C78A8", NEW: "E45756", APT: "F58518", HEAD: "54A24B", GONE: "9E9E9E", TXT: "333333" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "GANet 실험별 경로";
pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: "FFFFFF" },
  slideNumber: { x: 0.45, y: 7.12, w: 0.5, h: 0.3, fontSize: 10, color: "888888" },
});

// 열: 입력, backbone, C3·C4·C5, C5 처리, +C4, +C3, DCN | head
const BW = 1.2, BH = 0.36, PITCH = 1.4, X0 = 0.45, HX = 11.62, HW = 1.26;
const bx = (i) => X0 + i * PITCH;          // 박스 왼쪽
let uid = 0;

function box(s, i, y, text, color, faded = false, w = BW, x = null) {
  const left = x === null ? bx(i) : x;
  s.addText(text, {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x: left, y, w, h: BH,
    fill: { color: faded ? "FFFFFF" : color }, line: { color, width: 1.5, dashType: faded ? "dash" : "solid" },
    color: faded ? color : "FFFFFF", bold: true, fontSize: text.length > 13 ? 9 : 10, align: "center", valign: "middle", margin: 0,
    objectName: `box${uid++}`,
  });
}

function line(s, x1, y1, x2, y2, color, opts = {}) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipV: (y2 < y1) !== (x2 < x1), line: { color, width: 1.75, dashType: opts.dash ? "dash" : "solid",
      endArrowType: opts.noArrow ? undefined : "triangle" }, objectName: `line${uid++}`,
  });
}

function text(s, t, x, y, w, opt = {}) {
  s.addText(t, { x, y, w, h: 0.26, fontSize: 11, color: COL.TXT, margin: 0, valign: "middle", isTextBox: true,
    objectName: `text${uid++}`, ...opt });
}

// 한 줄(실험 하나). y = 줄 맨 위
function row(s, y, r) {
  text(s, r.tag, X0, y, 2.4, { bold: true, fontSize: 13 });
  text(s, r.desc, 2.95, y - 0.03, 7.0, { fontSize: 11, valign: "top", h: 0.24 });
  text(s, r.score, 9.9, y, 3.0, { bold: true, align: "right", fontSize: 11 });
  const by = y + 0.3, cy = by + BH / 2;
  const mid = r.mid || ["SA → AC5", COL.ORG, false];
  const labels = ["원본 이미지", "backbone", r.p3 ? "C3·C4·C5" : "C4·C5", mid[0], "+C4"];
  labels.forEach((t, i) => box(s, i, by, t, i === 3 ? mid[1] : COL.ORG, i === 3 && mid[2]));
  for (let i = 1; i < 5; i++) line(s, bx(i - 1) + BW, cy, bx(i), cy, COL.ORG, { dash: mid[2] && (i === 3 || i === 4) });
  let dcnFrom;
  if (r.p3) {
    box(s, 5, by, "+C3 (DCN 전)", COL.ORG);
    line(s, bx(4) + BW, cy, bx(5), cy, COL.ORG);
    box(s, 6, by, "DCN → P3", COL.ORG);
    line(s, bx(5) + BW, cy, bx(6), cy, COL.ORG);
    dcnFrom = 5;
  } else {
    box(s, 5, by, "+C3 없음", COL.GONE, true);
    box(s, 6, by, "DCN → P4", COL.ORG);
    // +C4 에서 +C3 칸 위로 건너뛰어 DCN 으로
    const top = by - 0.05;
    line(s, bx(4) + BW * 0.75, by, bx(4) + BW * 0.75, top, COL.ORG, { noArrow: true });
    line(s, bx(4) + BW * 0.75, top, bx(6) + BW * 0.25, top, COL.ORG, { noArrow: true });
    line(s, bx(6) + BW * 0.25, top, bx(6) + BW * 0.25, by, COL.ORG);
    dcnFrom = 4;
  }
  const hy2 = by + 0.44;
  box(s, 0, by, "head: conf", COL.HEAD, false, HW, HX);
  box(s, 0, hy2, "head: offset", COL.HEAD, false, HW, HX);
  line(s, bx(6) + BW, cy, HX, cy, COL.ORG);
  line(s, bx(dcnFrom) + BW / 2, by + BH, HX, hy2 + BH / 2, COL.ORG);
  if (r.apt) {
    const ay = hy2, acy = ay + BH / 2;
    box(s, 1, ay, "칸별 엔트로피", COL.APT);
    box(s, 2, ay, "τ<6.5 → 2×2 병합", COL.APT, false, 1.3);
    line(s, bx(0) + BW / 2, by + BH, bx(0) + BW / 2, acy, COL.APT, { noArrow: true });
    line(s, bx(0) + BW / 2, acy, bx(1), acy, COL.APT);
    line(s, bx(1) + BW, acy, bx(2), acy, COL.APT);
    line(s, bx(2) + 1.3, acy, bx(3) + BW / 2, by + BH, COL.APT);
  }
}

function legend(s, items) {
  let x = 9.3;
  items.forEach(([c, t, dash]) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.47, w: 0.26, h: 0.16, fill: { color: dash ? "FFFFFF" : c },
      line: { color: c, width: 1, dashType: dash ? "dash" : "solid" }, objectName: `leg${uid++}` });
    text(s, t, x + 0.32, 0.42, 1.0, { fontSize: 10, h: 0.26 });
    x += 0.32 + t.length * 0.105 + 0.25;
  });
}

const ROW_H = 1.22, Y0 = 1.0;
function title(s, t, sub) {
  s.addText(t, { x: X0, y: 0.3, w: 8.4, h: 0.5, fontSize: 26, bold: true, color: COL.TXT, margin: 0, valign: "middle",
    isTextBox: true, objectName: "title" });
  text(s, sub, X0, 0.82, 11.0, { fontSize: 11, color: "777777" });
}

pres.addSection({ title: "덜어내기" });
let s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "덜어내기" });
title(s, "실험별 경로 ① P3·SA 덜어내기", "평균 F1 = ep100~150 평균, 시간 = V100 1장 forward. 1회 학습이라 0.01 이하 차이는 순위 확정 불가");
legend(s, [[COL.ORG, "기존 경로"], [COL.HEAD, "head"], [COL.GONE, "뺀 단계", true]]);
[
  { tag: "004 기존 ★", desc: "GANet 원래 경로. C5 전체에 SA(전역 self-attention). 현재 채택 모델", score: "평균 F1 .742 · 14.23ms", p3: true },
  { tag: "007 −SA", desc: "기존에서 SA만 뺌. C5가 바로 +C4로", score: "평균 F1 .752 · 13.90ms", p3: true, mid: ["SA 없음", COL.GONE, true] },
  { tag: "006 P3del", desc: "+C3 단계를 뺌. +C4(=P4, 입력의 1/16)에 DCN, offset은 DCN 전 +C4", score: "평균 F1 .758 · 10.86ms", p3: false },
  { tag: "008 P3del−SA", desc: "+C3 단계와 SA를 둘 다 뺌", score: "평균 F1 .774 · 10.42ms (1위)", p3: false, mid: ["SA 없음", COL.GONE, true] },
].forEach((r, k) => row(s, Y0 + 0.3 + k * 1.45, r));

pres.addSection({ title: "바꾸기·더하기" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "바꾸기·더하기" });
title(s, "실험별 경로 ② SA 바꾸기·APT 더하기", "비교 기준: 008 P3del−SA 평균 F1 .774 · 10.42ms — 다섯 가지 모두 이보다 낮음");
legend(s, [[COL.NEW, "SA → DA"], [COL.APT, "APT 병합"], [COL.GONE, "뺀 단계", true]]);
[
  { tag: "009 P3del+DA (query)", desc: "SA 자리에 DA(각 칸이 32곳만 봄). 위치 인코딩은 query에만 더함", score: "평균 F1 .763 · 12.12ms", mid: ["DA (query) → AC5", COL.NEW, false] },
  { tag: "010 P3del+DA (add)", desc: "009와 같은데 위치 인코딩을 특징 자체에 더함(SA와 같은 방식)", score: "평균 F1 .754 · 12.09ms", mid: ["DA (add) → AC5", COL.NEW, false] },
  { tag: "011 P3del+SA+APT", desc: "원본 이미지에서 단순한 2×2칸을 큰 토큰 1개로 합친 뒤 SA", score: "평균 F1 .769 · 13.61ms", mid: ["SA+APT → AC5", COL.APT, false], apt: true },
  { tag: "012 P3del+DA+APT", desc: "011과 같은 병합 뒤 DA(query). 합친 토큰은 2×2 가운데에서 32곳을 봄", score: "평균 F1 .762 · 14.56ms", mid: ["DA+APT → AC5", COL.APT, false], apt: true },
  { tag: "013 P3del−SA+APT", desc: "attention 없이 병합만: 합친 칸은 큰 토큰 값으로 2×2를 채움", score: "평균 F1 .765 · 11.69ms", mid: ["APT 병합만 → C5", COL.APT, false], apt: true },
].forEach((r, k) => row(s, Y0 + 0.2 + k * ROW_H, { p3: false, ...r }));

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(OUT);
})();
