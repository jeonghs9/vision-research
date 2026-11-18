#!/usr/bin/env python3
"""주간 실험보고 PPT 생성기: content.json -> YYMMDD_이름_실험보고.pptx

디자인(색·폰트·배치)은 이 파일에 고정되어 있고, 매주 바뀌는 것은 content.json 뿐이다.
사진 중심 구성: 사진마다 제목+설명, 해석은 사진 슬라이드 아래에 붙는다.

사용법:
    python3 build_pptx.py <content.json> [-o 출력.pptx]

content.json 구조는 sample/content.json 참고.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

# ───────────────────────── 디자인 상수 ─────────────────────────
FONT = "맑은 고딕"
TITLE_C = RGBColor(0x1F, 0x2A, 0x44)   # 제목 (진한 남색, 거의 검정)
NAVY = RGBColor(0x1F, 0x3A, 0x5F)      # 제목 옆 막대, 번호 원
RED = RGBColor(0xC0, 0x1F, 0x1F)       # 목차에서 현재 주제 표시
TEXT = RGBColor(0x22, 0x22, 0x22)      # 본문
MUTED = RGBColor(0x6B, 0x72, 0x80)     # 보조 글씨, 바닥글
LINE = RGBColor(0xC8, 0xCD, 0xD4)      # 표 선
HEAD_BG = RGBColor(0xE9, 0xEC, 0xF0)   # 표 머리·해석 상자 배경
HILITE = RGBColor(0xFF, 0xF2, 0xCC)    # 표 강조 칸 (옅은 노랑)

SW, SH = Inches(13.333), Inches(7.5)   # 16:9
MX = Inches(0.6)
CONTENT_TOP = Inches(1.4)
CONTENT_BOTTOM = Inches(6.85)
CW = SW - 2 * MX

IMAGES_PER_SLIDE = 4
TABLE_ROWS_PER_SLIDE = 10


# ───────────────────────── 텍스트 도우미 ─────────────────────────
def _font(run, size, bold=False, color=TEXT):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set("typeface", FONT)


def _set_bullet(p, char="•", indent_in=0.22):
    ppr = p._p.get_or_add_pPr()
    ppr.set("marL", str(Inches(indent_in)))
    ppr.set("indent", str(-Inches(indent_in)))
    for child in list(ppr):
        if child.tag in (qn("a:buNone"), qn("a:buChar"), qn("a:buAutoNum")):
            ppr.remove(child)
    etree.SubElement(ppr, qn("a:buChar")).set("char", char)


def textbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None):
    tb = slide.shapes.add_textbox(int(x), int(y), int(w), int(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, pd in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = pd.get("runs") or [{"text": pd.get("text", "")}]
        for rd in runs:
            r = p.add_run()
            r.text = rd["text"]
            _font(r, rd.get("size", pd.get("size", 16)),
                  rd.get("bold", pd.get("bold", False)),
                  rd.get("color", pd.get("color", TEXT)))
        p.alignment = pd.get("align", PP_ALIGN.LEFT)
        if pd.get("bullet"):
            _set_bullet(p)
        p.space_after = Pt(pd.get("space_after", 6))
    return tb


def rect(slide, x, y, w, h, fill):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(x), int(y), int(w), int(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    st = s._element.find(qn("p:style"))   # 기본 그림자·테두리 제거
    if st is not None:
        s._element.remove(st)
    return s


def bullets(items, size=16, color=TEXT, space_after=8):
    return [{"text": t, "size": size, "color": color, "bullet": True,
             "space_after": space_after} for t in items]


def heading(text, size=18):
    return {"text": text, "size": size, "bold": True, "color": TITLE_C, "space_after": 6}


# ───────────────────────── 표 도우미 ─────────────────────────
def _cell_border(cell, color=LINE, width_pt=0.75, sides=("B",)):
    tcPr = cell._tc.get_or_add_tcPr()
    for side in ("L", "R", "T", "B"):
        tag = qn(f"a:ln{side}")
        old = tcPr.find(tag)
        if old is not None:
            tcPr.remove(old)
        ln = etree.SubElement(tcPr, tag)
        if side in sides:
            ln.set("w", str(int(Pt(width_pt))))
            sf = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", str(color))
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
    for f in [e for e in tcPr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]:
        tcPr.remove(f)
        tcPr.append(f)   # 선(lnX)이 채우기보다 앞에 와야 스키마에 맞는다


def _cell(cell, text, size, bold=False, color=TEXT, fill=None, align=PP_ALIGN.LEFT):
    cell.margin_left = cell.margin_right = Inches(0.12)
    cell.margin_top = cell.margin_bottom = Inches(0.06)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    r = p.add_run()
    r.text = str(text)
    _font(r, size, bold, color)
    p.alignment = align
    if fill is None:
        cell.fill.background()
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill


def _looks_numeric(v):
    s = str(v).replace(",", "").replace("%", "").replace("+", "").replace("−", "-")
    s = s.replace("→", "").replace(" ", "")
    try:
        float(s)
        return True
    except ValueError:
        return len(str(v)) <= 8


def table(slide, x, y, w, columns, rows, col_widths=None, size=14,
          highlight=(), row_h=Inches(0.45), align_numbers=True):
    """단순한 표: 옅은 회색 머리줄 + 얇은 가로선. highlight=[(r,c)] 는 본문 기준 0부터."""
    nr, nc = len(rows) + 1, len(columns)
    gf = slide.shapes.add_table(nr, nc, int(x), int(y), int(w), int(row_h * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        tblPr.remove(style)
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(w * cw / total))
    for r in range(nr):
        tbl.rows[r].height = int(row_h)
    hl = {tuple(h) for h in highlight}
    for c, name in enumerate(columns):
        _cell(tbl.cell(0, c), name, size, True, TITLE_C, HEAD_BG,
              PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT)
        _cell_border(tbl.cell(0, c), MUTED, 1.0, sides=("B",))
    for r, row in enumerate(rows):
        for c in range(nc):
            val = row[c] if c < len(row) else ""
            is_hl = (r, c) in hl
            numeric = align_numbers and c > 0 and _looks_numeric(val)
            _cell(tbl.cell(r + 1, c), val, size, bold=is_hl, color=TEXT,
                  fill=HILITE if is_hl else None,
                  align=PP_ALIGN.CENTER if numeric else PP_ALIGN.LEFT)
            _cell_border(tbl.cell(r + 1, c))
    return gf


# ───────────────────────── 슬라이드 공통 틀 ─────────────────────────
class Deck:
    def __init__(self, meta):
        self.meta = meta
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = SW, SH
        self.blank = self.prs.slide_layouts[6]
        self.footer_slides = []

    def new(self, title=None, kicker=None):
        s = self.prs.slides.add_slide(self.blank)
        if title:
            if kicker:
                textbox(s, MX, Inches(0.35), CW, Inches(0.3),
                        [{"text": kicker, "size": 12, "color": MUTED}])
            rect(s, MX, Inches(0.74), Inches(0.14), Inches(0.43), NAVY)
            textbox(s, MX + Inches(0.3), Inches(0.68), CW - Inches(0.3), Inches(0.55),
                    [{"text": title, "size": 26, "bold": True, "color": TITLE_C}],
                    anchor=MSO_ANCHOR.MIDDLE, name="Title")
            self.footer_slides.append(s)
        return s

    def finish(self, out):
        total = len(self.prs.slides)
        rd = self.meta["report_date"].replace("-", "")[2:]
        for s in self.footer_slides:
            idx = list(self.prs.slides).index(s) + 1
            textbox(s, MX, Inches(7.05), Inches(6), Inches(0.25),
                    [{"text": f"{self.meta['name']} · {rd} 실험보고", "size": 10, "color": MUTED}])
            textbox(s, SW - MX - Inches(2), Inches(7.05), Inches(2), Inches(0.25),
                    [{"text": f"{idx} / {total}", "size": 10, "color": MUTED,
                      "align": PP_ALIGN.RIGHT}])
        self.prs.save(out)


def fit_image(slide, path, x, y, w, h, center=False):
    """비율 유지하며 (x,y,w,h) 칸 안에 가로 가운데·세로 위쪽 정렬. 실제 (x,y,w,h) 반환."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    px, py = (int(x + (w - pw) // 2) if center else int(x)), int(y)   # 기본은 왼쪽 정렬 (설명 글과 줄을 맞춤)
    slide.shapes.add_picture(str(path), px, py, pw, ph)
    return px, py, pw, ph


# ───────────────────────── 슬라이드 종류 ─────────────────────────
def fmt_period(meta):
    s, e = date.fromisoformat(meta["period_start"]), date.fromisoformat(meta["period_end"])
    return f"{s:%Y.%m.%d} – {e:%m.%d}"


def slide_cover(deck, c):
    m = c["meta"]
    s = deck.new()
    textbox(s, Inches(1.0), Inches(2.0), Inches(11), Inches(1.0),
            [{"text": m.get("title", "주간 실험 보고"), "size": 40, "bold": True, "color": TITLE_C}])
    y = Inches(3.15)
    if m.get("subtitle"):
        textbox(s, Inches(1.0), y, Inches(11), Inches(0.5),
                [{"text": m["subtitle"], "size": 16, "color": MUTED}])
        y += Inches(0.75)
    textbox(s, Inches(1.0), y, Inches(11), Inches(0.5),
            [{"text": f"{fmt_period(m)}  ·  {m['name']}", "size": 18, "color": MUTED}])
    if m.get("cover_list", True):
        items = [{"runs": [{"text": f"{i}.  ", "bold": True}, {"text": e["title"]}],
                  "size": 18, "space_after": 8} for i, e in enumerate(c["experiments"], 1)]
        textbox(s, Inches(1.0), y + Inches(1.0), Inches(11), Inches(2.5), items)


def slide_agenda(deck, c, current):
    """목차: 현재 주제(current, 0부터)만 빨간색 굵게."""
    s = deck.new("목차")
    items = []
    for i, t in enumerate(c["agenda"]):
        on = (i == current)
        items.append({"text": f"{i + 1}.  {t}", "size": 22, "bold": on,
                      "color": RED if on else TEXT, "space_after": 16})
    textbox(s, MX + Inches(0.3), CONTENT_TOP + Inches(0.3), CW - Inches(0.6), Inches(4.5), items)


def slide_overview(deck, c):
    s = deck.new("이번 주 실험 한눈에 보기")
    rows = [[f"{i}", e.get("project", ""), e["title"], e.get("status", ""), e.get("one_line", "")]
            for i, e in enumerate(c["experiments"], 1)]
    table(s, MX, CONTENT_TOP, CW, ["#", "프로젝트", "실험", "상태", "핵심 결과"], rows,
          col_widths=[0.5, 2.0, 3.4, 1.2, 5.6], size=14, row_h=Inches(0.55),
          align_numbers=False)


def slide_background_setup(deck, n, e):
    s = deck.new(f"{e['title']} — 배경 · 목적 · 설정",
                 f"실험 {n} · {e.get('project', '')} · {e.get('status', '')}")
    lw = Inches(5.3)
    paras = [heading("배경")] + bullets(e.get("background", []), size=15, space_after=7)
    paras += [{"text": "", "size": 8}, heading("목적")]
    paras += bullets(e["purpose"], size=15, space_after=7)
    if e.get("hypothesis"):
        paras += [{"text": "", "size": 8}, heading("가설")]
        paras += bullets([e["hypothesis"]], size=15)
    textbox(s, MX, CONTENT_TOP, lw, CONTENT_BOTTOM - CONTENT_TOP, paras)
    # 설정 표: 구분 | 항목 | 설정 (구분 칸은 세로 병합)
    rows, groups = [], []
    for g in e["setup"]:
        start = len(rows)
        for k, v in g["items"]:
            rows.append([g["group"] if len(rows) == start else "", k, v])
        groups.append((start, len(rows) - 1))
    tx = MX + lw + Inches(0.5)
    tw = SW - MX - tx
    size = 14 if len(rows) <= 9 else 12
    row_h = Inches(0.45) if len(rows) <= 9 else Inches(0.38)
    gf = table(s, tx, CONTENT_TOP, tw, ["구분", "항목", "설정"], rows,
               col_widths=[2.1, 1.7, 3.2], size=size, row_h=row_h, align_numbers=False)
    tbl = gf.table
    for a, b in groups:
        if b > a:
            tbl.cell(a + 1, 0).merge(tbl.cell(b + 1, 0))
        cell = tbl.cell(a + 1, 0)
        _cell(cell, cell.text_frame.text, size, True, TITLE_C, None)
        _cell_border(cell)


def slide_results_table(deck, n, e, res, part=None):
    title = f"{e['title']} — 결과표"
    if res.get("title"):
        title += f": {res['title']}"
    if part:
        title += f" ({part})"
    s = deck.new(title, f"실험 {n} · {e.get('project', '')}")
    rows = res["rows"]
    size = 14 if len(rows) <= 7 else 12
    row_h = Inches(0.45) if len(rows) <= 7 else Inches(0.36)
    table(s, MX, CONTENT_TOP, CW, res["columns"], rows, col_widths=res.get("col_widths"),
          size=size, highlight=res.get("highlight", []), row_h=row_h)
    if res.get("note"):
        ny = CONTENT_TOP + row_h * (len(rows) + 1) + Inches(0.2)
        textbox(s, MX, ny, CW, Inches(0.5),
                [{"text": "※ " + res["note"], "size": 12, "color": MUTED}])


def numbered(slide, x, y, w, items, size=14, step=0.42, d=0.3):
    """남색 번호 원 + 문장. step = 줄 간격(인치). 긴 문장은 step을 키운다."""
    white = RGBColor(0xFF, 0xFF, 0xFF)
    for i, t in enumerate(items):
        yy = y + Inches(i * step)
        circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(x), int(yy), int(Inches(d)), int(Inches(d)))
        circ.fill.solid()
        circ.fill.fore_color.rgb = NAVY
        circ.line.fill.background()
        st = circ._element.find(qn("p:style"))
        if st is not None:
            circ._element.remove(st)
        tf = circ.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        r = tf.paragraphs[0].add_run()
        r.text = str(i + 1)
        _font(r, 12, True, white)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        textbox(slide, x + Inches(d + 0.15), yy, w - Inches(d + 0.15), Inches(step - 0.05),
                [{"text": t, "size": size}],
                anchor=MSO_ANCHOR.MIDDLE if step <= 0.45 else MSO_ANCHOR.TOP)


def slide_photos(deck, n, e, ph, imgs, part=None, last=True):
    """사진 슬라이드. layout: (기본) 사진 아래 설명 + 아래 해석 칸 / "side" 사진 왼쪽 + 오른쪽 설명·번호 / "wide" 큰 그림 가로 전체."""
    title = f"{e['title']} — {ph.get('title', '결과 사진')}"
    if ph.get("full_title"):
        title = ph["full_title"]
    if part:
        title += f" ({part})"
    s = deck.new(title, f"실험 {n} · {e.get('project', '')}")
    layout = ph.get("layout", "default")
    label = ph.get("interp_label", "해석")
    interp = ph.get("interpretation", []) if last else []
    full_h = CONTENT_BOTTOM - CONTENT_TOP
    k = len(imgs)

    if layout == "side" and k == 1:
        im = imgs[0]
        pw = CW * 0.64
        fit_image(s, im["path"], MX, CONTENT_TOP, pw, full_h)
        rx = MX + pw + Inches(0.4)
        rw = CW - pw - Inches(0.4)
        y = CONTENT_TOP
        paras = [{"text": im.get("caption", ""), "size": 15, "bold": True, "color": TITLE_C,
                  "space_after": 6}]
        if im.get("desc"):
            paras.append({"text": im["desc"], "size": 13, "color": TEXT})
        textbox(s, rx, y, rw, Inches(1.5), paras)
        if interp:
            ly = y + Inches(1.7) if im.get("desc") else y + Inches(0.9)
            textbox(s, rx, ly, rw, Inches(0.3),
                    [{"text": label, "size": 15, "bold": True, "color": TITLE_C}])
            numbered(s, rx, ly + Inches(0.5), rw, interp, size=14, step=0.95)
        return

    if layout == "wide" and k == 1:
        im = imgs[0]
        cap_h = Inches(0.4)
        interp_h = Inches(0.62 + 0.42 * len(interp)) if interp else 0
        area_h = full_h - cap_h - (interp_h + Inches(0.2) if interp else 0)
        _, py, _, ph_ = fit_image(s, im["path"], MX, CONTENT_TOP, CW, area_h, center=True)
        textbox(s, MX, py + ph_ + Inches(0.1), CW, cap_h,
                [{"text": im.get("caption", ""), "size": 12, "color": MUTED,
                  "align": PP_ALIGN.CENTER}])
        if interp:
            _interp_box(s, label, interp, CONTENT_BOTTOM - interp_h, interp_h)
        return

    interp_h = Inches(0.62 + 0.42 * len(interp)) if interp else 0
    gap = Inches(0.3)
    area_h = full_h - (interp_h + gap if interp else 0)
    cols, rows_ = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2)}[k]
    desc_h = Inches(0.95) if k <= 3 else Inches(0.8)
    cw = (CW - gap * (cols - 1)) / cols
    ch = (area_h - gap * (rows_ - 1)) / rows_
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x = MX + c * (cw + gap)
        y = CONTENT_TOP + r * (ch + gap)
        if k in (1, 4):   # 사진 옆에 설명: 1장이면 크게, 4장이면 칸마다 사진 왼쪽·설명 오른쪽
            pw = (CW * 0.68) if k == 1 else (cw * 0.58)
            fit_image(s, im["path"], x, y, pw, ch)
            tx = x + pw + Inches(0.25)
            tw_ = (CW if k == 1 else cw) - pw - Inches(0.25)
            textbox(s, tx, y, tw_, ch,
                    [{"text": im.get("caption", ""), "size": 15 if k == 1 else 13, "bold": True,
                      "color": TITLE_C, "space_after": 6},
                     {"text": im.get("desc", ""), "size": 14 if k == 1 else 12, "color": TEXT}])
            continue
        _, py, _, ph_ = fit_image(s, im["path"], x, y, cw, ch - desc_h - Inches(0.1))
        textbox(s, x, py + ph_ + Inches(0.1), cw, desc_h,
                [{"text": im.get("caption", ""), "size": 13, "bold": True,
                  "color": TITLE_C, "space_after": 3},
                 {"text": im.get("desc", ""), "size": 12, "color": TEXT}])
    if interp:
        _interp_box(s, label, interp, CONTENT_BOTTOM - interp_h, interp_h)


def _interp_box(s, label, interp, by, interp_h):
    rect(s, MX, by, CW, interp_h, HEAD_BG)
    pad = Inches(0.25)
    textbox(s, MX + pad, by + Inches(0.14), Inches(3), Inches(0.3),
            [{"text": label, "size": 14, "bold": True, "color": TITLE_C}])
    numbered(s, MX + pad, by + Inches(0.52), CW - 2 * pad, interp)


def slide_summary(deck, c):
    sm = c.get("summary", {})
    s = deck.new("종합 · 다음 계획")
    lw = (CW - Inches(0.6)) / 2
    left = [heading("이번 주 결론")] + bullets(sm.get("conclusions", []), size=15, space_after=8)
    left += [{"text": "", "size": 10}, heading("이슈 · 지원 요청")]
    left += bullets(sm.get("issues", []) or ["없음"], size=15, space_after=8)
    textbox(s, MX, CONTENT_TOP, lw, CONTENT_BOTTOM - CONTENT_TOP, left)
    right = [heading("다음 실험 계획")]
    for i, e in enumerate(c["experiments"], 1):
        if e.get("next"):
            right.append({"text": f"실험 {i}. {e['title']}", "size": 14, "bold": True,
                          "color": MUTED, "space_after": 3})
            right += bullets(e["next"], size=15, space_after=6)
    if sm.get("next_week"):
        right.append({"text": "그 외", "size": 14, "bold": True, "color": MUTED, "space_after": 3})
        right += bullets(sm["next_week"], size=15, space_after=6)
    textbox(s, MX + lw + Inches(0.6), CONTENT_TOP, lw, CONTENT_BOTTOM - CONTENT_TOP, right)


# ───────────────────────── 조립 ─────────────────────────
def chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def build(content, out, base_dir):
    deck = Deck(content["meta"])
    slide_cover(deck, content)
    if content.get("agenda"):
        slide_agenda(deck, content, content.get("agenda_start", 0))
    if content["meta"].get("show_overview", True):
        slide_overview(deck, content)
    for n, e in enumerate(content["experiments"], 1):
        slide_background_setup(deck, n, e)
        def do_results(res):
            parts = chunk(res["rows"], TABLE_ROWS_PER_SLIDE)
            for j, rows in enumerate(parts):
                lo, hi = j * TABLE_ROWS_PER_SLIDE, (j + 1) * TABLE_ROWS_PER_SLIDE
                r2 = dict(res, rows=rows,
                          highlight=[(h[0] - lo, h[1]) for h in res.get("highlight", [])
                                     if lo <= h[0] < hi])
                slide_results_table(deck, n, e, r2,
                                    f"{j + 1}/{len(parts)}" if len(parts) > 1 else None)

        def do_photos(ph):
            imgs = []
            for im in ph["images"]:
                p = Path(im["path"])
                if not p.is_absolute():
                    p = base_dir / p
                if not p.exists():
                    sys.exit(f"[오류] 이미지가 없습니다: {p}")
                imgs.append(dict(im, path=p))
            groups = chunk(imgs, IMAGES_PER_SLIDE)
            for j, g in enumerate(groups):
                slide_photos(deck, n, e, ph, g,
                             f"{j + 1}/{len(groups)}" if len(groups) > 1 else None,
                             last=(j == len(groups) - 1))

        # 슬라이드 순서: e["order"] = ["photo:0", "results:0", ...] 로 지정, 없으면 결과표 → 사진
        order = e.get("order") or ([f"results:{i}" for i in range(len(e.get("results", [])))]
                                   + [f"photo:{i}" for i in range(len(e.get("photos", [])))])
        for item in order:
            kind, idx = item.split(":")
            if kind == "agenda":
                slide_agenda(deck, content, int(idx))
            elif kind == "results":
                do_results(e["results"][int(idx)])
            else:
                do_photos(e["photos"][int(idx)])
    slide_summary(deck, content)
    if content.get("agenda"):
        slide_agenda(deck, content, content.get("agenda_next", 1))
    deck.finish(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("content", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    content = json.loads(a.content.read_text(encoding="utf-8"))
    m = content["meta"]
    out = a.out or a.content.parent / f"{m['report_date'].replace('-', '')[2:]}_{m['name']}_실험보고.pptx"
    build(content, out, a.content.parent.resolve())
    print(f"생성 완료: {out}  (슬라이드 {len(Presentation(out).slides)}장)")


if __name__ == "__main__":
    main()
