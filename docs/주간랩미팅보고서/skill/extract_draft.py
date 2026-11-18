#!/usr/bin/env python3
"""사용자가 대충 만든 초안 PPT에서 글·표·노트·그림을 뽑아 draft.md + images/ 로 저장한다.

사용법:
    python3 extract_draft.py <초안.pptx> [출력폴더]

출력폴더 기본값: 초안 파일과 같은 폴더의 `<파일명>_추출/`
- draft.md  : 슬라이드별 글(위치 순), 표, 발표자 노트, 그림 파일명
- images/   : s<슬라이드>_<순번>.<확장자> (PPT 안에서 잘라 쓴 크롭은 반영하지 않은 원본 그림)
그룹 도형 안의 그림·글도 찾는다.
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu


def walk(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from walk(sh.shapes)
        else:
            yield sh


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / f"{src.stem}_추출"
    (out / "images").mkdir(parents=True, exist_ok=True)
    prs = Presentation(src)
    lines = [f"# {src.name} 추출 ({len(prs.slides)}장)\n"]
    for n, slide in enumerate(prs.slides, 1):
        lines.append(f"\n## 슬라이드 {n}")
        shapes = sorted(walk(slide.shapes), key=lambda s: (round(Emu(s.top or 0).inches), Emu(s.left or 0).inches))
        k = 0
        empty = True
        for sh in shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                k += 1
                name = f"s{n:02d}_{k}.{sh.image.ext}"
                (out / "images" / name).write_bytes(sh.image.blob)
                crop = ""
                if any((sh.crop_left, sh.crop_right, sh.crop_top, sh.crop_bottom)):
                    crop = f" (PPT에서 잘라 씀: 위 {sh.crop_top:.0%}, 아래 {sh.crop_bottom:.0%}, 왼 {sh.crop_left:.0%}, 오른 {sh.crop_right:.0%})"
                lines.append(f"- [그림] images/{name}{crop}")
                empty = False
            elif sh.has_table if hasattr(sh, "has_table") else False:
                lines.append("- [표]")
                for r in sh.table.rows:
                    lines.append("  | " + " | ".join(c.text.strip() for c in r.cells) + " |")
                empty = False
            elif sh.has_text_frame and sh.text_frame.text.strip():
                t = sh.text_frame.text.strip().replace("\n", " / ")
                lines.append(f"- [글] ({Emu(sh.left).inches:.1f}, {Emu(sh.top).inches:.1f}) {t}")
                empty = False
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            lines.append("- [노트] " + slide.notes_slide.notes_text_frame.text.strip().replace("\n", " / "))
            empty = False
        if empty:
            lines.append("- (빈 슬라이드)")
    (out / "draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"완료: {out}/draft.md, 그림 {len(list((out / 'images').iterdir()))}개")


if __name__ == "__main__":
    main()
