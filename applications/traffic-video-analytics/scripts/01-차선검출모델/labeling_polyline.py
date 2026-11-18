import sys
import os
import shutil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox, QCheckBox
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF, QCursor, QFont
from PyQt6.QtCore import Qt, QRectF, QPointF


# 폴리라인(차선) 모드 클래스 이름.
# 차선은 "어디에 있나" 보다 "어떤 종류인가" 가 판정을 좌우한다 —
# 실선을 넘으면 위반, 점선은 정상이다.
POLYLINE_CLASS_NAMES = {
    0: "실선",
    1: "점선",
    2: "중앙분리대",
    3: "갓길경계",
}


# ==========================
#   ImageLabel 위젯
# ==========================
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 이미지 상태
        self.image: QPixmap | None = None
        self.img_w = 0
        self.img_h = 0

        # 모드: "bbox" / "seg" / "polyline"
        self.mode = "bbox"

        # BBox 데이터: (cls, x1, y1, x2, y2) in pixel
        self.boxes: list[tuple[int, float, float, float, float]] = []
        self.box_start: tuple[float, float] | None = None
        self.box_preview_end: tuple[float, float] | None = None

        # Seg 데이터: (cls, [(x,y), ...]) in pixel
        self.polygons: list[tuple[int, list[tuple[float, float]]]] = []
        self.current_polygon: list[tuple[float, float]] = []
        self.seg_preview_point: tuple[float, float] | None = None

        # Polyline 데이터: (cls, [(x,y), ...]) in pixel — 닫히지 않는 열린 선
        # 차선은 면이 아니라 선이므로 채우지 않고 2점부터 유효하다.
        self.polylines: list[tuple[int, list[tuple[float, float]]]] = []
        self.current_polyline: list[tuple[float, float]] = []
        self.pl_preview_point: tuple[float, float] | None = None
        # 확정된 선의 꼭짓점을 끌어 옮기기 위한 상태 (선 index, 점 index)
        self.pl_drag: tuple[int, int] | None = None
        self.pl_line_width = 3
        self.pl_vertex_radius = 4.0
        self.pl_hit_px = 8.0          # 꼭짓점/선 집기 허용치 (화면 픽셀)

        # Seg·Polyline 공용 우클릭 드래그 삭제 범위 (모드는 배타적이라 공유한다)
        self.seg_delete_dragging = False
        self.seg_delete_start: tuple[float, float] | None = None
        self.seg_delete_end: tuple[float, float] | None = None
        self.seg_delete_drag_threshold = 5.0

        # 줌/팬
        self.zoom_factor = 1.0
        self.base_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pan_dx = 0.0
        self.pan_dy = 0.0
        self.panning = False
        self.pan_start = None

        # 현재 클래스
        self.current_class = 0
        self.class_colors: dict[int, Qt.GlobalColor] = {
            0: Qt.GlobalColor.yellow,
            1: Qt.GlobalColor.green,
            2: Qt.GlobalColor.red,
            3: Qt.GlobalColor.blue,
            4: Qt.GlobalColor.cyan,
            5: Qt.GlobalColor.magenta,
        }

        self.setMouseTracking(True)

    # ----------------------------
    #   기본 설정/조회 함수
    # ----------------------------
    def set_image(self, pixmap: QPixmap | None):
        self.image = pixmap
        if pixmap is not None:
            self.img_w = pixmap.width()
            self.img_h = pixmap.height()
        else:
            self.img_w = self.img_h = 0

        # 뷰 리셋
        self.zoom_factor = 1.0
        self.pan_dx = self.pan_dy = 0.0
        self.box_start = None
        self.box_preview_end = None
        self.current_polygon = []
        self.seg_preview_point = None
        self.current_polyline = []
        self.pl_preview_point = None
        self.pl_drag = None
        self.seg_delete_dragging = False
        self.seg_delete_start = None
        self.seg_delete_end = None

        self.update()

    def set_current_class(self, cls_id: int):
        self.current_class = int(cls_id)

    def set_boxes(self, boxes):
        self.boxes = list(boxes)
        # 박스 설정 시 진행 중이던 것 초기화
        self.box_start = None
        self.box_preview_end = None
        self.update()

    def set_polygons(self, polygons):
        # 깊은 복사 수준까진 아니어도 list copy
        self.polygons = [(cls, list(pts)) for cls, pts in polygons]
        self.current_polygon = []
        self.seg_preview_point = None
        self.update()

    def get_boxes(self):
        return list(self.boxes)

    def get_polygons(self):
        return [(cls, list(pts)) for cls, pts in self.polygons]

    def set_polylines(self, polylines):
        self.polylines = [(cls, list(pts)) for cls, pts in polylines]
        self.current_polyline = []
        self.pl_preview_point = None
        self.pl_drag = None
        self.update()

    def get_polylines(self):
        return [(cls, list(pts)) for cls, pts in self.polylines]

    # ----------------------------
    #   마우스 이벤트
    # ----------------------------
    def mousePressEvent(self, event):
        if self.image is None:
            return

        mods = event.modifiers()
        pos = event.position()

        # Ctrl + 좌클릭 -> 패닝 시작
        if (
            event.button() == Qt.MouseButton.LeftButton
            and (mods & Qt.KeyboardModifier.ControlModifier)
        ):
            self.panning = True
            self.pan_start = pos
            return

        x, y = self._map_to_image(pos)
        if x is None:
            return

        # 우클릭
        if event.button() == Qt.MouseButton.RightButton:
            if self.mode == "bbox":
                self._delete_bbox_at(x, y)
            else:
                # seg 와 polyline 모두 같은 방식: 클릭=하나 삭제, 드래그=범위 삭제
                self.seg_delete_dragging = True
                self.seg_delete_start = (x, y)
                self.seg_delete_end = (x, y)
                self.update()
            return

        # 좌클릭
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "bbox":
                self._bbox_click(x, y)
            elif self.mode == "polyline":
                # 확정된 선의 꼭짓점 근처를 누르면 **점 추가가 아니라 끌기**로 본다.
                # 차선은 위치가 몇 px 로 갈려 미세 조정이 잦다.
                hit = self._find_polyline_vertex(x, y)
                if hit is not None and not self.current_polyline:
                    self.pl_drag = hit
                else:
                    self._polyline_add_point(x, y)
            else:
                self._seg_add_point(x, y)

    def mouseDoubleClickEvent(self, event):
        # 세그 모드에서 더블클릭 -> 폴리곤 닫기
        if (
            self.mode == "seg"
            and event.button() == Qt.MouseButton.LeftButton
            and len(self.current_polygon) >= 3
        ):
            self.finish_polygon()
            event.accept()
        # 폴리라인 모드: 더블클릭 -> 선 확정 (2점부터 유효)
        elif (
            self.mode == "polyline"
            and event.button() == Qt.MouseButton.LeftButton
            and len(self.current_polyline) >= 2
        ):
            self.finish_polyline()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning and self.pan_start is not None:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            self.pan_dx += delta.x()
            self.pan_dy += delta.y()
            self.update()
            return

        if self.image is None:
            return

        x, y = self._map_to_image(event.position())
        if x is None:
            return

        # Seg·Polyline 우클릭 드래그 삭제 범위 갱신
        if (
            self.mode in ("seg", "polyline")
            and self.seg_delete_dragging
            and (event.buttons() & Qt.MouseButton.RightButton)
        ):
            self.seg_delete_end = (x, y)
            self.update()
            return

        # BBox 프리뷰
        if self.mode == "bbox" and self.box_start is not None:
            self.box_preview_end = (x, y)
            self.update()
            return

        # Seg 프리뷰: 마지막 점에서 마우스까지 라인
        if self.mode == "seg" and len(self.current_polygon) > 0:
            self.seg_preview_point = (x, y)
            self.update()
            return

        if self.mode == "polyline":
            # 꼭짓점을 끄는 중이면 좌표 갱신
            if self.pl_drag is not None:
                li, pi = self.pl_drag
                if li < len(self.polylines) and pi < len(self.polylines[li][1]):
                    self.polylines[li][1][pi] = (x, y)
                    self.update()
                return
            if len(self.current_polyline) > 0:
                self.pl_preview_point = (x, y)
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.panning:
            self.panning = False
            self.pan_start = None
            return

        if event.button() == Qt.MouseButton.LeftButton and self.pl_drag is not None:
            self.pl_drag = None
            return

        if (
            event.button() == Qt.MouseButton.RightButton
            and self.mode in ("seg", "polyline")
            and self.seg_delete_dragging
            and self.seg_delete_start is not None
        ):
            end_x, end_y = self._map_to_image(event.position())

            if end_x is not None:
                self.seg_delete_end = (end_x, end_y)

            start = self.seg_delete_start
            end = self.seg_delete_end or start

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            drag_distance2 = dx * dx + dy * dy
            threshold2 = self.seg_delete_drag_threshold ** 2

            self.seg_delete_dragging = False
            self.seg_delete_start = None
            self.seg_delete_end = None

            if self.mode == "polyline":
                if drag_distance2 <= threshold2:
                    self._delete_polyline_at(start[0], start[1])
                else:
                    self._delete_polylines_in_rect(start, end)
            elif drag_distance2 <= threshold2:
                self._delete_seg_at(start[0], start[1])
            else:
                self._delete_segs_in_rect(start, end)

            self.update()
            return

    # ----------------------------
    #   휠 줌
    # ----------------------------
    def wheelEvent(self, event):
        if self.image is None or self.img_w == 0 or self.img_h == 0:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1
        elif delta < 0:
            self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
        self.update()

    # ----------------------------
    #   그리기
    # ----------------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.image is None or self.img_w == 0 or self.img_h == 0:
            return

        label_w = self.width()
        label_h = self.height()

        self.base_scale = min(label_w / self.img_w, label_h / self.img_h)
        self.scale = self.base_scale * self.zoom_factor

        draw_w = self.img_w * self.scale
        draw_h = self.img_h * self.scale

        self.offset_x = (label_w - draw_w) / 2 + self.pan_dx
        self.offset_y = (label_h - draw_h) / 2 + self.pan_dy

        # 이미지
        target_rect = QRectF(self.offset_x, self.offset_y, draw_w, draw_h)
        source_rect = QRectF(0, 0, self.img_w, self.img_h)
        painter.drawPixmap(target_rect, self.image, source_rect)

        # --------- Seg: 확정된 폴리곤 채우기 + 테두리 ---------
        for cls_id, pts in self.polygons:
            if len(pts) < 3:
                continue
            color = QColor(self.class_colors.get(cls_id, Qt.GlobalColor.red))
            fill_color = QColor(color.red(), color.green(), color.blue(), 80)  # 투명도 80

            qpoints = []
            for (x, y) in pts:
                px = self.offset_x + x * self.scale
                py = self.offset_y + y * self.scale
                qpoints.append(QPointF(px, py))

            poly = QPolygonF(qpoints)
            painter.setBrush(fill_color)
            painter.setPen(QPen(color, 2))
            painter.drawPolygon(poly)

        # --------- BBox 박스들 ---------
        for cls_id, x1, y1, x2, y2 in self.boxes:
            color = self.class_colors.get(cls_id, Qt.GlobalColor.red)
            pen = QPen(color, 2)
            painter.setPen(pen)

            rx1 = self.offset_x + x1 * self.scale
            ry1 = self.offset_y + y1 * self.scale
            rx2 = self.offset_x + x2 * self.scale
            ry2 = self.offset_y + y2 * self.scale
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))

        # --------- BBox 프리뷰 박스 ---------
        if self.mode == "bbox" and self.box_start is not None and self.box_preview_end is not None:
            x1, y1 = self.box_start
            x2, y2 = self.box_preview_end
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            color = self.class_colors.get(self.current_class, Qt.GlobalColor.white)
            pen = QPen(color, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)

            rx1 = self.offset_x + x_min * self.scale
            ry1 = self.offset_y + y_min * self.scale
            rw = (x_max - x_min) * self.scale
            rh = (y_max - y_min) * self.scale
            painter.drawRect(QRectF(rx1, ry1, rw, rh))

        # --------- Seg·Polyline: 우클릭 드래그 삭제 범위 ---------
        if (
            self.mode in ("seg", "polyline")
            and self.seg_delete_dragging
            and self.seg_delete_start is not None
            and self.seg_delete_end is not None
        ):
            x1, y1 = self.seg_delete_start
            x2, y2 = self.seg_delete_end
            x_min, x_max = sorted((x1, x2))
            y_min, y_max = sorted((y1, y2))

            rx = self.offset_x + x_min * self.scale
            ry = self.offset_y + y_min * self.scale
            rw = (x_max - x_min) * self.scale
            rh = (y_max - y_min) * self.scale

            delete_color = QColor(255, 80, 80)
            delete_fill = QColor(255, 80, 80, 55)
            painter.setPen(QPen(delete_color, 2, Qt.PenStyle.DashLine))
            painter.setBrush(delete_fill)
            painter.drawRect(QRectF(rx, ry, rw, rh))

        # --------- Seg: 현재 그리고 있는 폴리곤 선 ---------
        if self.mode == "seg" and len(self.current_polygon) > 0:
            color = self.class_colors.get(self.current_class, Qt.GlobalColor.white)
            pen = QPen(color, 2)
            painter.setPen(pen)

            # 클릭으로 만든 점들 사이 선
            for i in range(len(self.current_polygon) - 1):
                x1, y1 = self.current_polygon[i]
                x2, y2 = self.current_polygon[i + 1]
                sx1 = self.offset_x + x1 * self.scale
                sy1 = self.offset_y + y1 * self.scale
                sx2 = self.offset_x + x2 * self.scale
                sy2 = self.offset_y + y2 * self.scale
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

            # 마지막 점 -> 마우스 위치 프리뷰
            if self.seg_preview_point is not None:
                x1, y1 = self.current_polygon[-1]
                x2, y2 = self.seg_preview_point
                sx1 = self.offset_x + x1 * self.scale
                sy1 = self.offset_y + y1 * self.scale
                sx2 = self.offset_x + x2 * self.scale
                sy2 = self.offset_y + y2 * self.scale
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))


        # --------- Polyline: 확정된 선들 ---------
        # 차선은 화면에서 가늘어(5~30px) 얇게 그리면 원본에 묻힌다.
        # 선은 굵게, 꼭짓점은 원으로 그려 잡기 쉽게 한다.
        if self.mode == "polyline":
            for cls_id, pts in self.polylines:
                if len(pts) < 2:
                    continue
                color = QColor(self.class_colors.get(cls_id, Qt.GlobalColor.red))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(color, self.pl_line_width))
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i + 1]
                    painter.drawLine(
                        QPointF(self.offset_x + x1 * self.scale,
                                self.offset_y + y1 * self.scale),
                        QPointF(self.offset_x + x2 * self.scale,
                                self.offset_y + y2 * self.scale))
                painter.setBrush(color)
                for (x, y) in pts:
                    painter.drawEllipse(
                        QPointF(self.offset_x + x * self.scale,
                                self.offset_y + y * self.scale),
                        self.pl_vertex_radius, self.pl_vertex_radius)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # --------- Polyline: 현재 그리고 있는 선 ---------
            if self.current_polyline:
                color = QColor(self.class_colors.get(self.current_class,
                                                     Qt.GlobalColor.white))
                painter.setPen(QPen(color, self.pl_line_width, Qt.PenStyle.DashLine))
                for i in range(len(self.current_polyline) - 1):
                    x1, y1 = self.current_polyline[i]
                    x2, y2 = self.current_polyline[i + 1]
                    painter.drawLine(
                        QPointF(self.offset_x + x1 * self.scale,
                                self.offset_y + y1 * self.scale),
                        QPointF(self.offset_x + x2 * self.scale,
                                self.offset_y + y2 * self.scale))
                if self.pl_preview_point is not None:
                    x1, y1 = self.current_polyline[-1]
                    x2, y2 = self.pl_preview_point
                    painter.drawLine(
                        QPointF(self.offset_x + x1 * self.scale,
                                self.offset_y + y1 * self.scale),
                        QPointF(self.offset_x + x2 * self.scale,
                                self.offset_y + y2 * self.scale))
                painter.setBrush(color)
                painter.setPen(QPen(color, 1))
                for (x, y) in self.current_polyline:
                    painter.drawEllipse(
                        QPointF(self.offset_x + x * self.scale,
                                self.offset_y + y * self.scale),
                        self.pl_vertex_radius, self.pl_vertex_radius)
                painter.setBrush(Qt.BrushStyle.NoBrush)

    # ----------------------------
    #   BBox 모드 로직
    # ----------------------------
    def _bbox_click(self, x, y):
        # 첫 클릭: 시작점
        if self.box_start is None:
            self.box_start = (x, y)
            self.box_preview_end = None
        # 두 번째 클릭: 박스 확정
        else:
            x1, y1 = self.box_start
            x2, y2 = x, y
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            # 너무 작은 박스는 무시(옵션)
            if (x_max - x_min) > 1 and (y_max - y_min) > 1:
                self.boxes.append((self.current_class, x_min, y_min, x_max, y_max))

            self.box_start = None
            self.box_preview_end = None
            self.update()

    def _delete_bbox_at(self, x, y):
        # 최근 것부터 탐색
        for i in range(len(self.boxes) - 1, -1, -1):
            cls_id, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                del self.boxes[i]
                break
        self.update()

    def undo_bbox(self):
        if self.box_start is not None:
            # 진행 중이던 박스 취소
            self.box_start = None
            self.box_preview_end = None
        elif self.boxes:
            self.boxes.pop()
        self.update()

    # ----------------------------
    #   Seg 모드 로직
    # ----------------------------
    def _seg_add_point(self, x, y):
        self.current_polygon.append((x, y))
        self.seg_preview_point = None
        self.update()

    def finish_polygon(self):
        if len(self.current_polygon) >= 3:
            self.polygons.append((self.current_class, list(self.current_polygon)))
        self.current_polygon = []
        self.seg_preview_point = None
        self.update()

    def _delete_seg_at(self, x, y):
        """
        Seg 모드에서 우클릭한 위치의 폴리곤 전체를 삭제한다.
        여러 폴리곤이 겹친 경우 가장 나중에 생성된 폴리곤을 삭제한다.
        완성된 폴리곤을 누르지 않았고 현재 작성 중인 폴리곤이 있으면
        작성 중인 폴리곤을 전체 취소한다.
        """
        # 가장 최근에 생성된 폴리곤부터 검사
        for poly_idx in range(len(self.polygons) - 1, -1, -1):
            _cls_id, pts = self.polygons[poly_idx]

            if len(pts) < 3:
                continue

            if self._point_in_polygon(x, y, pts):
                del self.polygons[poly_idx]
                self.update()
                return

        # 완성된 폴리곤을 선택하지 못한 경우,
        # 현재 작성 중인 폴리곤이 있다면 전체 취소
        if self.current_polygon:
            self.current_polygon = []
            self.seg_preview_point = None
            self.update()

    def _delete_segs_in_rect(self, start, end):
        """
        우클릭 드래그 사각형과 조금이라도 겹치는 모든 완성된 Seg 폴리곤을 삭제한다.
        """
        x1, y1 = start
        x2, y2 = end
        rect = (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )

        self.polygons = [
            (cls_id, pts)
            for cls_id, pts in self.polygons
            if not self._polygon_intersects_rect(pts, rect)
        ]

    @classmethod
    def _polygon_intersects_rect(cls, points, rect):
        if len(points) < 3:
            return False

        left, top, right, bottom = rect

        poly_xs = [p[0] for p in points]
        poly_ys = [p[1] for p in points]

        if (
            max(poly_xs) < left
            or min(poly_xs) > right
            or max(poly_ys) < top
            or min(poly_ys) > bottom
        ):
            return False

        for px, py in points:
            if left <= px <= right and top <= py <= bottom:
                return True

        rect_corners = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]

        for cx, cy in rect_corners:
            if cls._point_in_polygon(cx, cy, points):
                return True

        rect_edges = [
            ((left, top), (right, top)),
            ((right, top), (right, bottom)),
            ((right, bottom), (left, bottom)),
            ((left, bottom), (left, top)),
        ]

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            for q1, q2 in rect_edges:
                if cls._segments_intersect(p1, p2, q1, q2):
                    return True

        return False

    @staticmethod
    def _segments_intersect(p1, p2, q1, q2):
        eps = 1e-9

        def orientation(a, b, c):
            value = (
                (b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0])
            )
            if abs(value) <= eps:
                return 0
            return 1 if value > 0 else -1

        def on_segment(a, b, c):
            return (
                min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
                and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
            )

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, p2, q2):
            return True
        if o3 == 0 and on_segment(q1, q2, p1):
            return True
        if o4 == 0 and on_segment(q1, q2, p2):
            return True

        return False

    @staticmethod
    def _point_in_polygon(x, y, points):
        """Ray casting 방식으로 점이 폴리곤 내부인지 판별한다."""
        inside = False
        count = len(points)
        j = count - 1

        for i in range(count):
            xi, yi = points[i]
            xj, yj = points[j]

            if (yi > y) != (yj > y):
                intersect_x = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < intersect_x:
                    inside = not inside

            j = i

        return inside

    def undo_seg(self):
        # 진행 중인 폴리곤이면 마지막 점 삭제
        if self.current_polygon:
            self.current_polygon.pop()
            self.seg_preview_point = None
        # 아니면 마지막 확정 폴리곤 삭제
        elif self.polygons:
            self.polygons.pop()
        self.update()

    # ----------------------------
    #   Polyline 모드 로직
    # ----------------------------
    def _polyline_add_point(self, x, y):
        self.current_polyline.append((x, y))
        self.pl_preview_point = None
        self.update()

    def finish_polyline(self):
        """진행 중인 선을 확정한다. 폴리곤과 달리 **2점부터** 유효하다."""
        if len(self.current_polyline) >= 2:
            self.polylines.append((self.current_class, list(self.current_polyline)))
        self.current_polyline = []
        self.pl_preview_point = None
        self.update()

    def _hit_tol(self):
        """집기 허용치를 이미지 좌표로 환산한다.

        **화면 픽셀** 기준으로 잡아야 한다. 이미지 픽셀로 고정하면 확대했을 때
        지나치게 예민하고 축소했을 때 못 잡는다.
        """
        return self.pl_hit_px / max(self.scale, 1e-6)

    def _find_polyline_vertex(self, x, y):
        """허용치 안의 가장 가까운 꼭짓점. 반환: (선 index, 점 index) 또는 None"""
        thr2 = self._hit_tol() ** 2
        best, best_d = None, thr2
        for li, (_cls, pts) in enumerate(self.polylines):
            for pi, (px, py) in enumerate(pts):
                d = (x - px) ** 2 + (y - py) ** 2
                if d <= best_d:
                    best_d, best = d, (li, pi)
        return best

    @staticmethod
    def _point_seg_dist2(x, y, p1, p2):
        """점에서 선분까지 거리의 제곱."""
        ax, ay = p1
        bx, by = p2
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return (x - ax) ** 2 + (y - ay) ** 2
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        cx, cy = ax + t * dx, ay + t * dy
        return (x - cx) ** 2 + (y - cy) ** 2

    def _delete_polyline_at(self, x, y):
        """우클릭한 위치의 선 **전체**를 삭제한다 (seg 모드와 같은 방식).

        면이 아니라 선이므로 "안에 들어왔나" 가 아니라 "선에 가까운가" 로 판정한다.
        완성된 선을 못 집었고 작성 중인 선이 있으면 그것을 취소한다.
        """
        thr2 = self._hit_tol() ** 2
        for li in range(len(self.polylines) - 1, -1, -1):   # 최근 것부터
            pts = self.polylines[li][1]
            for i in range(len(pts) - 1):
                if self._point_seg_dist2(x, y, pts[i], pts[i + 1]) <= thr2:
                    del self.polylines[li]
                    self.update()
                    return
        if self.current_polyline:
            self.current_polyline = []
            self.pl_preview_point = None
            self.update()

    def _delete_polylines_in_rect(self, start, end):
        """드래그 사각형과 조금이라도 겹치는 선을 모두 삭제한다."""
        x1, y1 = start
        x2, y2 = end
        rx1, ry1 = min(x1, x2), min(y1, y2)
        rx2, ry2 = max(x1, x2), max(y1, y2)

        def hits(pts):
            for (px, py) in pts:                     # 꼭짓점이 사각형 안
                if rx1 <= px <= rx2 and ry1 <= py <= ry2:
                    return True
            corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
            for i in range(len(pts) - 1):            # 변과 교차
                for j in range(4):
                    if self._segments_intersect(pts[i], pts[i + 1],
                                                corners[j], corners[(j + 1) % 4]):
                        return True
            return False

        self.polylines = [(c, pts) for c, pts in self.polylines if not hits(pts)]

    def undo_polyline(self):
        if self.current_polyline:
            self.current_polyline.pop()
            self.pl_preview_point = None
        elif self.polylines:
            self.polylines.pop()
        self.update()

    # ----------------------------
    #   좌표 변환
    # ----------------------------
    def _map_to_image(self, pos):
        if self.image is None or self.scale == 0:
            return None, None
        x = (pos.x() - self.offset_x) / self.scale
        y = (pos.y() - self.offset_y) / self.scale
        if x < 0 or y < 0 or x > self.img_w or y > self.img_h:
            return None, None
        return x, y


# ==========================
# YOLO 포맷 변환 함수
# ==========================
def yolo_from_boxes(boxes, img_w, img_h):
    """
    boxes: [(cls, x1,y1,x2,y2), ...] in pixel
    -> "cls xc yc w h" (정규화) 줄들을 합친 문자열
    """
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def yolo_from_polygons(polygons, img_w, img_h):
    """
    polygons: [(cls, [(x,y),...]), ...] in pixel
    -> "cls x1 y1 x2 y2 ... xn yn" (정규화) 줄들을 합친 문자열
    """
    lines = []
    for cls_id, pts in polygons:
        coords = []
        for x, y in pts:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))
    return "\n".join(lines)


def yolo_from_polylines(polylines, img_w, img_h):
    """polylines: [(cls, [(x,y),...]), ...] in pixel
    -> "cls x1 y1 x2 y2 ... xn yn" (정규화)

    형식이 폴리곤과 같고, 2점 폴리라인은 bbox 와도 길이가 같아
    **파일만 보고는 구분되지 않는다.** 시작 시 고른 모드로 해석하며,
    차선 라벨은 별도 폴더에 두는 것을 전제로 한다.
    """
    lines = []
    for cls_id, pts in polylines:
        coords = []
        for x, y in pts:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))
    return "\n".join(lines)


class ImageListWidget(QListWidget):
    """숫자·백틱 키를 **목록이 삼키지 않고** 창으로 넘긴다.

    QListWidget 은 인쇄 가능한 문자를 받으면 "그 글자로 시작하는 항목 찾기"
    (keyboard search) 를 한다. 이미지 이름이 `19700101_...` 로 시작하는 탓에
    클래스 변경용 숫자 키를 누르면 그 항목으로 점프해 **이미지가 바뀌어 버렸다.**

    해당 키를 ignore() 하면 Qt 가 부모 위젯(LabelingTool)의 keyPressEvent 로
    넘겨주므로 클래스 변경만 일어난다.
    """

    def keyPressEvent(self, event):
        key = event.key()
        if (Qt.Key.Key_0 <= key <= Qt.Key.Key_9
                or key == Qt.Key.Key_QuoteLeft):
            event.ignore()          # 부모로 전달 -> 클래스 변경
            return
        super().keyPressEvent(event)


# ==========================
#   메인 툴
# ==========================
class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO BBox + Seg Labeling Tool")

        self.image_dir = ""
        self.label_dir = ""
        self.image_files: list[str] = []
        self.current_index = -1
        self.last_enter_time = 0

        # 캐시
        self.label_cache_boxes: dict[str, list] = {}
        self.label_cache_polygons: dict[str, list] = {}
        self.label_cache_polylines: dict[str, list] = {}
        self.modified_images: set[str] = set()
        self.review_completed_images: set[str] = set()

        # 원본 스냅샷 (로드 시점의 상태 저장, 변경 비교용)
        self._original_boxes: dict[str, list] = {}
        self._original_polygons: dict[str, list] = {}
        self._original_polylines: dict[str, list] = {}

        # 라벨 완료된 파일명 (확장자 제외 basename) 세트
        self.labeled_basenames: set[str] = set()

        self.mode = "bbox"

        self.init_ui()
        self.choose_mode_popup()

    # ----------------------------
    #   모드 선택 팝업
    # ----------------------------
    def choose_mode_popup(self):
        box = QMessageBox(self)
        box.setWindowTitle("라벨링 모드 선택")
        box.setText("어떤 라벨링 방식을 사용하시겠습니까?")
        bbox_button = box.addButton("Bounding Box", QMessageBox.ButtonRole.AcceptRole)
        seg_button = box.addButton("Segmentation", QMessageBox.ButtonRole.AcceptRole)
        pl_button = box.addButton("Polyline (차선)", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(bbox_button)
        box.exec()

        clicked = box.clickedButton()
        if clicked == seg_button:
            self.mode = "seg"
        elif clicked == pl_button:
            self.mode = "polyline"
        else:
            self.mode = "bbox"

        self.image_label.mode = self.mode
        self.setWindowTitle(f"YOLO Labeling Tool — {self.mode} 모드")
        self._update_class_label()

    # ----------------------------
    #   UI 구성
    # ----------------------------
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 왼쪽 패널
        left_layout = QVBoxLayout()
        btn_img_dir = QPushButton("이미지 폴더 선택")
        btn_img_dir.clicked.connect(self.select_image_dir)
        btn_lbl_dir = QPushButton("라벨 폴더 선택")
        btn_lbl_dir.clicked.connect(self.select_label_dir)
        btn_labeled_dir = QPushButton("라벨된 폴더 선택")
        btn_labeled_dir.clicked.connect(self.select_labeled_dir)

        self.class_label = QLabel("현재 클래스: 0")
        self.review_checkbox = QCheckBox("검수완료")
        self.review_checkbox.stateChanged.connect(self.on_review_checkbox_changed)

        self.list_widget = ImageListWidget()
        self.list_widget.currentRowChanged.connect(self.on_list_changed)

        self.clear_button = QPushButton("현재 이미지 라벨 삭제")
        self.clear_button.clicked.connect(self.clear_current_labels)

        self.save_button = QPushButton("전체 저장")
        self.save_button.clicked.connect(self.save_all)

        left_layout.addWidget(btn_img_dir)
        left_layout.addWidget(btn_lbl_dir)
        left_layout.addWidget(btn_labeled_dir)
        left_layout.addWidget(self.class_label)
        left_layout.addWidget(self.review_checkbox)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(self.clear_button)
        left_layout.addWidget(self.save_button)

        # 이미지 영역
        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: black;")

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.image_label, 4)

        self.setCentralWidget(main_widget)
        self.resize(1500, 900)

    # ----------------------------
    #   키 입력
    # ----------------------------
    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        # 클래스 변경 (0~9). 백틱(`)은 0 키가 멀어서 두는 0번 클래스 단축키다.
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9 or key == Qt.Key.Key_QuoteLeft:
            cls_id = 0 if key == Qt.Key.Key_QuoteLeft else key - Qt.Key.Key_0
            self.image_label.set_current_class(cls_id)
            self._update_class_label()

        # Ctrl+Z -> Undo
        elif (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Z:
            if self.mode == "bbox":
                self.image_label.undo_bbox()
            elif self.mode == "polyline":
                self.image_label.undo_polyline()
            else:
                self.image_label.undo_seg()

        # Enter -> 세그 폴리곤 닫기
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.mode == "polyline":
                import time
                current_time = time.time()
                if current_time - self.last_enter_time < 0.25:   # 더블 엔터 -> 확정
                    self.image_label.finish_polyline()
                    self.last_enter_time = 0
                    return
                self.last_enter_time = current_time
                cursor_pos = self.image_label.mapFromGlobal(QCursor.pos())
                x, y = self.image_label._map_to_image(cursor_pos)
                if x is not None:
                    self.image_label._polyline_add_point(x, y)
                return
            if self.mode == "seg":
                import time
                current_time = time.time()

                # 더블 엔터 감지 (0.25초 이내 두 번)
                if current_time - self.last_enter_time < 0.25:
                    # 더블 엔터 -> 폴리곤 닫기
                    self.image_label.finish_polygon()
                    self.last_enter_time = 0
                    return
                else:
                    self.last_enter_time = current_time

                # 일반 엔터 -> 현재 마우스 위치에 점 추가
                cursor_pos = self.image_label.mapFromGlobal(QCursor.pos())
                x, y = self.image_label._map_to_image(cursor_pos)

                if x is not None:
                    self.image_label.current_polygon.append((x, y))
                    self.image_label.seg_preview_point = None
                    self.image_label.update()

        # 좌우 화살표 -> 이전/다음 이미지
        elif key == Qt.Key.Key_Right:
            self.next_image()
        elif key == Qt.Key.Key_Left:
            self.prev_image()
        else:
            super().keyPressEvent(event)

    def _update_class_label(self):
        cls_id = self.image_label.current_class
        if self.mode == "polyline":
            name = POLYLINE_CLASS_NAMES.get(cls_id, "미정의")
            self.class_label.setText(f"현재 클래스: {cls_id} ({name})")
        else:
            self.class_label.setText(f"현재 클래스: {cls_id}")

    # ----------------------------
    #   폴더 선택
    # ----------------------------
    def select_image_dir(self):
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if d:
            self.image_dir = d
            self.load_image_list()

    def select_label_dir(self):
        d = QFileDialog.getExistingDirectory(self, "라벨 폴더 선택")
        if d:
            self.label_dir = d
            if self.image_files and self.current_index != -1:
                self.load_current_image(force_reload=True)

    def select_labeled_dir(self):
        d = QFileDialog.getExistingDirectory(self, "라벨된 폴더 선택")
        if d:
            self.labeled_basenames.clear()
            for f in os.listdir(d):
                if f.lower().endswith(".txt"):
                    base = os.path.splitext(f)[0]
                    self.labeled_basenames.add(base)
            self.highlight_labeled_images()

    def highlight_labeled_images(self):
        """리스트 항목 배경색 갱신: 수정됨/검수완료(연하늘) > 라벨완료(연노란) > 기본(투명)"""
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            img_name = item.text()
            base = os.path.splitext(img_name)[0]
            if img_name in self.modified_images or img_name in self.review_completed_images:
                item.setBackground(QColor(200, 230, 255))  # 연하늘색 (수정됨)
            elif base in self.labeled_basenames:
                item.setBackground(QColor(255, 255, 200))  # 연노란색 (라벨 완료)
            else:
                item.setBackground(QColor(0, 0, 0, 0))  # 투명 (기본)

    # ----------------------------
    #   이미지 리스트 로드
    # ----------------------------
    def load_image_list(self):
        if not self.image_dir:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = [
            f for f in sorted(os.listdir(self.image_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]

        self.list_widget.clear()
        self.list_widget.addItems(self.image_files)
        self.highlight_labeled_images()

        self.label_cache_boxes.clear()
        self.label_cache_polygons.clear()
        self.modified_images.clear()
        self.review_completed_images.clear()
        self._original_boxes.clear()
        self._original_polygons.clear()

        if self.image_files:
            self.current_index = 0
            self.list_widget.setCurrentRow(0)
            self.load_current_image(force_reload=True)

    def on_list_changed(self, row: int):
        if row < 0 or row >= len(self.image_files):
            return

        # 현재 이미지 캐시 저장
        self.update_cache_for_current_image()

        self.current_index = row
        self.load_current_image()

    # ----------------------------
    #   현재 이미지/라벨 로드
    # ----------------------------
    def current_image_path(self):
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return None
        return os.path.join(self.image_dir, self.image_files[self.current_index])

    def current_label_path(self):
        if not self.label_dir or self.current_index < 0:
            return None
        img_name = self.image_files[self.current_index]
        base, _ = os.path.splitext(img_name)
        return os.path.join(self.label_dir, base + ".txt")

    def load_current_image(self, force_reload: bool = False):
        img_path = self.current_image_path()
        if not img_path:
            return
        pix = QPixmap(img_path)
        if pix.isNull():
            return

        self.image_label.set_image(pix)

        img_name = self.image_files[self.current_index]
        self._sync_review_checkbox(img_name)

        # 캐시 우선
        if not force_reload and (
            img_name in self.label_cache_boxes
            or img_name in self.label_cache_polygons
            or img_name in self.label_cache_polylines
        ):
            self.image_label.set_boxes(self.label_cache_boxes.get(img_name, []))
            self.image_label.set_polygons(self.label_cache_polygons.get(img_name, []))
            self.image_label.set_polylines(self.label_cache_polylines.get(img_name, []))
            # 원본 스냅샷이 없으면 캐시 값을 원본으로 저장
            if img_name not in self._original_boxes:
                cached_boxes = self.label_cache_boxes.get(img_name, [])
                cached_polys = self.label_cache_polygons.get(img_name, [])
                cached_pls = self.label_cache_polylines.get(img_name, [])
                self._original_boxes[img_name] = list(cached_boxes)
                self._original_polygons[img_name] = [(c, list(p)) for c, p in cached_polys]
                self._original_polylines[img_name] = [(c, list(p)) for c, p in cached_pls]
            return

        # 라벨 파일에서 로드
        boxes = []
        polygons = []
        polylines = []

        lbl_path = self.current_label_path()
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])

                    # 폴리라인 모드에서는 좌표 줄을 전부 폴리라인으로 읽는다.
                    # 2점 폴리라인은 bbox 와, n점 폴리라인은 폴리곤과 형식이
                    # 같아 파일만으로는 구분되지 않기 때문이다.
                    if self.mode == "polyline":
                        coords = list(map(float, parts[1:]))
                        if len(coords) >= 4 and len(coords) % 2 == 0:
                            pts = [(coords[i] * self.image_label.img_w,
                                    coords[i + 1] * self.image_label.img_h)
                                   for i in range(0, len(coords), 2)]
                            polylines.append((cls_id, pts))
                        continue

                    # bbox: cls xc yc w h
                    if len(parts) == 5:
                        xc, yc, w, h = map(float, parts[1:])
                        x1 = (xc - w / 2) * self.image_label.img_w
                        y1 = (yc - h / 2) * self.image_label.img_h
                        x2 = (xc + w / 2) * self.image_label.img_w
                        y2 = (yc + h / 2) * self.image_label.img_h
                        boxes.append((cls_id, x1, y1, x2, y2))
                    # seg: cls x1 y1 x2 y2 ... xn yn
                    elif len(parts) > 5 and len(parts[1:]) % 2 == 0:
                        coords = list(map(float, parts[1:]))
                        pts = []
                        for i in range(0, len(coords), 2):
                            x = coords[i] * self.image_label.img_w
                            y = coords[i + 1] * self.image_label.img_h
                            pts.append((x, y))
                        if len(pts) >= 3:
                            polygons.append((cls_id, pts))

        self.image_label.set_boxes(boxes)
        self.image_label.set_polygons(polygons)
        self.image_label.set_polylines(polylines)

        # 원본 스냅샷 저장 (변경 비교용)
        self._original_boxes[img_name] = list(boxes)
        self._original_polygons[img_name] = [(cls, list(pts)) for cls, pts in polygons]
        self._original_polylines[img_name] = [(cls, list(pts)) for cls, pts in polylines]

        self.image_label.setFocus()  # 이미지가 바뀔 때마다 포커스를 Label로 지정

    def _sync_review_checkbox(self, img_name: str):
        self.review_checkbox.blockSignals(True)
        self.review_checkbox.setChecked(img_name in self.review_completed_images)
        self.review_checkbox.blockSignals(False)

    def on_review_checkbox_changed(self, state: int):
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return

        img_name = self.image_files[self.current_index]
        if state == Qt.CheckState.Checked.value:
            self.review_completed_images.add(img_name)
        else:
            self.review_completed_images.discard(img_name)

        self.highlight_labeled_images()

    # ----------------------------
    #   캐시/체크
    # ----------------------------
    def update_cache_for_current_image(self):
        if self.current_index < 0 or not self.image_files:
            return
        img_name = self.image_files[self.current_index]

        current_boxes = self.image_label.get_boxes()
        current_polygons = self.image_label.get_polygons()
        current_polylines = self.image_label.get_polylines()

        # 캐시에 현재 상태 저장 (항상)
        self.label_cache_boxes[img_name] = current_boxes
        self.label_cache_polygons[img_name] = current_polygons
        self.label_cache_polylines[img_name] = current_polylines

        # 원본과 비교하여 실제 변경이 있을 때만 modified 처리
        orig_boxes = self._original_boxes.get(img_name, [])
        orig_polygons = self._original_polygons.get(img_name, [])
        orig_polylines = self._original_polylines.get(img_name, [])

        if (current_boxes != orig_boxes or current_polygons != orig_polygons
                or current_polylines != orig_polylines):
            self.modified_images.add(img_name)
        else:
            # 되돌렸을 경우 modified에서 제거
            self.modified_images.discard(img_name)

        # 리스트 배경색 갱신
        self.highlight_labeled_images()

    # ----------------------------
    #   라벨 삭제
    # ----------------------------
    def clear_current_labels(self):
        if self.current_index < 0:
            return
        reply = QMessageBox.question(
            self,
            "라벨 삭제 확인",
            "현재 이미지의 모든 라벨을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.image_label.set_boxes([])
        self.image_label.set_polygons([])
        self.image_label.set_polylines([])
        self.update_cache_for_current_image()

    # ----------------------------
    #   저장
    # ----------------------------
    def save_all(self):
        if not self.label_dir:
            QMessageBox.warning(self, "경고", "라벨 폴더를 먼저 선택하세요.")
            return

        # 현재 화면도 캐시에 반영
        self.update_cache_for_current_image()

        save_targets = self.modified_images | self.review_completed_images

        if not save_targets:
            QMessageBox.information(self, "알림", "수정되었거나 검수완료된 이미지가 없습니다.")
            return

        # 저장 방식 선택 팝업
        box = QMessageBox(self)
        box.setWindowTitle("저장 방식 선택")
        box.setText("어떤 방식으로 저장하시겠습니까?")
        box.setInformativeText(
            f"저장 대상 이미지: {len(save_targets)}개\n\n"
            "덮어쓰기: 기존 라벨 폴더에 덮어씁니다.\n"
            "별도 저장: 저장 대상 이미지+라벨을 새 경로에 저장합니다."
        )
        btn_overwrite = box.addButton("덮어쓰기 저장", QMessageBox.ButtonRole.AcceptRole)
        btn_separate = box.addButton("별도 경로 저장", QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = box.addButton("취소", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_overwrite)
        box.exec()

        clicked = box.clickedButton()
        if clicked == btn_cancel:
            return
        elif clicked == btn_separate:
            self._save_to_separate_dir(save_targets)
        else:
            self._save_overwrite(save_targets)

    def _save_overwrite(self, save_targets):
        """기존 라벨 폴더에 덮어쓰기 저장"""
        os.makedirs(self.label_dir, exist_ok=True)

        saved = 0
        for img_name in save_targets:
            if self._save_label_for_image(img_name, self.label_dir):
                saved += 1

        QMessageBox.information(
            self,
            "저장 완료",
            f"{saved}개 이미지에 대해 라벨을 덮어쓰기 저장했습니다.",
        )

    def _save_to_separate_dir(self, save_targets):
        """저장 대상 이미지+라벨을 별도 경로에 저장"""
        target_dir = QFileDialog.getExistingDirectory(self, "저장할 폴더 선택")
        if not target_dir:
            return

        img_out = os.path.join(target_dir, "images")
        lbl_out = os.path.join(target_dir, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        saved = 0
        for img_name in save_targets:
            # 이미지 복사
            src_img = os.path.join(self.image_dir, img_name)
            dst_img = os.path.join(img_out, img_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            # 라벨 저장
            if self._save_label_for_image(img_name, lbl_out):
                saved += 1

        QMessageBox.information(
            self,
            "저장 완료",
            f"{saved}개 수정된 이미지+라벨을 별도 저장했습니다.\n"
            f"이미지: {img_out}\n"
            f"라벨: {lbl_out}",
        )

    def _save_label_for_image(self, img_name, label_dir):
        """개별 이미지에 대한 라벨 파일 저장 (공통 로직)"""
        img_path = os.path.join(self.image_dir, img_name)
        pix = QPixmap(img_path)
        if pix.isNull():
            return False
        img_w = pix.width()
        img_h = pix.height()

        boxes = self.label_cache_boxes.get(img_name, [])
        polygons = self.label_cache_polygons.get(img_name, [])
        polylines = self.label_cache_polylines.get(img_name, [])

        lines = []
        if boxes:
            lines.extend(yolo_from_boxes(boxes, img_w, img_h).splitlines())
        if polygons:
            lines.extend(yolo_from_polygons(polygons, img_w, img_h).splitlines())
        if polylines:
            lines.extend(yolo_from_polylines(polylines, img_w, img_h).splitlines())

        lbl_path = os.path.join(
            label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return True

    # ----------------------------
    #   prev / next
    # ----------------------------
    def next_image(self):
        if not self.image_files:
            return
        if self.current_index < len(self.image_files) - 1:
            self.list_widget.setCurrentRow(self.current_index + 1)

    def prev_image(self):
        if not self.image_files:
            return
        if self.current_index > 0:
            self.list_widget.setCurrentRow(self.current_index - 1)

    # ----------------------------
    #   종료 확인
    # ----------------------------
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "프로그램 종료",
            "정말 종료하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


# ==========================
#   main
# ==========================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 한글 폰트 설정 (시스템에 맞는 폰트 자동 탐색)
    import platform
    system = platform.system()
    font_candidates = []
    if system == "Windows":
        font_candidates = ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]
    elif system == "Darwin":  # macOS
        font_candidates = ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic"]
    else:  # Linux
        font_candidates = ["NanumGothic", "Noto Sans KR", "UnDotum", "Gulim"]

    from PyQt6.QtGui import QFontDatabase
    available = QFontDatabase.families()
    chosen_font = None
    for candidate in font_candidates:
        if candidate in available:
            chosen_font = candidate
            break

    if chosen_font:
        font = QFont(chosen_font, 9)
        app.setFont(font)

    win = LabelingTool()
    win.show()
    sys.exit(app.exec())
