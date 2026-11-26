from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from .. import theme
from ..controller import ProjectController
from ..icons import icon
from ..models import PIPELINE_STEPS, STEP_TITLES
from .common import Card

GREEN = QColor("#22c55e")
BLUE = QColor("#3b82f6")
GREY = QColor("#6b7ba0")


class StepsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.steps: dict = {}
        self.setMinimumHeight(56)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._spin)

    def set_steps(self, steps: dict) -> None:
        self.steps = steps
        running = any(s.get("state") == "running" for s in steps.values())
        if running and not self._timer.isActive():
            self._timer.start()
        elif not running:
            self._timer.stop()
        self.update()

    def _spin(self) -> None:
        self._phase = (self._phase + 1) % 8
        self.update()

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = len(PIPELINE_STEPS)
        cell = self.width() / n
        cy = self.height() / 2
        r = 15
        title_font = QFont(self.font())
        title_font.setPixelSize(12)
        title_font.setBold(True)
        sub_font = QFont(self.font())
        sub_font.setPixelSize(11)
        for i, key in enumerate(PIPELINE_STEPS):
            info = self.steps.get(key, {})
            state = info.get("state", "pending")
            cx = i * cell + 8 + r
            col = {"done": GREEN, "running": BLUE}.get(state, GREY)
            # connector to the next step
            if i < n - 1:
                nxt = self.steps.get(PIPELINE_STEPS[i + 1], {}).get("state", "pending")
                text_end = cx + r + 8 + 92
                x0 = max(text_end, cx + r + 70)
                x1 = (i + 1) * cell + 8 - 6
                if x1 - x0 > 12:
                    c = GREEN if state == "done" and nxt == "done" else (BLUE if state == "done" and nxt == "running" else GREY)
                    pen = QPen(c, 1.6)
                    if state != "done":
                        pen.setStyle(Qt.DashLine)
                    p.setPen(pen)
                    p.drawLine(QPointF(x0, cy), QPointF(x1 - 5, cy))
                    if state == "done":
                        a = QPainterPath()
                        a.moveTo(x1 - 6, cy - 4)
                        a.lineTo(x1, cy)
                        a.lineTo(x1 - 6, cy + 4)
                        p.drawPath(a)
            # icon
            if state == "done":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(34, 197, 94, 45))
                p.drawEllipse(QPointF(cx, cy), r, r)
                p.setPen(QPen(GREEN, 1.8))
                p.drawEllipse(QPointF(cx, cy), r - 1, r - 1)
                chk = QPainterPath()
                chk.moveTo(cx - 5, cy)
                chk.lineTo(cx - 1.5, cy + 4)
                chk.lineTo(cx + 5.5, cy - 4)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(GREEN, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.drawPath(chk)
            elif state == "running":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(59, 130, 246, 60))
                p.drawEllipse(QPointF(cx, cy), r + 2, r + 2)
                p.setPen(QPen(BLUE, 2.2))
                p.setBrush(QColor("#0d1b3d"))
                p.drawEllipse(QPointF(cx, cy), r - 1, r - 1)
                tri = QPainterPath()
                tri.moveTo(cx - 3.5, cy - 6)
                tri.lineTo(cx + 6, cy)
                tri.lineTo(cx - 3.5, cy + 6)
                tri.closeSubpath()
                p.setPen(Qt.NoPen)
                p.setBrush(BLUE)
                p.drawPath(tri)
            else:
                pen = QPen(GREY, 1.4)
                pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(cx, cy), r - 1, r - 1)
                p.setPen(Qt.NoPen)
                p.setBrush(GREY)
                p.drawEllipse(QPointF(cx, cy), 2.5, 2.5)
            # text
            tx = cx + r + 8
            p.setFont(title_font)
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(tx, cy - 19, 110, 18), Qt.AlignLeft | Qt.AlignVCenter, STEP_TITLES[key])
            p.setFont(sub_font)
            if state == "done":
                p.setPen(GREEN)
                sub = f"✓ {info.get('info', '')}"
            elif state == "running":
                p.setPen(BLUE)
                sub = "↻ Running..." if self._phase % 2 == 0 else "↺ Running..."
            else:
                p.setPen(GREY)
                sub = "● Pending"
            p.drawText(QRectF(tx, cy + 1, 110, 18), Qt.AlignLeft | Qt.AlignVCenter, sub)


class PipelineBar(Card):
    def __init__(self, controller: ProjectController):
        super().__init__("", None, margins=(14, 8, 14, 8))
        self.c = controller
        row = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(icon("cog", "#9db4ff").pixmap(20, 20))
        t = QLabel("Processing Pipeline")
        t.setObjectName("cardTitle")
        t.setMinimumWidth(130)
        row.addWidget(ic)
        row.addWidget(t)
        self.steps = StepsWidget()
        row.addWidget(self.steps, 1)
        self.body.addLayout(row)
        self.c.videoLoaded.connect(lambda _id: self.refresh())
        self.c.pipelineChanged.connect(self.refresh)
        self.c.videosChanged.connect(self.refresh)

    def refresh(self) -> None:
        v = self.c.video
        self.steps.set_steps(dict(v.steps) if v else {})


class ActivityLog(Card):
    def __init__(self, controller: ProjectController):
        super().__init__("Activity Log", "cog", margins=(12, 8, 8, 8), spacing=4)
        self.c = controller
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        self.rows = QVBoxLayout(host)
        self.rows.setContentsMargins(0, 0, 6, 0)
        self.rows.setSpacing(3)
        self.rows.addStretch(1)
        scroll.setWidget(host)
        self.scroll = scroll
        self.body.addWidget(scroll, 1)
        self.c.logAdded.connect(self.add_entry)
        self.c.videoLoaded.connect(lambda _id: self.clear())
        self._count = 0

    def clear(self) -> None:
        while self.rows.count() > 1:
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._count = 0

    def add_entry(self, e: dict) -> None:
        kind = e.get("kind", "edit")
        icon_name, color = {
            "ok": ("check-circle", "#22c55e"),
            "warn": ("alert-circle", "#f59e0b"),
            "info": ("information-outline", "#3b82f6"),
        }.get(kind, ("information-outline", "#3b82f6"))
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        ic = QLabel()
        ic.setPixmap(icon(icon_name, color).pixmap(16, 16))
        seg = e.get("segment")
        msg = e.get("detail", "")
        if seg:
            body = f'<span style="color:#4da3ff">Segment #{seg}</span>: {msg} by {e.get("user", "")}'
        else:
            body = msg
        lbl = QLabel(f'<span style="color:#4da3ff">{e.get("time", "")}</span> &nbsp; {body}')
        lbl.setTextFormat(Qt.RichText)
        h.addWidget(ic)
        h.addWidget(lbl, 1)
        self.rows.insertWidget(0, w)
        self._count += 1
        if self._count > 250:
            last = self.rows.itemAt(self.rows.count() - 2)
            if last and last.widget():
                last.widget().deleteLater()
                self._count -= 1
        self.scroll.verticalScrollBar().setValue(0)


class StatTile(QWidget):
    def __init__(self, icon_name: str, color: str, title: str):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 0, 6, 0)
        h.setSpacing(10)
        ic = QLabel()
        ic.setPixmap(icon(icon_name, color).pixmap(34, 34))
        col = QVBoxLayout()
        col.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet(f"color:{color}; font-weight:600;")
        self.value = QLabel("0")
        self.value.setStyleSheet("font-size:22px; font-weight:700;")
        col.addWidget(t)
        col.addWidget(self.value)
        h.addWidget(ic)
        h.addLayout(col)
        h.addStretch(1)


class StatsPanel(Card):
    def __init__(self, controller: ProjectController):
        super().__init__("", None, margins=(12, 8, 12, 8))
        self.c = controller
        row = QHBoxLayout()
        row.setSpacing(6)
        self.accepted = StatTile("check-circle", "#22c55e", "Accepted")
        self.review = StatTile("alert-circle", "#f59e0b", "Needs Review")
        self.difficult = StatTile("alert-circle", "#ef4444", "Difficult")
        for w in (self.accepted, self.review, self.difficult):
            row.addWidget(w, 1)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{theme.BORDER_STRONG};")
        row.addWidget(sep)
        auto = QHBoxLayout()
        auto.setSpacing(8)
        self.auto_icon = QLabel()
        self.auto_icon.setPixmap(icon("check-circle", "#22c55e").pixmap(24, 24))
        col = QVBoxLayout()
        col.setSpacing(0)
        self.auto_title = QLabel("Autosave enabled")
        self.auto_title.setStyleSheet("color:#34d399; font-weight:600;")
        self.auto_sub = QLabel("Last saved: -")
        self.auto_sub.setObjectName("muted")
        col.addWidget(self.auto_title)
        col.addWidget(self.auto_sub)
        auto.addWidget(self.auto_icon)
        auto.addLayout(col)
        row.addLayout(auto, 1)
        self.body.addLayout(row)
        for sig in (self.c.structureChanged, self.c.videoLoaded):
            sig.connect(lambda *a: self.refresh())
        self.c.segmentChanged.connect(lambda _id: self.refresh())
        self.c.savedStateChanged.connect(self.refresh_save)
        self.refresh()

    def refresh(self) -> None:
        c = self.c.counts()
        self.accepted.value.setText(str(c["accepted"]))
        self.review.value.setText(str(c["needs_review"]))
        self.difficult.value.setText(str(c["difficult"]))
        self.refresh_save()

    def refresh_save(self) -> None:
        if self.c.dirty:
            self.auto_icon.setPixmap(icon("progress-clock", "#f59e0b").pixmap(24, 24))
            self.auto_title.setText("Unsaved changes")
            self.auto_title.setStyleSheet("color:#fbbf24; font-weight:600;")
        else:
            self.auto_icon.setPixmap(icon("check-circle", "#22c55e").pixmap(24, 24))
            self.auto_title.setText("Autosave enabled")
            self.auto_title.setStyleSheet("color:#34d399; font-weight:600;")
        self.auto_sub.setText(f"Last saved: {self.c.last_saved or '-'}")
