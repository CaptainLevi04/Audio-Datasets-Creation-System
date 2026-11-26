from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from numpy.lib.stride_tricks import sliding_window_view
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

from .. import theme
from ..controller import ProjectController
from ..icons import icon
from ..models import LABEL_TITLES, fmt_time
from ..player import Player
from .common import Card, tool_button

N_FFT = 512
MAX_FREQ = 8000.0
DETAIL_MAX_SPAN = 150.0

SPEC_COLORS = [
    (0.00, (6, 10, 62)),
    (0.30, (24, 52, 190)),
    (0.55, (190, 40, 96)),
    (0.78, (246, 112, 28)),
    (1.00, (255, 228, 64)),
]


def stft_db(x: np.ndarray, hop: int) -> np.ndarray:
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    frames = sliding_window_view(x, N_FFT)[::hop]
    win = np.hanning(N_FFT).astype(np.float32)
    spec = np.fft.rfft(frames * win, axis=1)
    power = spec.real**2 + spec.imag**2
    return (10 * np.log10(power + 1e-10)).astype(np.float32)


class Envelope:
    BLOCK = 256

    def __init__(self, x: np.ndarray, sr: int):
        self.sr = sr
        n = len(x) // self.BLOCK * self.BLOCK
        blocks = x[:n].reshape(-1, self.BLOCK)
        self.mn = blocks.min(axis=1)
        self.mx = blocks.max(axis=1)

    def window(self, t0: float, t1: float, target: int = 1800):
        i0 = max(0, int(t0 * self.sr / self.BLOCK))
        i1 = min(len(self.mn), int(t1 * self.sr / self.BLOCK) + 1)
        n = i1 - i0
        if n <= 0:
            return np.array([]), np.array([])
        stride = max(1, n // target)
        m = (n // stride) * stride
        if m == 0:
            return np.array([]), np.array([])
        mn = self.mn[i0 : i0 + m].reshape(-1, stride).min(axis=1)
        mx = self.mx[i0 : i0 + m].reshape(-1, stride).max(axis=1)
        t = (i0 + (np.arange(len(mn)) + 0.5) * stride) * self.BLOCK / self.sr
        x = np.repeat(t, 2)
        y = np.empty(len(mn) * 2)
        y[0::2] = mn
        y[1::2] = mx
        return x, y


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [fmt_time(v) for v in values]


class RegionOverlay(pg.GraphicsObject):
    """Draws the labelled regions in view coordinates and lets the user drag their borders."""

    def __init__(self, mode: str, ymax: float):
        super().__init__()
        self.mode = mode  # "bar" or "spec"
        self.ymax = ymax
        self.segs: list[tuple] = []
        self.selected: set[int] = set()
        self.duration = 1.0
        self.handles: list[tuple[float, int, str]] = []
        self.drag: tuple[float, int, str] | None = None
        self.on_begin = None
        self.on_drag = None
        self.on_end = None
        self.setZValue(50)
        self._bold = QFont()
        self._bold.setPixelSize(12)
        self._bold.setBold(True)
        self._norm = QFont()
        self._norm.setPixelSize(11)

    def set_data(self, segs, selected, duration):
        self.prepareGeometryChange()
        self.segs = segs
        self.selected = set(selected)
        self.duration = max(duration, 1.0)
        self.handles = []
        prev_end = None
        for sid, a, b, *_ in segs:
            if prev_end is None or abs(prev_end - a) > 1e-3:
                self.handles.append((a, sid, "start"))
            self.handles.append((b, sid, "end"))
            prev_end = b
        self.update()

    def boundingRect(self):
        return QRectF(0, 0, self.duration, self.ymax)

    def paint(self, p, opt, widget=None):
        if not self.segs:
            return
        vb = self.getViewBox()
        if vb is None:
            return
        tr = p.worldTransform()
        top = tr.map(QPointF(0, self.ymax)).y()
        bot = tr.map(QPointF(0, 0)).y()
        x0, x1 = vb.viewRange()[0]

        def px(t: float) -> float:
            return tr.map(QPointF(t, 0)).x()

        p.save()
        p.resetTransform()
        p.setRenderHint(p.RenderHint.Antialiasing)
        for sid, a, b, lab, conf in self.segs:
            if b < x0 or a > x1:
                continue
            xa, xb = px(max(a, x0)), px(min(b, x1))
            w = xb - xa
            if w < 0.5:
                continue
            col = QColor(theme.LABEL_COLORS[lab])
            sel = sid in self.selected
            rect = QRectF(xa, top, w, bot - top)
            if self.mode == "bar":
                fill = QColor(col)
                fill.setAlphaF(0.55 if sel else 0.30)
                p.setBrush(fill)
                pen = QPen(col.lighter(130) if sel else col, 2.4 if sel else 1.3)
                p.setPen(pen)
                p.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
                if w > 54:
                    p.setPen(QColor("#ffffff"))
                    p.setFont(self._bold)
                    fm = p.fontMetrics()
                    title = fm.elidedText(LABEL_TITLES[lab], Qt.ElideRight, int(w - 6))
                    p.drawText(QRectF(xa, top + 5, w, 18), Qt.AlignHCenter | Qt.AlignVCenter, title)
                    p.setFont(self._norm)
                    p.drawText(QRectF(xa, top + 23, w, 16), Qt.AlignHCenter | Qt.AlignVCenter, f"{conf:.2f}")
                elif w > 22:
                    p.setPen(QColor("#ffffff"))
                    p.setFont(self._norm)
                    p.drawText(rect, Qt.AlignCenter, LABEL_TITLES[lab][0])
            else:
                if sel:
                    fill = QColor(col)
                    fill.setAlphaF(0.14)
                    p.setBrush(fill)
                    p.setPen(QPen(col.lighter(125), 2.2))
                    p.drawRect(rect.adjusted(1, 1, -1, -1))
                else:
                    p.setBrush(Qt.NoBrush)
                    line = QColor(col)
                    line.setAlphaF(0.75)
                    p.setPen(QPen(line, 1))
                    p.drawLine(QPointF(px(a), top), QPointF(px(a), bot))
        # handles on the borders
        p.setBrush(QColor("#ffffff"))
        for t, sid, which in self.handles:
            if t < x0 or t > x1:
                continue
            x = px(t)
            if self.mode == "spec":
                # only draw handles for the selected region to keep the plot clean
                seg = next((s for s in self.segs if s[0] == sid), None)
                if seg is None or sid not in self.selected:
                    continue
            lab = next((s[3] for s in self.segs if s[0] == sid), "speech")
            p.setPen(QPen(QColor(theme.LABEL_COLORS[lab]), 1.5))
            p.drawEllipse(QPointF(x, top + 3), 3.8, 3.8)
            p.drawEllipse(QPointF(x, bot - 3), 3.8, 3.8)
        p.restore()

    # dragging
    def _handle_at(self, t: float):
        tol = 7 * self.pixelWidth()
        best, best_d = None, tol
        for h in self.handles:
            d = abs(h[0] - t)
            if d <= best_d:
                best, best_d = h, d
        return best

    def hoverEvent(self, ev):
        if ev.isExit():
            self.unsetCursor()
            return
        if self._handle_at(ev.pos().x()):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.unsetCursor()

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            h = self._handle_at(ev.buttonDownPos().x())
            if h is None:
                ev.ignore()
                return
            self.drag = h
            if self.on_begin:
                self.on_begin(h[1], h[2])
        if self.drag is None:
            ev.ignore()
            return
        ev.accept()
        if self.on_drag:
            self.on_drag(self.drag[1], self.drag[2], ev.pos().x())
        if ev.isFinish():
            self.drag = None
            if self.on_end:
                self.on_end()


class AudioView(Card):
    def __init__(self, controller: ProjectController, player: Player):
        super().__init__("Audio Visualization", "waveform", margins=(12, 10, 12, 8), spacing=6)
        self.c = controller
        self.player = player
        self.sr = 16000
        self.duration = 0.0
        self.audio: np.ndarray | None = None
        self.env: Envelope | None = None
        self._detail_range: tuple[float, float] | None = None
        self._levels = (-90.0, -20.0)
        self._own_selection = False

        pg.setConfigOptions(antialias=False, background=theme.PANEL, foreground=theme.MUTED)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.PANEL)
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        self.glw.ci.layout.setSpacing(0)
        self.glw.setMinimumHeight(260)
        self.body.addWidget(self.glw, 1)
        self.loading_lbl = QLabel("Loading audio...")
        self.loading_lbl.setAlignment(Qt.AlignCenter)
        self.loading_lbl.setObjectName("muted")
        self.loading_lbl.hide()
        self.body.addWidget(self.loading_lbl, 1)

        self.p_wave = self.glw.addPlot(row=0, col=0, axisItems={"top": TimeAxis("top")})
        self.p_bar = self.glw.addPlot(row=1, col=0)
        self.p_spec = self.glw.addPlot(row=2, col=0)
        self.glw.ci.layout.setRowFixedHeight(0, 92)
        self.glw.ci.layout.setRowFixedHeight(1, 46)
        self.glw.ci.layout.setRowStretchFactor(2, 1)
        for pl in (self.p_wave, self.p_bar, self.p_spec):
            pl.setMenuEnabled(False)
            pl.hideButtons()
            pl.vb.setMouseEnabled(x=True, y=False)
            pl.vb.setDefaultPadding(0)
            pl.getAxis("left").setWidth(44)
        self.p_bar.setXLink(self.p_wave)
        self.p_spec.setXLink(self.p_wave)

        # waveform plot
        self.p_wave.showAxis("top")
        self.p_wave.hideAxis("bottom")
        top_ax = self.p_wave.getAxis("top")
        top_ax.setStyle(tickLength=-6)
        top_ax.setHeight(22)
        self._quiet_left(self.p_wave)
        self.p_wave.setYRange(-1, 1, padding=0)
        self.p_wave.vb.setLimits(yMin=-1, yMax=1)
        self.wave_curve = pg.PlotCurveItem(pen=pg.mkPen("#2f80ff", width=1), connect="pairs")
        self.p_wave.addItem(self.wave_curve)
        self.sel_band = pg.LinearRegionItem(
            values=(0, 0), movable=False, brush=pg.mkBrush(255, 255, 255, 26), pen=pg.mkPen(None)
        )
        for ln in self.sel_band.lines:
            ln.setPen(pg.mkPen(None))
        self.sel_band.setZValue(5)
        self.sel_band.hide()
        self.p_wave.addItem(self.sel_band)

        # region bar
        for ax in ("top", "bottom"):
            self.p_bar.hideAxis(ax)
        self._quiet_left(self.p_bar)
        self.p_bar.setYRange(0, 1, padding=0)
        self.p_bar.vb.setLimits(yMin=0, yMax=1)

        # spectrogram
        for ax in ("top", "bottom"):
            self.p_spec.hideAxis(ax)
        self.p_spec.setYRange(0, MAX_FREQ, padding=0)
        self.p_spec.vb.setLimits(yMin=0, yMax=MAX_FREQ)
        self.p_spec.vb.setBackgroundColor("#050a22")
        left = self.p_spec.getAxis("left")
        left.setTicks([[(0, "0"), (2000, "2k"), (4000, "4k"), (6000, "6k"), (8000, "8k")]])
        left.setStyle(tickLength=-4)
        cmap = pg.ColorMap([c[0] for c in SPEC_COLORS], [c[1] for c in SPEC_COLORS])
        self._lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.img_overview = pg.ImageItem(axisOrder="col-major")
        self.img_detail = pg.ImageItem(axisOrder="col-major")
        self.img_overview.setZValue(0)
        self.img_detail.setZValue(1)
        for im in (self.img_overview, self.img_detail):
            im.setLookupTable(self._lut)
            self.p_spec.addItem(im)
        self.img_detail.hide()

        # overlays
        self.bar_overlay = RegionOverlay("bar", 1.0)
        self.spec_overlay = RegionOverlay("spec", MAX_FREQ)
        self.p_bar.addItem(self.bar_overlay)
        self.p_spec.addItem(self.spec_overlay)
        for ov in (self.bar_overlay, self.spec_overlay):
            ov.on_begin = self.c.begin_drag
            ov.on_drag = self.c.drag_boundary
            ov.on_end = self.c.end_drag

        # playhead
        self.heads = []
        for pl in (self.p_wave, self.p_bar, self.p_spec):
            ln = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff", width=1.2, style=Qt.DashLine))
            ln.setZValue(80)
            pl.addItem(ln)
            self.heads.append(ln)
        self.tag = pg.TextItem(
            "00:00:00", color="#ffffff", anchor=(0.5, 0), fill=pg.mkBrush(theme.ACCENT), border=None
        )
        self.tag.setZValue(90)
        self.p_wave.addItem(self.tag, ignoreBounds=True)
        self.tag.setPos(0, 1.0)

        self._build_transport()

        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(50)
        self._view_timer.timeout.connect(self._update_view)
        self.p_wave.vb.sigXRangeChanged.connect(lambda *a: self._view_timer.start())
        self.glw.scene().sigMouseClicked.connect(self._scene_click)

        self.c.structureChanged.connect(self._refresh_regions)
        self.c.segmentChanged.connect(lambda _id: self._refresh_regions())
        self.c.selectionChanged.connect(self._on_selection)
        self.player.positionChanged.connect(self._on_position)
        self.player.playingChanged.connect(self._on_playing)

    @staticmethod
    def _quiet_left(pl) -> None:
        ax = pl.getAxis("left")
        ax.setTicks([[]])
        ax.setStyle(tickLength=0, showValues=False)
        ax.setPen(pg.mkPen(None))

    # transport row
    def _build_transport(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(2)
        self.b_prev = tool_button("skip-previous", "Previous segment", size=22)
        self.b_play = QToolButton()
        self.b_play.setObjectName("round")
        self.b_play.setIcon(icon("play", "#ffffff"))
        self.b_play.setFixedSize(34, 34)
        from PySide6.QtCore import QSize

        self.b_play.setIconSize(QSize(20, 20))
        self.b_play.setToolTip("Play / pause (Space)")
        self.b_next = tool_button("skip-next", "Next segment", size=22)
        self.b_stop = tool_button("stop", "Stop", size=20)
        self.time_lbl = QLabel("00:00:00 / 00:00:00")
        self.time_lbl.setMinimumWidth(112)
        self.b_zout = tool_button("magnify-minus-outline", "Zoom out", size=20)
        self.b_zin = tool_button("magnify-plus-outline", "Zoom in", size=20)
        self.b_fit = tool_button("arrow-collapse-horizontal", "Show whole file", size=20)
        self.speed = QComboBox()
        for s in ("0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"):
            self.speed.addItem(s)
        self.speed.setCurrentText("1.0x")
        self.speed.setFixedWidth(64)
        vol_icon = QLabel()
        vol_icon.setPixmap(icon("volume-high", "#cfd9f5").pixmap(18, 18))
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(80)
        self.vol.setFixedWidth(60)
        self.b_loop = tool_button("repeat", "Loop the playing region", size=18, checkable=True)
        for w in (self.b_prev, self.b_play, self.b_next, self.b_stop):
            row.addWidget(w)
        row.addSpacing(4)
        row.addWidget(self.time_lbl)
        row.addSpacing(2)
        for w in (self.b_zout, self.b_zin, self.b_fit):
            row.addWidget(w)
        row.addWidget(self.speed)
        row.addWidget(vol_icon)
        row.addWidget(self.vol)
        row.addWidget(self.b_loop)
        row.addStretch(1)
        legend = QHBoxLayout()
        legend.setSpacing(5)
        for lab in ("speech", "music", "mixed", "silence"):
            sw = QLabel()
            sw.setFixedSize(11, 11)
            sw.setStyleSheet(f"background:{theme.LABEL_COLORS[lab]}; border-radius:2px;")
            t = QLabel(LABEL_TITLES[lab])
            legend.addWidget(sw)
            legend.addWidget(t)
            legend.addSpacing(6)
        row.addLayout(legend)
        self.body.addLayout(row)

        self.b_play.clicked.connect(self.player.toggle)
        self.b_stop.clicked.connect(self.player.stop)
        self.b_prev.clicked.connect(lambda: self._jump(-1))
        self.b_next.clicked.connect(lambda: self._jump(1))
        self.b_zin.clicked.connect(lambda: self.zoom(0.5))
        self.b_zout.clicked.connect(lambda: self.zoom(2.0))
        self.b_fit.clicked.connect(self.fit_all)
        self.speed.currentTextChanged.connect(lambda s: self.player.set_rate(float(s.rstrip("x"))))
        self.vol.valueChanged.connect(lambda v: self.player.set_volume(v / 100))
        self.b_loop.toggled.connect(self.player.set_loop)

    def clear(self) -> None:
        # one second of silence, used when no video is open
        self.set_audio(np.zeros(self.sr, dtype=np.float32), self.sr)

    def set_loading(self, on: bool) -> None:
        self.glw.setVisible(not on)
        self.loading_lbl.setVisible(on)

    # loading audio
    def set_audio(self, samples: np.ndarray, sr: int) -> None:
        self.audio = samples
        self.sr = sr
        self.duration = len(samples) / sr
        self.env = Envelope(samples, sr)
        hop = max(160, int(len(samples) / 6000))
        db = stft_db(samples, hop)
        self._levels = (float(np.percentile(db, 12)), float(np.percentile(db, 99.8)))
        x = (N_FFT / 2 - hop / 2) / sr
        self.img_overview.setImage(db, levels=self._levels, autoLevels=False)
        self.img_overview.setRect(QRectF(x, 0, db.shape[0] * hop / sr, MAX_FREQ))
        self.img_detail.hide()
        self._detail_range = None
        for pl in (self.p_wave, self.p_bar, self.p_spec):
            pl.vb.setLimits(xMin=0, xMax=self.duration, minXRange=0.5, maxXRange=self.duration)
        self.fit_all()
        self._refresh_regions()
        self._on_position(0.0)

    # view control
    def fit_all(self) -> None:
        self.p_wave.vb.setXRange(0, max(self.duration, 1.0), padding=0)

    def view_range(self) -> tuple[float, float]:
        return tuple(self.p_wave.vb.viewRange()[0])

    def set_view(self, a: float, b: float) -> None:
        self.p_wave.vb.setXRange(a, b, padding=0)

    def zoom(self, factor: float) -> None:
        x0, x1 = self.view_range()
        span = (x1 - x0) * factor
        span = min(max(span, 0.5), self.duration)
        t = self.player.pos
        c = t if x0 <= t <= x1 else (x0 + x1) / 2
        a = min(max(c - span / 2, 0), max(self.duration - span, 0))
        self.set_view(a, a + span)

    def ensure_visible(self, a: float, b: float) -> None:
        x0, x1 = self.view_range()
        mid = (a + b) / 2
        if x0 <= mid <= x1:
            return
        span = max(x1 - x0, (b - a) * 1.4)
        start = min(max(mid - span / 2, 0), max(self.duration - span, 0))
        self.set_view(start, start + span)

    def _update_view(self) -> None:
        if self.env is None or self.audio is None:
            return
        x0, x1 = self.view_range()
        x, y = self.env.window(x0, x1)
        if len(x):
            self.wave_curve.setData(x, y)
        span = x1 - x0
        if span > DETAIL_MAX_SPAN:
            self.img_detail.hide()
            self._detail_range = None
            return
        dr = self._detail_range
        if dr and dr[0] <= x0 and dr[1] >= x1 and (dr[1] - dr[0]) < span * 4:
            return
        pad = span * 0.5
        a = max(0.0, x0 - pad)
        b = min(self.duration, x1 + pad)
        ia, ib = int(a * self.sr), int(b * self.sr)
        hop = 160
        db = stft_db(self.audio[ia:ib], hop)
        self.img_detail.setImage(db, levels=self._levels, autoLevels=False)
        self.img_detail.setRect(QRectF(a + (N_FFT / 2 - hop / 2) / self.sr, 0, db.shape[0] * hop / self.sr, MAX_FREQ))
        self.img_detail.show()
        self._detail_range = (a, b)

    # regions
    def _refresh_regions(self) -> None:
        segs = [(s.id, s.start, s.end, s.label, s.confidence) for s in self.c.segments]
        for ov in (self.bar_overlay, self.spec_overlay):
            ov.set_data(segs, self.c.selected, self.duration)
        self._update_band()

    def _update_band(self) -> None:
        cur = self.c.current
        if cur:
            self.sel_band.setRegion((cur.start, cur.end))
            self.sel_band.show()
        else:
            self.sel_band.hide()

    def _on_selection(self) -> None:
        self._refresh_regions()
        cur = self.c.current
        if cur and not self._own_selection:
            self.ensure_visible(cur.start, cur.end)

    def _jump(self, step: int) -> None:
        self.c.select_relative(step)
        cur = self.c.current
        if cur:
            self.player.clear_range()
            self.player.seek(cur.start)

    # mouse
    def _scene_click(self, ev) -> None:
        if ev.button() != Qt.LeftButton or self.duration <= 0:
            return
        pos = ev.scenePos()
        for pl in (self.p_wave, self.p_bar, self.p_spec):
            if not pl.vb.sceneBoundingRect().contains(pos):
                continue
            t = pl.vb.mapSceneToView(pos).x()
            if not (0 <= t <= self.duration):
                return
            seg = self.c.segment_at(t)
            self._own_selection = True
            if seg:
                if ev.modifiers() & Qt.ControlModifier:
                    ids = list(self.c.selected)
                    if seg.id in ids:
                        ids.remove(seg.id)
                    else:
                        ids.append(seg.id)
                    self.c.set_selection(ids)
                else:
                    self.c.set_selection([seg.id])
            self._own_selection = False
            self.player.clear_range()
            self.player.seek(t)
            if ev.double() and seg:
                self.player.play_range(seg.start, seg.end)
            return

    # playhead
    def _on_position(self, t: float) -> None:
        for ln in self.heads:
            ln.setValue(t)
        self.tag.setPos(t, 1.0)
        self.tag.setText(fmt_time(t))
        self.time_lbl.setText(f"{fmt_time(t)} / {fmt_time(self.duration)}")
        if self.player.playing:
            x0, x1 = self.view_range()
            span = x1 - x0
            if t > x1 - span * 0.03 or t < x0:
                start = min(max(t - span * 0.1, 0), max(self.duration - span, 0))
                self.set_view(start, start + span)

    def _on_playing(self, playing: bool) -> None:
        self.b_play.setIcon(icon("pause" if playing else "play", "#ffffff"))
