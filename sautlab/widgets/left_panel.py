from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap, QBrush
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..icons import icon
from ..models import PIPELINE_STEPS, VideoInfo, fmt_time
from .common import Card


def video_progress(v: VideoInfo) -> float:
    done = sum(1 for s in PIPELINE_STEPS[:6] if v.steps.get(s, {}).get("state") == "done")
    review = v.progress * 2 if v.steps.get("dual_asr", {}).get("state") == "done" else 0
    return min(1.0, (done + review) / len(PIPELINE_STEPS))


def placeholder_thumb(w: int = 96, h: int = 66) -> QPixmap:
    pm = QPixmap(w, h)
    p = QPainter(pm)
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0, QColor("#3a2b55"))
    g.setColorAt(0.55, QColor("#d98a4a"))
    g.setColorAt(1, QColor("#1a2236"))
    p.fillRect(0, 0, w, h, QBrush(g))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#10172a"))
    p.drawRect(0, int(h * 0.72), w, h)
    p.setBrush(QColor("#1b2742"))
    for x, bw, bh in ((8, 10, 24), (24, 14, 34), (46, 8, 22), (60, 16, 30)):
        p.drawRect(x, int(h * 0.72) - bh, bw, bh)
    p.end()
    return pm


class VideoSourceCard(Card):
    downloadRequested = Signal(str)

    def __init__(self):
        super().__init__("Video Source", "video-outline")
        self.url = QLineEdit()
        self.url.setPlaceholderText("Paste a video link here...")
        self.dataset_label = QLabel("Kaggle Dataset Name")
        self.dataset_label.setObjectName("muted")
        self.dataset = QLineEdit()
        self.dataset.setPlaceholderText("Kaggle dataset slug (e.g. my-arabic-corpus)")
        self.dataset_label.hide()
        self.dataset.hide()
        self.btn = QPushButton("  Download")
        self.btn.setObjectName("primary")
        self.btn.setIcon(icon("download", "#ffffff"))
        self.btn.setMinimumHeight(34)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.msg = QLabel("")
        self.msg.setObjectName("muted")
        self.msg.setWordWrap(True)
        self.msg.hide()
        self.body.addWidget(self.url)
        self.body.addWidget(self.dataset_label)
        self.body.addWidget(self.dataset)
        self.body.addWidget(self.btn)
        self.body.addWidget(self.progress)
        self.body.addWidget(self.msg)
        self.btn.clicked.connect(self._clicked)
        self.url.returnPressed.connect(self._clicked)

    def set_dataset_controls(self, visible: bool, default_name: str = "") -> None:
        self.dataset_label.setVisible(visible)
        self.dataset.setVisible(visible)
        if default_name:
            self.dataset.setText(default_name)

    def _clicked(self) -> None:
        text = self.url.text().strip()
        if not text:
            self.show_message("Paste a video link first.", error=True)
            return
        if not text.lower().startswith(("http://", "https://")):
            self.show_message("The link must start with http:// or https://", error=True)
            return
        self.downloadRequested.emit(text)

    def show_message(self, text: str, error: bool = False) -> None:
        self.msg.setText(text)
        self.msg.setStyleSheet("color:#ff7b7b;" if error else "color:#8b9bbd;")
        self.msg.setVisible(bool(text))

    def set_busy(self, busy: bool, fraction: float = 0.0, text: str = "") -> None:
        self.btn.setEnabled(not busy)
        self.url.setEnabled(not busy)
        self.dataset.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.progress.setValue(int(fraction * 100))
        if busy:
            self.show_message(text)
        elif not text:
            self.msg.hide()


class VideoInfoCard(Card):
    def __init__(self):
        super().__init__("Video Information", "chart-box-outline")
        row = QHBoxLayout()
        row.setSpacing(10)
        self.thumb = QLabel()
        self.thumb.setFixedSize(96, 66)
        self.thumb.setPixmap(placeholder_thumb())
        self.thumb.setStyleSheet("border-radius:6px;")
        row.addWidget(self.thumb, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(5)
        self.title = QLabel("No video selected")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-size:13px; font-weight:600;")
        col.addWidget(self.title)
        self.channel = self._info_row(col, "account-circle-outline")
        self.duration = self._info_row(col, "clock-outline")
        self.license = self._info_row(col, "creative-commons")
        row.addLayout(col, 1)
        self.body.addLayout(row)

    def _info_row(self, parent: QVBoxLayout, icon_name: str) -> QLabel:
        r = QHBoxLayout()
        r.setSpacing(6)
        ic = QLabel()
        ic.setPixmap(icon(icon_name, "#9db4ff").pixmap(15, 15))
        lbl = QLabel("–")
        lbl.setObjectName("muted")
        lbl.setStyleSheet("color:#c3cfeb;")
        r.addWidget(ic)
        r.addWidget(lbl, 1)
        parent.addLayout(r)
        return lbl

    def set_video(self, v: VideoInfo | None) -> None:
        if v is None:
            return
        self.title.setText(v.title)
        self.title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.channel.setText(v.channel)
        self.duration.setText(fmt_time(v.duration))
        self.license.setText(v.license)
        pm = QPixmap(v.thumbnail) if v.thumbnail else QPixmap()
        self.thumb.setPixmap(pm.scaled(96, 66, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation) if not pm.isNull() else placeholder_thumb())


class AudioExtractionCard(Card):
    extractRequested = Signal(int, int)

    def __init__(self):
        super().__init__("Audio Extraction", "cog")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.fmt = QComboBox()
        self.fmt.addItem("WAV")
        self.rate = QComboBox()
        self.rate.addItem("16 kHz", 16000)  # the corpus pipeline is fixed to 16 kHz mono 16-bit
        self.chan = QComboBox()
        self.chan.addItem("Mono, 16-bit", 1)
        for r, (name, w) in enumerate((("Output Format:", self.fmt), ("Sample Rate:", self.rate), ("Channels:", self.chan))):
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(w, r, 1)
        grid.setColumnStretch(1, 1)
        self.body.addLayout(grid)
        row = QHBoxLayout()
        self.ic = QLabel()
        self.ic.setPixmap(icon("check-circle", "#22c55e").pixmap(16, 16))
        self.status = QLabel("Ready to extract audio")
        self.status.setObjectName("ok")
        self.btn = QPushButton("Extract")
        self.btn.setIcon(icon("waveform", "#e8eeff"))
        row.addWidget(self.ic)
        row.addWidget(self.status, 1)
        row.addWidget(self.btn)
        self.body.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.body.addWidget(self.progress)
        self.btn.clicked.connect(lambda: self.extractRequested.emit(self.rate.currentData(), self.chan.currentData()))

    def set_state(self, text: str, ok: bool = True, busy: bool = False, fraction: float = 0.0) -> None:
        self.status.setText(text)
        self.ic.setPixmap(icon("check-circle" if ok else "progress-clock", "#22c55e" if ok else "#fbbf24").pixmap(16, 16))
        self.btn.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.progress.setValue(int(fraction * 100))


class VideoRow(QWidget):
    def __init__(self, index: int, v: VideoInfo):
        super().__init__()
        g = QGridLayout(self)
        g.setContentsMargins(8, 6, 8, 6)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)
        num = QLabel(f"{index}.")
        title = QLabel(v.title)
        title.setStyleSheet("font-weight:600;")
        dur = QLabel(fmt_time(v.duration))
        dur.setStyleSheet("color:#c3cfeb;")
        done = v.steps.get("human_review", {}).get("state") == "done" or v.progress >= 0.999
        st = QLabel()
        st.setPixmap(icon("check-circle" if done else "circle-outline", "#22c55e" if done else "#7084ad").pixmap(18, 18))
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(video_progress(v) * 100))
        bar.setTextVisible(False)
        g.addWidget(num, 0, 0)
        g.addWidget(title, 0, 1)
        g.addWidget(dur, 0, 2)
        g.addWidget(st, 0, 3)
        g.addWidget(bar, 1, 1, 1, 2)
        g.setColumnStretch(1, 1)
        for w in (num, title, dur, st, bar):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)


class ProjectVideosCard(Card):
    videoSelected = Signal(str)

    def __init__(self):
        super().__init__("Project Videos (0)", "video-outline", margins=(8, 10, 8, 8))
        self.title_lbl = self.header.itemAt(1).widget()
        self.list = QListWidget()
        self.list.setSpacing(2)
        self.body.addWidget(self.list, 1)
        self.list.currentRowChanged.connect(self._row_changed)
        self._videos: list[VideoInfo] = []
        self._loading = False

    def set_videos(self, videos: list[VideoInfo], current_id: str | None) -> None:
        self._loading = True
        self._videos = videos
        self.list.clear()
        self.title_lbl.setText(f"Project Videos ({len(videos)})")
        for i, v in enumerate(videos, 1):
            item = QListWidgetItem()
            w = VideoRow(i, v)
            item.setSizeHint(w.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, w)
            if v.id == current_id:
                self.list.setCurrentItem(item)
        self._loading = False

    def _row_changed(self, row: int) -> None:
        if not self._loading and 0 <= row < len(self._videos):
            self.videoSelected.emit(self._videos[row].id)
