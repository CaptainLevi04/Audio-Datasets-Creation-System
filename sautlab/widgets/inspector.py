from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

from .. import theme
from ..controller import ProjectController
from ..icons import icon
from ..models import LABEL_TITLES, LABELS, Segment, fmt_time
from ..player import Player
from ..textutil import compare_transcripts, highlight_html, normalize_ar
from .common import Card, TimeSpinBox, muted_label


def rtl_html(inner: str, size: int = 14) -> str:
    return f'<div dir="rtl" align="right" style="font-size:{size}px; line-height:140%;">{inner}</div>'


class AsrCard(QFrame):
    def __init__(self, name: str, icon_name: str, color: str):
        super().__init__()
        self.setObjectName("asrCard")
        self.setProperty("chosen", False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(icon(icon_name, color).pixmap(16, 16))
        title = QLabel(name)
        title.setStyleSheet("font-weight:600;")
        self.check = QLabel()
        self.check.setPixmap(icon("check-circle", "#3b82f6").pixmap(18, 18))
        head.addWidget(ic)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.check)
        lay.addLayout(head)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        self.text.setLayoutDirection(Qt.RightToLeft)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text.setMinimumHeight(40)
        lay.addWidget(self.text)

    def set_content(self, html: str, chosen: bool) -> None:
        self.text.setText(rtl_html(html) if html else '<span style="color:#6c7ca0">No transcript</span>')
        self.check.setVisible(chosen)
        self.setProperty("chosen", chosen)
        self.style().unpolish(self)
        self.style().polish(self)


class _EditFilter(QObject):
    def __init__(self, on_save, on_cancel):
        super().__init__()
        self.on_save = on_save
        self.on_cancel = on_cancel

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress:
            if ev.key() in (Qt.Key_Return, Qt.Key_Enter) and ev.modifiers() & Qt.ControlModifier:
                self.on_save()
                return True
            if ev.key() == Qt.Key_Escape:
                self.on_cancel()
                return True
        return False


class Inspector(Card):
    reprocessRequested = Signal(int)

    def __init__(self, controller: ProjectController, player: Player):
        super().__init__("Segment Inspector", "information-outline", margins=(12, 10, 6, 10), spacing=8)
        self.c = controller
        self.player = player
        self.context_sec = 1.5
        self._sid: int | None = None
        self._loading = False
        self._editing = False
        self.sub = QLabel("")
        self.sub.setObjectName("muted")
        self.header.addWidget(self.sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content.setObjectName("inspContent")
        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(9)
        scroll.setWidget(self.content)
        scroll.setMinimumWidth(240)
        self.body.addWidget(scroll, 1)

        self.empty = QLabel("Select a segment in the table or on the spectrogram to inspect it.")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setObjectName("muted")
        lay.addWidget(self.empty)

        self.main = QWidget()
        m = QVBoxLayout(self.main)
        m.setContentsMargins(0, 0, 0, 0)
        m.setSpacing(9)
        lay.addWidget(self.main)
        lay.addStretch(1)

        # mini player
        row = QHBoxLayout()
        self.b_play = QToolButton()
        self.b_play.setObjectName("round")
        self.b_play.setFixedSize(30, 30)
        self.b_play.setIconSize(QSize(18, 18))
        self.b_play.setIcon(icon("play", "#ffffff"))
        self.b_play.setStyleSheet("QToolButton#round { border-radius: 15px; }")
        self.pos_lbl = QLabel("00:00:00 / 00:00:00")
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 1000)
        vol = QLabel()
        vol.setPixmap(icon("volume-high", "#cfd9f5").pixmap(18, 18))
        row.addWidget(self.b_play)
        row.addWidget(self.pos_lbl)
        row.addWidget(self.seek, 1)
        row.addWidget(vol)
        m.addLayout(row)

        self.b_listen = QPushButton("  Listen Around Cut")
        self.b_listen.setIcon(icon("headphones", "#e8eeff"))
        menu = QMenu(self.b_listen)
        menu.addAction("Around start cut", lambda: self._listen("start"))
        menu.addAction("Around end cut", lambda: self._listen("end"))
        self.b_listen.setMenu(menu)
        m.addWidget(self.b_listen)

        # fields
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.start_spin = TimeSpinBox()
        self.end_spin = TimeSpinBox()
        self.label_combo = QComboBox()
        for lab in LABELS:
            self.label_combo.addItem(theme.dot_icon(theme.LABEL_COLORS[lab]), LABEL_TITLES[lab], lab)
        self.conf_val = QLabel("0.00")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setTextVisible(False)
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_val)
        conf_row.addWidget(self.conf_bar, 1)
        for r, (name, w) in enumerate(
            (("Start Time:", self.start_spin), ("End Time:", self.end_spin), ("Label:", self.label_combo))
        ):
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(w, r, 1)
        grid.addWidget(QLabel("Confidence:"), 3, 0)
        grid.addLayout(conf_row, 3, 1)
        grid.setColumnStretch(1, 1)
        m.addLayout(grid)

        snr = QHBoxLayout()
        self.snr_lbl = QLabel()
        self.snr_lbl.setTextFormat(Qt.RichText)
        self.snr_lbl.setWordWrap(True)
        snr.addWidget(self.snr_lbl)
        m.addLayout(snr)

        st = QHBoxLayout()
        st.addWidget(QLabel("Status:"))
        self.chip = QLabel("Clean")
        self.chip.setObjectName("chip")
        self.chip.setAlignment(Qt.AlignCenter)
        st.addWidget(self.chip)
        st.addStretch(1)
        self.b_override = QPushButton("Override")
        self.b_override.setIcon(icon("chevron-down", "#e8eeff"))
        self.b_override.setLayoutDirection(Qt.RightToLeft)
        om = QMenu(self.b_override)
        om.addAction("Set Clean", lambda: self._set_quality("clean"))
        om.addAction("Set Noisy", lambda: self._set_quality("noisy"))
        om.addSeparator()
        om.addAction("Back to automatic", lambda: self._set_quality(None))
        self.b_override.setMenu(om)
        st.addWidget(self.b_override)
        m.addLayout(st)

        # transcripts
        self.asr = QWidget()
        a = QVBoxLayout(self.asr)
        a.setContentsMargins(0, 0, 0, 0)
        a.setSpacing(8)
        names = self.c.asr_names
        self.card_whisper = AsrCard(names[0], "creation", "#8b9bff")
        self.card_mms = AsrCard(names[1], "account-voice", "#c084fc")
        a.addWidget(self.card_whisper)
        a.addWidget(self.card_mms)
        cmp_row = QHBoxLayout()
        self.cmp_lbl = QLabel()
        self.cmp_lbl.setTextFormat(Qt.RichText)
        self.cmp_lbl.setWordWrap(True)
        self.b_rerun = QToolButton()
        self.b_rerun.setIcon(icon("refresh", "#9db4ff"))
        self.b_rerun.setToolTip("Re-run SNR and both ASR systems on this segment")
        cmp_row.addWidget(self.cmp_lbl, 1)
        cmp_row.addWidget(self.b_rerun)
        a.addLayout(cmp_row)
        btns = QHBoxLayout()
        self.b_whisper = QPushButton(f"Choose\n{names[0]}")
        self.b_mms = QPushButton(f"Choose\n{names[1]}")
        self.b_edit = QPushButton("Edit Final\nTranscript")
        self.b_edit.setIcon(icon("pencil", "#e8eeff"))
        for b in (self.b_whisper, self.b_mms, self.b_edit):
            b.setMinimumHeight(40)
            btns.addWidget(b)
        a.addLayout(btns)
        self.final_lbl = muted_label("Final transcript")
        a.addWidget(self.final_lbl)
        self.final_edit = QPlainTextEdit()
        self.final_edit.setLayoutDirection(Qt.RightToLeft)
        self.final_edit.setReadOnly(True)
        self.final_edit.setFixedHeight(76)
        self.final_edit.setPlaceholderText("Choose a transcript or edit it by hand")
        a.addWidget(self.final_edit)
        self.edit_row = QWidget()
        er = QHBoxLayout(self.edit_row)
        er.setContentsMargins(0, 0, 0, 0)
        self.b_cancel = QPushButton("Cancel")
        er.addStretch(1)
        er.addWidget(self.b_cancel)
        a.addWidget(self.edit_row)
        self.edit_row.hide()
        self._filter = _EditFilter(self._commit_edit, self._cancel_edit)
        self.final_edit.installEventFilter(self._filter)
        m.addWidget(self.asr)

        self.no_text = QLabel("Music and silence regions have no transcript.")
        self.no_text.setObjectName("muted")
        self.no_text.setWordWrap(True)
        m.addWidget(self.no_text)

        act = QHBoxLayout()
        self.b_accept = QPushButton("Accept")
        self.b_accept.setObjectName("success")
        self.b_accept.setIcon(icon("check-circle-outline", "#34d399"))
        self.b_difficult = QPushButton("Mark as Difficult")
        self.b_difficult.setObjectName("warn")
        self.b_difficult.setIcon(icon("alert", "#1a1204"))
        act.addWidget(self.b_accept)
        act.addWidget(self.b_difficult, 1)
        m.addLayout(act)

        notes = QHBoxLayout()
        notes.addWidget(QLabel("Notes:"))
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("Add annotator notes...")
        notes.addWidget(self.notes, 1)
        m.addLayout(notes)

        self._time_timer = QTimer(self)
        self._time_timer.setSingleShot(True)
        self._time_timer.setInterval(450)
        self._time_timer.timeout.connect(self._commit_times)

        self.b_play.clicked.connect(self._play_clicked)
        self.seek.sliderMoved.connect(self._seek_moved)
        self.start_spin.valueChanged.connect(self._time_edited)
        self.end_spin.valueChanged.connect(self._time_edited)
        self.label_combo.activated.connect(self._label_chosen)
        self.b_whisper.clicked.connect(lambda: self._choose("a"))
        self.b_mms.clicked.connect(lambda: self._choose("b"))
        self.b_edit.clicked.connect(self._toggle_edit)
        self.b_cancel.clicked.connect(self._cancel_edit)
        self.b_accept.clicked.connect(lambda: self._set_status("accepted"))
        self.b_difficult.clicked.connect(lambda: self._set_status("difficult"))
        self.b_rerun.clicked.connect(lambda: self._sid and self.reprocessRequested.emit(self._sid))
        self.notes.editingFinished.connect(self._notes_done)

        self.c.selectionChanged.connect(self._on_selection)
        self.c.segmentChanged.connect(self._on_segment_changed)
        self.c.structureChanged.connect(self._on_selection)
        self.player.positionChanged.connect(self._on_position)
        self.player.playingChanged.connect(self._on_playing)
        self._on_selection()

    # helpers
    @property
    def seg(self) -> Segment | None:
        return self.c.seg(self._sid) if self._sid else None

    def _on_selection(self) -> None:
        if self._editing:
            self._commit_edit()
        cur = self.c.current
        self._sid = cur.id if cur else None
        self._refresh()

    def _on_segment_changed(self, sid: int) -> None:
        if sid == self._sid and not self._editing:
            self._refresh()
        elif sid == self._sid:
            self._refresh(keep_edit=True)

    def _refresh(self, keep_edit: bool = False) -> None:
        s = self.seg
        self.empty.setVisible(s is None)
        self.main.setVisible(s is not None)
        if s is None:
            self.sub.setText("")
            return
        self._loading = True
        n = len(self.c.selected)
        extra = f"  (+{n - 1} selected)" if n > 1 else ""
        self.sub.setText(f"#{self.c.number_of(s.id)} / {len(self.c.segments)}{extra}")
        self.start_spin.setValue(s.start)
        self.end_spin.setValue(s.end)
        self.label_combo.setCurrentIndex(LABELS.index(s.label))
        self.conf_val.setText(f"{s.confidence:.2f}")
        self.conf_bar.setValue(int(s.confidence * 100))
        if s.has_text:
            e = f"{s.snr_energy:.1f} dB" if s.snr_energy is not None else "–"
            w = f"{s.snr_wada:.1f} dB" if s.snr_wada is not None else "–"
            self.snr_lbl.setText(
                f'SNR &nbsp; <span style="color:{theme.MUTED}">Energy SNR:</span> {e} &nbsp;&nbsp; '
                f'<span style="color:{theme.MUTED}">WADA SNR:</span> {w}'
            )
            q = s.quality or "none"
        else:
            self.snr_lbl.setText(f'SNR &nbsp; <span style="color:{theme.MUTED}">not used for {s.label}</span>')
            q = "none"
        self.chip.setText({"clean": "Clean", "noisy": "Noisy"}.get(q, "–"))
        self.chip.setProperty("tone", q)
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)
        self.b_override.setEnabled(s.has_text)
        self.asr.setVisible(s.has_text)
        self.no_text.setVisible(not s.has_text)
        if s.has_text:
            cmp = compare_transcripts(s.asr_a, s.asr_b) if (s.asr_a or s.asr_b) else None
            nf = normalize_ar(s.final_text)
            chosen_w = bool(nf) and nf == normalize_ar(s.asr_a)
            chosen_m = bool(nf) and nf == normalize_ar(s.asr_b)
            warn = "#fbbf24"
            self.card_whisper.set_content(
                highlight_html(s.asr_a, (cmp["tokens_a"], cmp["diff_a"]) if cmp else None, warn), chosen_w
            )
            self.card_mms.set_content(
                highlight_html(s.asr_b, (cmp["tokens_b"], cmp["diff_b"]) if cmp else None, warn), chosen_m
            )
            self.cmp_lbl.setText(self._cmp_text(cmp))
            if not keep_edit:
                self.final_edit.setPlainText(s.final_text)
        self.notes.setText(s.notes)
        self._update_seek()
        self._loading = False

    @staticmethod
    def _cmp_text(cmp) -> str:
        if cmp is None:
            return f'<span style="color:{theme.MUTED}">Comparison: no ASR output yet</span>'
        stats = f"WER: {cmp['wer'] * 100:.1f}%, CER: {cmp['cer'] * 100:.1f}%"
        if cmp["agree"]:
            return f'Comparison: <span style="color:#34d399">&#10004; Agree</span> ({stats})'
        n = cmp["diff_words"]
        if cmp["status"] == "needs_review":
            return f'Comparison: <span style="color:#fbbf24">&#9888; {n} word{"s" if n != 1 else ""} differ</span> ({stats})'
        return f'Comparison: <span style="color:#ff7b7b">&#10006; Very different, {n} words</span> ({stats})'

    # player
    def _play_clicked(self) -> None:
        s = self.seg
        if not s:
            return
        if self.player.playing and self.player.range == (s.start, s.end):
            self.player.pause()
        else:
            self.player.play_range(s.start, s.end)

    def _seek_moved(self, v: int) -> None:
        s = self.seg
        if s:
            self.player.seek(s.start + (s.end - s.start) * v / 1000)

    def _listen(self, which: str) -> None:
        s = self.seg
        if not s:
            return
        cut = s.start if which == "start" else s.end
        self.player.play_range(cut - self.context_sec, cut + self.context_sec)

    def _on_position(self, t: float) -> None:
        s = self.seg
        if s:
            self._update_seek(t)

    def _update_seek(self, t: float | None = None) -> None:
        s = self.seg
        if not s:
            return
        t = self.player.pos if t is None else t
        self.pos_lbl.setText(f"{fmt_time(t)} / {fmt_time(s.end)}")
        if not self.seek.isSliderDown():
            frac = (t - s.start) / max(s.duration, 1e-6)
            self.seek.blockSignals(True)
            self.seek.setValue(int(min(max(frac, 0), 1) * 1000))
            self.seek.blockSignals(False)

    def _on_playing(self, playing: bool) -> None:
        self.b_play.setIcon(icon("pause" if playing else "play", "#ffffff"))

    # edits
    def _time_edited(self) -> None:
        if not self._loading:
            self._time_timer.start()

    def _commit_times(self) -> None:
        s = self.seg
        if s:
            self.c.set_times(s.id, self.start_spin.value(), self.end_spin.value())

    def _label_chosen(self, idx: int) -> None:
        if not self._loading:
            self.c.set_label(list(self.c.selected), self.label_combo.itemData(idx))

    def _set_quality(self, q) -> None:
        self.c.set_quality(list(self.c.selected), q)

    def _set_status(self, status: str) -> None:
        self.c.set_status(list(self.c.selected), status)

    def _choose(self, engine: str) -> None:
        if self._sid:
            self.c.choose_transcript(self._sid, engine)

    def _notes_done(self) -> None:
        if self._sid and not self._loading:
            self.c.set_notes(self._sid, self.notes.text())

    def _toggle_edit(self) -> None:
        if self._editing:
            self._commit_edit()
            return
        s = self.seg
        if not s:
            return
        self._editing = True
        if not s.final_text.strip():
            self.final_edit.setPlainText(s.asr_a)
        self.final_edit.setReadOnly(False)
        self.final_edit.setFocus()
        self.b_edit.setText("Save\nTranscript")
        self.b_edit.setIcon(icon("check", "#e8eeff"))
        self.edit_row.show()

    def _end_edit_ui(self) -> None:
        self._editing = False
        self.final_edit.setReadOnly(True)
        self.b_edit.setText("Edit Final\nTranscript")
        self.b_edit.setIcon(icon("pencil", "#e8eeff"))
        self.edit_row.hide()

    def _commit_edit(self) -> None:
        if not self._editing:
            return
        text = self.final_edit.toPlainText().strip()
        sid = self._sid
        self._end_edit_ui()
        if sid:
            self.c.set_final_text(sid, text)
        self._refresh()

    def _cancel_edit(self) -> None:
        if not self._editing:
            return
        self._end_edit_ui()
        s = self.seg
        self.final_edit.setPlainText(s.final_text if s else "")
