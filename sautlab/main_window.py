from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import export, theme
from .backend.base import Backend
from .controller import ProjectController
from .icons import icon
from .models import LABEL_TITLES, LABELS, fmt_time
from .player import Player
from .widgets.audio_view import AudioView
from .widgets.bottom import ActivityLog, PipelineBar, StatsPanel
from .widgets.inspector import Inspector
from .widgets.left_panel import (
    AudioExtractionCard,
    ProjectVideosCard,
    VideoInfoCard,
    VideoSourceCard,
)
from .widgets.segment_table import SegmentTable


class Worker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self.fn(lambda f, m="": self.progress.emit(f, m)))
        except Exception as e:  # shown to the user, not swallowed
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(
        self,
        backend: Backend,
        user: str = "Mohamed Ahmed",
        backend_factory=None,
        watch_downloads: list[str | Path] | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.backend_factory = backend_factory  # callable(data_dir) -> Backend, enables "Open data folder"
        self._watch_dirs = [Path(p) for p in watch_downloads] if watch_downloads else []
        self._auto_importing = False
        self._download_timer: QTimer | None = None
        self.c = ProjectController(backend, user)
        self.player = Player()
        self._workers: set[Worker] = set()
        self._pipeline_busy = False
        self.setWindowTitle("SautLab - Arabic Speech Corpus Annotation")
        self.resize(1600, 1000)
        self.setMinimumSize(1280, 760)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        # panels
        self.source = VideoSourceCard()
        if hasattr(backend, "set_dataset_name"):
            self.source.set_dataset_controls(True, getattr(backend, "dataset_name", "sautlab-corpus"))
        self.info = VideoInfoCard()
        self.extract = AudioExtractionCard()
        self.videos = ProjectVideosCard()
        self.audio = AudioView(self.c, self.player)
        self.table = SegmentTable(self.c)
        self.inspector = Inspector(self.c, self.player)
        self.pipeline = PipelineBar(self.c)
        self.log = ActivityLog(self.c)
        self.stats = StatsPanel(self.c)

        outer.addWidget(self._build_topbar())

        left = QWidget()
        left.setFixedWidth(306)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)
        for w in (self.source, self.info, self.extract):
            lv.addWidget(w)
        lv.addWidget(self.videos, 1)

        centre = QSplitter(Qt.Vertical)
        centre.setChildrenCollapsible(False)
        centre.addWidget(self.audio)
        centre.addWidget(self.table)
        centre.setStretchFactor(0, 11)
        centre.setStretchFactor(1, 9)
        centre.setSizes([520, 380])

        self.inspector.setFixedWidth(344)
        mid = QHBoxLayout()
        mid.setSpacing(8)
        mid.addWidget(left)
        mid.addWidget(centre, 1)
        mid.addWidget(self.inspector)
        outer.addLayout(mid, 1)

        self.pipeline.setFixedHeight(76)
        outer.addWidget(self.pipeline)
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.log.setFixedHeight(104)
        self.stats.setFixedHeight(104)
        bottom.addWidget(self.log, 3)
        bottom.addWidget(self.stats, 2)
        outer.addLayout(bottom)

        self._build_actions()
        self._wire()
        QApplication.instance().aboutToQuit.connect(self._stop_workers)
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.timeout.connect(self._startup)
        self._startup_timer.start(0)

    # top bar
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("card")
        bar.setFixedHeight(56)
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 4, 14, 4)
        h.setSpacing(4)
        logo = QLabel()
        logo.setPixmap(theme.logo_pixmap(34))
        name = QLabel("SautLab")
        name.setStyleSheet("font-size:22px; font-weight:700;")
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{theme.BORDER_STRONG};")
        proj = QLabel("Build a Labelled Arabic Speech Corpus")
        proj.setStyleSheet("font-size:13px;")
        h.addWidget(logo)
        h.addWidget(name)
        h.addSpacing(8)
        h.addWidget(sep)
        h.addSpacing(8)
        h.addWidget(proj)
        h.addStretch(1)
        self.saved_icon = QLabel()
        self.saved_lbl = QLabel("")
        h.addWidget(self.saved_icon)
        h.addWidget(self.saved_lbl)
        h.addStretch(1)
        self.menu_buttons: dict[str, QToolButton] = {}
        for title in ("Project", "Video", "Processing", "Annotation", "Export", "Settings"):
            b = QToolButton()
            b.setObjectName("menuBtn")
            b.setText(title)
            b.setIcon(icon("chevron-down", "#9db4ff"))
            b.setLayoutDirection(Qt.RightToLeft)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setMenu(QMenu(b))
            self.menu_buttons[title] = b
            h.addWidget(b)
        help_btn = QToolButton()
        help_btn.setObjectName("menuBtn")
        help_btn.setText("Help")
        help_btn.clicked.connect(self._show_help)
        h.addWidget(help_btn)
        h.addSpacing(12)
        self.b_undo = self._stacked_button("undo", "Undo")
        self.b_redo = self._stacked_button("redo", "Redo")
        h.addWidget(self.b_undo)
        h.addWidget(self.b_redo)
        h.addSpacing(8)
        self.b_run = QPushButton("  Run Pipeline")
        self.b_run.setObjectName("primary")
        self.b_run.setIcon(icon("play", "#ffffff"))
        self.b_run.setMinimumHeight(34)
        self.b_export = QPushButton("  Export CSV")
        self.b_export.setObjectName("primary")
        self.b_export.setIcon(icon("download", "#ffffff"))
        self.b_export.setMinimumHeight(34)
        h.addWidget(self.b_run)
        h.addWidget(self.b_export)
        return bar

    @staticmethod
    def _stacked_button(icon_name: str, text: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(icon(icon_name, "#e8eeff"))
        b.setText(text)
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        b.setStyleSheet("QToolButton { font-size:10px; padding:2px 8px; } QToolButton:disabled { color:#4d5c80; }")
        return b

    # actions and menus
    def _act(self, text, slot, shortcut=None, icon_name=None, checkable=False) -> QAction:
        a = QAction(text, self)
        if icon_name:
            a.setIcon(icon(icon_name, "#e8eeff"))
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.WindowShortcut)
        a.setCheckable(checkable)
        if checkable:
            a.toggled.connect(slot)
        else:
            a.triggered.connect(lambda _=False: slot())
        self.addAction(a)
        return a

    def _build_actions(self) -> None:
        m = self.menu_buttons
        self.a_save = self._act("Save", self.c.save, "Ctrl+S", "content-save-outline")
        a_quit = self._act("Quit", self.close, "Ctrl+Q")
        if self.backend_factory is not None:
            m["Project"].menu().addAction(self._act("Open data folder...", self._open_data_folder, "Ctrl+O", "folder-open-outline"))
        if hasattr(self.backend, "import_kaggle_zip"):
            m["Project"].menu().addAction(self._act("Import Kaggle output zip...", self._import_kaggle_zip, None, "folder-open-outline"))
        m["Project"].menu().addActions([self.a_save])
        m["Project"].menu().addSeparator()
        m["Project"].menu().addAction(a_quit)

        a_dl = self._act("Download new video...", lambda: self.source.url.setFocus(), None, "download")
        a_ex = self._act("Extract audio again", lambda: self._extract_current(), None, "waveform")
        a_loc = self._act("Add local video/audio file...", self._add_local_file, None, "video-outline")
        m["Video"].menu().addActions([a_dl, a_loc, a_ex])

        self.a_run = self._act("Run pipeline on this video", self._run_pipeline, "Ctrl+R", "play")
        m["Processing"].menu().addAction(self.a_run)
        if hasattr(self.backend, "pull_remote_videos"):
            m["Processing"].menu().addAction(self._act("Pull finished videos from Kaggle...", self._pull_from_kaggle, None, "download"))
        if hasattr(self.backend, "shutdown_session"):
            m["Processing"].menu().addAction(self._act("Stop Kaggle GPU Session", self._stop_kaggle_session))
        self.a_rerun = self._act("Recompute SNR for selected segment", self._reprocess_selected, None, "refresh")
        a_asr = self._act("Import ASR results (Kaggle segments.csv)...", self._import_asr, None, "file-export-outline")
        m["Processing"].menu().addActions([self.a_rerun, a_asr])

        self.a_undo = self._act("Undo", self.c.undo, "Ctrl+Z", "undo")
        self.a_redo = self._act("Redo", self.c.redo, "Ctrl+Y", "redo")
        self.a_add = self._act("Add segment at playhead", self._add, "Ctrl+N", "plus")
        self.a_split = self._act("Split at playhead", self._split, "S", "content-cut")
        self.a_merge = self._act("Merge selected", self._merge, "M", "call-merge")
        self.a_delete = self._act("Delete selected", self._delete, "Delete", "delete-outline")
        self.a_next = self._act("Next conflict", lambda: self.c.select_next_conflict(1), "N", "chevron-down")
        self.a_prev = self._act("Previous conflict", lambda: self.c.select_next_conflict(-1), "Shift+N", "chevron-up")
        self.a_accept = self._act("Accept segment", lambda: self.c.set_status(list(self.c.selected), "accepted"), "A")
        self.a_diff = self._act("Mark as difficult", lambda: self.c.set_status(list(self.c.selected), "difficult"), "D")
        am = m["Annotation"].menu()
        am.addActions([self.a_undo, self.a_redo])
        am.addSeparator()
        am.addActions([self.a_add, self.a_split, self.a_merge, self.a_delete])
        am.addSeparator()
        am.addActions([self.a_next, self.a_prev, self.a_accept, self.a_diff])
        am.addSeparator()
        lm = am.addMenu("Set label")
        for i, lab in enumerate(LABELS, 1):
            a = self._act(LABEL_TITLES[lab], lambda l=lab: self.c.set_label(list(self.c.selected), l), str(i))
            a.setIcon(theme.dot_icon(theme.LABEL_COLORS[lab]))
            lm.addAction(a)
        self.a_conflicts = self._act("Show conflicts only", self._toggle_conflicts, "C", checkable=True)
        am.addSeparator()
        am.addAction(self.a_conflicts)

        a_csv = self._act("Export CSV for this video...", self._export_csv, "Ctrl+E", "file-export-outline")
        a_all = self._act("Export CSV for all videos...", self._export_all_csv)
        a_aud = self._act("Export segment audio + transcripts...", self._export_audio)
        m["Export"].menu().addActions([a_csv, a_all, a_aud])

        sm = m["Settings"].menu()
        sm.addAction(self._act("Annotator name...", self._set_user))
        sm.addAction(self._act("Clean/Noisy SNR threshold...", self._set_threshold))
        if hasattr(self.backend, "classifier"):
            cm = sm.addMenu("Region classifier")
            cg = QActionGroup(self)
            for key, text in (("auto", "Auto (AST if torch is installed)"), ("ast", "AST neural model"), ("heuristic", "Heuristic (fast, less accurate)")):
                a = QAction(text, self, checkable=True)
                a.setChecked(self.backend.classifier == key)
                a.triggered.connect(lambda _=False, k=key: setattr(self.backend, "classifier", k))
                cg.addAction(a)
                cm.addAction(a)
        if hasattr(self.backend, "cookies_file"):
            sm.addAction(self._act("YouTube cookies.txt...", self._set_cookies))
        if hasattr(self.backend, "remote_url"):
            sm.addAction(self._act("Kaggle Remote Server URL...", self._set_kaggle_url))
            sm.addAction(self._act("Connect to Kaggle Auto-Beacon", self._connect_auto_beacon))
            a_fb = self._act(
                "Allow local CPU fallback if Kaggle offline",
                lambda on: setattr(self.backend, "allow_local_fallback", bool(on)),
                checkable=True,
            )
            a_fb.setChecked(bool(getattr(self.backend, "allow_local_fallback", True)))
            sm.addAction(a_fb)
        ctx = sm.addMenu("Listen-around-cut context")
        grp = QActionGroup(self)
        for sec in (1.0, 1.5, 2.0, 3.0):
            a = QAction(f"{sec:g} s each side", self, checkable=True)
            a.setChecked(sec == 1.5)
            a.triggered.connect(lambda _=False, s=sec: setattr(self.inspector, "context_sec", s))
            grp.addAction(a)
            ctx.addAction(a)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.player.toggle)

    def _wire(self) -> None:
        self.b_undo.clicked.connect(self.c.undo)
        self.b_redo.clicked.connect(self.c.redo)
        self.b_run.clicked.connect(self._run_pipeline)
        self.b_export.clicked.connect(self._export_csv)
        self.source.downloadRequested.connect(self._download)
        self.extract.extractRequested.connect(self._extract_clicked)
        self.videos.videoSelected.connect(self._select_video)
        self.table.addClicked.connect(self._add)
        self.table.splitClicked.connect(self._split)
        self.table.mergeClicked.connect(self._merge)
        self.table.deleteClicked.connect(self._delete)
        self.table.playSegment.connect(self._play_segment)
        self.table.reprocessRequested.connect(self._reprocess)
        self.inspector.reprocessRequested.connect(self._reprocess)
        self.c.videosChanged.connect(self._refresh_videos)
        self.c.undoStateChanged.connect(self._undo_state)
        self.c.savedStateChanged.connect(self._saved_state)
        self.c.logAdded.connect(lambda e: None)
        self._undo_state()
        self._saved_state()

    # state display
    def _undo_state(self) -> None:
        self.b_undo.setEnabled(self.c.can_undo())
        self.b_redo.setEnabled(self.c.can_redo())
        self.a_undo.setEnabled(self.c.can_undo())
        self.a_redo.setEnabled(self.c.can_redo())

    def _saved_state(self) -> None:
        if self.c.dirty:
            self.saved_icon.setPixmap(icon("progress-clock", "#f59e0b").pixmap(18, 18))
            self.saved_lbl.setText("Unsaved changes")
            self.saved_lbl.setStyleSheet("color:#fbbf24;")
        else:
            self.saved_icon.setPixmap(icon("check-circle", "#22c55e").pixmap(18, 18))
            when = f" • {self.c.last_saved}" if self.c.last_saved else ""
            self.saved_lbl.setText(f"Project saved{when}")
            self.saved_lbl.setStyleSheet("color:#c3cfeb;")

    def _refresh_videos(self) -> None:
        self.videos.set_videos(self.c.videos, self.c.video.id if self.c.video else None)

    # startup and video loading
    def _startup(self) -> None:
        self._refresh_videos()
        if self.c.videos:
            self._select_video(self.c.videos[0].id)
        if getattr(self.backend, "is_remote", False):
            self.c.log_info(f"Connected to Kaggle GPU server: {self.backend.remote_url}", kind="ok")
        elif getattr(self.backend, "auto_beacon", False):
            self.c.log_info("Kaggle Auto-Beacon active (will bind automatically once Kaggle Run All starts).", kind="info")
        if self._watch_dirs and hasattr(self.backend, "import_kaggle_zip"):
            self._download_timer = QTimer(self)
            self._download_timer.setInterval(3000)
            self._download_timer.timeout.connect(self._poll_downloads)
            self._download_timer.start()
            self.c.log_info(
                f"Auto-watch active on: {', '.join(str(p) for p in self._watch_dirs)}",
                kind="info",
            )

    def _select_video(self, vid: str) -> None:
        if self.c.video and self.c.video.id == vid:
            return
        self.player.pause()
        self.c.load_video(vid)
        v = self.c.video
        self.info.set_video(v)
        self.setWindowTitle(f"SautLab - {v.title}")
        self._refresh_videos()
        self.audio.set_loading(True)
        self.extract.set_state("Loading audio...", ok=False, busy=True)
        self.c.log_info(f"Loaded {v.title}: {len(self.c.segments)} segments", kind="ok")
        w = Worker(lambda prog: (self.backend.load_audio(vid), self.backend.audio_path(vid)))
        w.done.connect(lambda res, v=v: self._audio_ready(v.id, res))
        w.failed.connect(lambda msg: self._error("Could not load the audio", msg))
        self._run_worker(w)

    def _audio_ready(self, vid: str, res) -> None:
        if not self.c.video or self.c.video.id != vid:
            return
        (samples, sr), path = res
        self.audio.set_loading(False)
        self.audio.set_audio(samples, sr)
        self.player.load(path, len(samples) / sr)
        self.extract.set_state("Audio ready")
        if self.c.segments:
            self.c.set_selection([self.c.segments[0].id])

    def _run_worker(self, w: Worker) -> Worker:
        self._workers.add(w)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        return w

    def _error(self, title: str, msg: str) -> None:
        self.c.log_info(f"{title}: {msg}", kind="warn")
        QMessageBox.warning(self, title, msg)
        self.audio.set_loading(False)
        self.extract.set_state("Ready to extract audio")
        self.source.set_busy(False)

    # download and extraction
    def _download(self, url: str) -> None:
        if hasattr(self.backend, "set_dataset_name"):
            slug = self.backend.set_dataset_name(self.source.dataset.text())
            self.source.dataset.setText(slug)
            self.c.log_info(f"Submitting Kaggle CLI run -> dataset: {slug}", kind="info")
        self._last_dl_stage = ""
        self.source.set_busy(True, 0.0, "Starting download...")
        w = Worker(lambda prog: self.backend.download(url, prog))
        w.progress.connect(self._download_progress)
        w.done.connect(self._downloaded)
        w.failed.connect(lambda msg: self._error("Download failed", msg))
        self._run_worker(w)

    def _download_progress(self, f: float, m: str) -> None:
        self.source.set_busy(True, f, m)
        stage = m.split("(")[0].strip() if m else ""
        if stage and stage != getattr(self, "_last_dl_stage", ""):
            self._last_dl_stage = stage
            self.c.log_info(m, kind="info")

    def _downloaded(self, v) -> None:
        self.source.set_busy(False, text="Download complete")
        self.source.url.clear()
        self.c.refresh_videos()
        self._refresh_videos()
        self.c.log_info(f"Ready: {v.title}", kind="ok")
        self._extract(v.id, 16000, 1, then_select=True)

    def _stop_kaggle_session(self) -> None:
        if not hasattr(self.backend, "shutdown_session"):
            return
        self.c.log_info("Sending shutdown signal to Kaggle session...", kind="info")
        w = Worker(lambda _prog: self.backend.shutdown_session())
        w.done.connect(lambda _r: self.c.log_info("Kaggle shutdown signal sent.", kind="ok"))
        w.failed.connect(self._worker_failed("Could not stop Kaggle session"))
        self._run_worker(w)

    def _add_local_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add a video or audio file", "", "Media files (*.mp4 *.mkv *.webm *.mov *.avi *.mp3 *.wav *.m4a *.flac *.ogg);;All files (*)"
        )
        if not path:
            return
        self.source.set_busy(True, 0.0, "Converting to 16 kHz mono WAV...")
        w = Worker(lambda prog: self.backend.add_local_file(path, prog))
        w.progress.connect(lambda f, m: self.source.set_busy(True, f, m))
        w.done.connect(self._downloaded)
        w.failed.connect(self._worker_failed("Could not add the file"))
        self._run_worker(w)

    def _worker_failed(self, title: str):
        def handler(msg: str) -> None:
            self._error(title, msg)

        return handler

    def _import_asr(self) -> None:
        if not self.c.video:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "ASR result from the Kaggle runner (segments.csv)", "", "CSV files (*.csv)"
        )
        if not path:
            return
        if self.c.dirty:
            self.c.save()
        vid = self.c.video.id
        self.c.log_info("Importing ASR results...", kind="info")
        w = Worker(lambda prog: self.backend.import_asr(vid, path))
        w.done.connect(lambda segs: self._asr_imported(vid, len(segs)))
        w.failed.connect(self._worker_failed("Could not import the ASR results"))
        self._run_worker(w)

    def _asr_imported(self, vid: str, n: int) -> None:
        self.c.refresh_videos()
        self.c.load_video(vid)
        self.c.log_info(f"Imported ASR results: {n} segments", kind="ok")
        conflicts = self.c.conflict_ids()
        if conflicts:
            self.c.set_selection([conflicts[0]])
        elif self.c.segments:
            self.c.set_selection([self.c.segments[0].id])

    def _open_data_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Corpus data folder (contains one folder per video)")
        if not folder or self.backend_factory is None:
            return
        try:
            backend = self.backend_factory(folder)
            self.player.pause()
            self.c.switch_backend(backend)
        except Exception as e:
            self._error("Could not open the folder", f"{type(e).__name__}: {e}")
            return
        self.backend = backend
        self.audio.clear()
        self._refresh_videos()
        if self.c.videos:
            self._select_video(self.c.videos[0].id)
        else:
            self.c.log_info(f"No videos found in {folder}. Use Download or Add local file.", kind="warn")

    def _set_cookies(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "YouTube cookies.txt", "", "Text files (*.txt);;All files (*)")
        if path:
            self.backend.cookies_file = path
            self.c.log_info("YouTube cookies file set.")

    def _set_kaggle_url(self) -> None:
        if not hasattr(self.backend, "remote_url"):
            return
        cur = getattr(self.backend, "remote_url", None) or ""
        url, ok = QInputDialog.getText(
            self,
            "Kaggle Remote Server URL",
            "Public URL printed by main-runner.ipynb (leave blank for local execution):",
            text=cur,
        )
        if not ok:
            return
        cleaned = url.strip().rstrip("/") or None
        self.backend.remote_url = cleaned
        if not cleaned:
            self.c.log_info("Switched to local backend execution.", kind="info")
            return
        self.c.log_info(f"Checking Kaggle server at {cleaned} ...", kind="info")
        w = Worker(lambda _prog: self.backend.check_health())
        w.done.connect(
            lambda info: self.c.log_info(
                f"Connected to Kaggle GPU server: {info.get('device', 'unknown')} | "
                f"cookies={'yes' if info.get('has_cookies') else 'no'} | videos={info.get('videos_count', 0)}",
                kind="ok",
            )
        )
        w.failed.connect(self._worker_failed("Could not reach Kaggle Remote Server"))
        self._run_worker(w)

    def _connect_auto_beacon(self) -> None:
        if not hasattr(self.backend, "connect_auto_beacon"):
            return
        self.c.log_info("Checking private Kaggle Auto-Beacon...", kind="info")

        def _do(_prog):
            health = self.backend.connect_auto_beacon()
            if not health:
                raise RuntimeError(
                    "No live Kaggle Auto-Beacon found. Make sure Phase 4 (`START_REMOTE_SERVER = True`) is running "
                    "on Kaggle and `KAGGLE_API_TOKEN` is set in Kaggle Add-ons -> Secrets."
                )
            return health

        w = Worker(_do)
        w.done.connect(
            lambda info: self.c.log_info(
                f"Auto-connected to Kaggle GPU ({self.backend.remote_url}): {info.get('device', 'unknown')} | "
                f"cookies={'yes' if info.get('has_cookies') else 'no'}",
                kind="ok",
            )
        )
        w.failed.connect(self._worker_failed("Kaggle Auto-Beacon unavailable"))
        self._run_worker(w)

    def _import_kaggle_zip(self) -> None:
        if not hasattr(self.backend, "import_kaggle_zip"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Kaggle corpus archive (corpus_final_output.zip)", "", "ZIP archives (*.zip)"
        )
        if not path:
            return
        self._start_zip_import(path, auto=False)

    def _poll_downloads(self) -> None:
        if self._auto_importing or self._pipeline_busy or not hasattr(self.backend, "import_kaggle_zip"):
            return
        data_root = getattr(self.backend, "root", None)
        if data_root is None:
            return
        from .kaggle_bridge import find_latest_kaggle_zip, should_auto_import

        cand = find_latest_kaggle_zip(search_dirs=self._watch_dirs)
        if cand is not None and should_auto_import(cand, data_root):
            self._start_zip_import(cand, auto=True)

    def _start_zip_import(self, path: str | Path, auto: bool = False) -> None:
        if self._auto_importing:
            return
        zp = Path(path)
        if self.c.dirty:
            self.c.save()
        self._auto_importing = True
        prefix = "Auto-detected Kaggle archive in Downloads" if auto else "Importing & hydrating"
        self.c.log_info(f"{prefix}: {zp.name} ...", kind="info")

        def _do(_prog):
            vids = self.backend.import_kaggle_zip(zp)
            data_root = getattr(self.backend, "root", None)
            if data_root is not None:
                from .kaggle_bridge import record_auto_import

                record_auto_import(zp, data_root, vids)
            return vids

        w = Worker(_do)
        w.done.connect(lambda vids: self._zip_import_finished(vids, zp.name, auto))
        w.failed.connect(lambda msg: self._zip_import_failed(msg, auto))
        self._run_worker(w)

    def _zip_import_finished(self, vids: list[str], name: str, auto: bool) -> None:
        self._auto_importing = False
        src = f"Downloads ({name})" if auto else f"Kaggle archive ({name})"
        self._kaggle_imported(vids, source=src)

    def _zip_import_failed(self, msg: str, auto: bool) -> None:
        self._auto_importing = False
        if auto:
            self.c.log_info(f"Auto-import waiting ({msg})", kind="warn")
        else:
            self._error("Could not import Kaggle archive", msg)

    def _pull_from_kaggle(self) -> None:
        if not hasattr(self.backend, "pull_remote_videos"):
            return
        if not getattr(self.backend, "is_remote", False):
            self._set_kaggle_url()
            if not getattr(self.backend, "is_remote", False):
                return
        if self.c.dirty:
            self.c.save()
        self.c.log_info("Pulling compact FLAC + CSV bundles from Kaggle...", kind="info")
        w = Worker(lambda prog: self.backend.pull_remote_videos(progress=prog))
        w.progress.connect(lambda _f, m: self.c.log_info(m, kind="info") if m else None)
        w.done.connect(lambda vids: self._kaggle_imported(vids, source="Kaggle server"))
        w.failed.connect(self._worker_failed("Could not pull videos from Kaggle"))
        self._run_worker(w)

    def _kaggle_imported(self, vids: list[str], source: str) -> None:
        self.c.refresh_videos()
        self._refresh_videos()
        if vids:
            self.c.video = None
            self._select_video(vids[0])
            self.c.log_info(f"Loaded {len(vids)} video(s) from {source}: {', '.join(vids)}", kind="ok")
        else:
            self.c.log_info(f"No video folders found in {source}.", kind="warn")

    def _extract_clicked(self, sr: int, ch: int) -> None:
        if self.c.video:
            self._extract(self.c.video.id, sr, ch)

    def _extract_current(self) -> None:
        self._extract_clicked(self.extract.rate.currentData(), self.extract.chan.currentData())

    def _extract(self, vid: str, sr: int, ch: int, then_select: bool = False) -> None:
        self.extract.set_state("Extracting audio...", ok=False, busy=True)
        w = Worker(lambda prog: self.backend.extract_audio(vid, sr, ch, prog))
        w.progress.connect(lambda f, m: self.extract.set_state(m or "Extracting audio...", ok=False, busy=True, fraction=f))
        w.done.connect(lambda path: self._extracted(vid, then_select))
        w.failed.connect(lambda msg: self._error("Audio extraction failed", msg))
        self._run_worker(w)

    def _extracted(self, vid: str, then_select: bool) -> None:
        self.c.refresh_videos()
        self.c.log_info("Audio extraction completed. 16 kHz, mono, 16-bit WAV.")
        self.extract.set_state("Audio ready")
        if then_select:
            self._select_video(vid)

    # pipeline
    def _run_pipeline(self) -> None:
        if not self.c.video or self._pipeline_busy:
            return
        if self.c.segments or self.c.dirty:
            n = len(self.c.segments)
            ans = QMessageBox.question(
                self,
                "Run pipeline",
                f"This video already has {n} segments (with any transcripts and your edits).\n"
                "Running the pipeline again cuts it from scratch and replaces all of them. "
                "Your change history stays in changes.csv.\n\nContinue?",
            )
            if ans != QMessageBox.Yes:
                return
        vid = self.c.video.id
        self._pipeline_busy = True
        self.b_run.setEnabled(False)
        self._last_stage_msg = ""
        w = Worker(lambda prog: self.backend.run_pipeline(vid, prog))
        w.progress.connect(self._pipeline_progress)
        w.done.connect(lambda segs: self._pipeline_done(vid, segs))
        w.failed.connect(lambda msg: (self._pipeline_failed(msg)))
        self._run_worker(w)
        self.c.pipelineChanged.emit()

    def _pipeline_progress(self, fraction: float, message: str) -> None:
        if message and message != getattr(self, "_last_stage_msg", ""):
            self._last_stage_msg = message
            self.c.log_info(message, kind="info")
        self.c.pipelineChanged.emit()

    def _pipeline_failed(self, msg: str) -> None:
        self._pipeline_busy = False
        self.b_run.setEnabled(True)
        self.c.pipelineChanged.emit()
        self._error("Pipeline failed", msg)

    def _pipeline_done(self, vid: str, segs) -> None:
        self._pipeline_busy = False
        self.b_run.setEnabled(True)
        if self.c.video and self.c.video.id == vid:
            self.c.replace_segments(segs)
            # keep the saved audit trail in step with what the backend already wrote
            self.c.log[:] = self.backend.load_log(vid)
            if any(s.asr_a or s.asr_b for s in segs):
                self.c.log_info("Pipeline finished: regions, segments, SNR and ASR updated.")
            else:
                self.c.log_info("Pipeline finished: regions, segments and SNR are ready.")
                self.c.log_info("Dual ASR is still pending: run it in the Kaggle runner, then Processing > Import ASR results.", kind="warn")
            self.c.pipelineChanged.emit()
            self.c.refresh_videos()

    def _reprocess_selected(self) -> None:
        if self.c.selected:
            self._reprocess(self.c.selected[0])

    def _reprocess(self, sid: int) -> None:
        s = self.c.seg(sid)
        if not s or not self.c.video:
            return
        vid = self.c.video.id
        copy_seg = copy.deepcopy(s)
        self.c.log_info(f"Re-running SNR and ASR for segment #{self.c.number_of(sid)}...", kind="info")
        w = Worker(lambda prog: self.backend.reprocess_segment(vid, copy_seg))
        w.done.connect(lambda new: self.c.apply_reprocessed(sid, new))
        w.failed.connect(lambda msg: self._error("Re-run failed", msg))
        self._run_worker(w)

    # segment actions
    def _play_segment(self, sid: int) -> None:
        s = self.c.seg(sid)
        if s:
            self.player.play_range(s.start, s.end)

    def _add(self) -> None:
        self.c.add(self.player.pos)

    def _split(self) -> None:
        t = self.player.pos
        sid = self.c.selected[0] if self.c.selected else None
        s = self.c.seg(sid) if sid else self.c.segment_at(t)
        if s:
            self.c.split(s.id, t if s.start < t < s.end else None)

    def _merge(self) -> None:
        if not self.c.merge(list(self.c.selected)):
            QMessageBox.information(self, "Merge", "Select two or more neighbouring segments (Ctrl+click) to merge.")

    def _delete(self) -> None:
        ids = list(self.c.selected)
        if len(ids) >= 3:
            ans = QMessageBox.question(self, "Delete", f"Delete {len(ids)} segments?")
            if ans != QMessageBox.Yes:
                return
        self.c.delete(ids)

    def _toggle_conflicts(self, on: bool) -> None:
        idx = self.table.status_combo.findData("conflicts" if on else "all")
        self.table.status_combo.setCurrentIndex(idx)

    # export
    def _export_csv(self) -> None:
        if not self.c.video:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"{self.c.video.id}_review.csv", "CSV files (*.csv)"
        )
        if path:
            export.write_csv(path, self.c.segments)
            self.backend.mark_exported(self.c.video.id)
            self.c.refresh_videos()
            self.c.log_info(f"Exported CSV: {Path(path).name}")

    def _export_all_csv(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export CSV for all videos")
        if not folder:
            return
        for v in self.c.videos:
            segs = self.c.segments if self.c.video and self.c.video.id == v.id else self.backend.load_segments(v.id)
            export.write_csv(Path(folder) / f"{v.id}_segments.csv", segs)
        self.c.log_info(f"Exported {len(self.c.videos)} CSV files")

    def _export_audio(self) -> None:
        if not self.c.video:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for segment .wav and .txt files")
        if not folder:
            return
        n = export.export_segment_audio(self.backend.audio_path(self.c.video.id), folder, self.c.video, self.c.segments)
        self.c.log_info(f"Exported {n} accepted segments (.wav + .txt)")

    # settings
    def _set_user(self) -> None:
        name, ok = QInputDialog.getText(self, "Annotator name", "Name recorded with every change:", text=self.c.user)
        if ok and name.strip():
            self.c.user = name.strip()

    def _set_threshold(self) -> None:
        val, ok = QInputDialog.getDouble(
            self, "SNR threshold", "Segments at or above this SNR (dB) are clean:", self.c.snr_threshold, 0, 60, 1
        )
        if ok:
            n = self.c.apply_snr_threshold(val)
            QMessageBox.information(self, "SNR threshold", f"{n} segment(s) changed their automatic clean/noisy value.")

    def _show_help(self) -> None:
        QMessageBox.information(
            self,
            "Shortcuts",
            "Space  play / pause\n"
            "Ctrl+Z / Ctrl+Y  undo / redo\n"
            "Ctrl+S  save      Ctrl+E  export CSV\n"
            "S  split at playhead      M  merge selected\n"
            "Ctrl+N  add segment at playhead      Del  delete\n"
            "N / Shift+N  next / previous conflict\n"
            "A  accept      D  mark difficult\n"
            "1 2 3 4  label: speech, music, mixed, silence\n"
            "C  show conflicts only\n\n"
            "Mouse: click selects and moves the playhead, double-click plays the segment,\n"
            "Ctrl+click adds to the selection, drag a region border to move the cut,\n"
            "wheel zooms, drag on empty space pans.",
        )

    def _stop_workers(self) -> None:
        # never leave a QThread running when the interpreter exits (it aborts the process)
        for w in list(self._workers):
            if w.isRunning() and not w.wait(8000):
                w.terminate()
                w.wait(2000)

    def closeEvent(self, ev) -> None:
        busy = [w for w in self._workers if w.isRunning()]
        if self._pipeline_busy or len(busy) > 1:
            ans = QMessageBox.question(
                self,
                "Quit",
                "A background task (download, extraction or pipeline) is still running.\n"
                "Quitting now stops it; finished steps are kept on disk.\n\nQuit anyway?",
            )
            if ans != QMessageBox.Yes:
                ev.ignore()
                return
        if getattr(self, "_startup_timer", None) is not None:
            self._startup_timer.stop()
        if self._download_timer is not None:
            self._download_timer.stop()
        if hasattr(self.backend, "shutdown_session"):
            self.backend.shutdown_session()
        if self.c.dirty:
            self.c.save()
        self.player.pause()
        super().closeEvent(ev)
