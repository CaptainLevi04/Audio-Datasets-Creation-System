"""Audio playback with a playhead and optional range/loop.

Uses QMediaPlayer when an audio output device exists. Otherwise (headless
machines, tests) a simulated clock moves the playhead so the GUI still works.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QElapsedTimer, QObject, QTimer, QUrl, Signal


class Player(QObject):
    positionChanged = Signal(float)
    playingChanged = Signal(bool)
    rangeFinished = Signal()

    def __init__(self):
        super().__init__()
        self.duration = 0.0
        self.pos = 0.0
        self.rate = 1.0
        self.volume = 0.8
        self.loop = False
        self.range: tuple[float, float] | None = None
        self.playing = False
        self._qt = None
        self._out = None
        self._path: str | None = None

        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick_sim)

    @property
    def uses_device(self) -> bool:
        return self._qt is not None

    def load(self, path: str, duration: float) -> None:
        self.stop()
        self.duration = duration
        self.pos = 0.0
        self.range = None
        self._path = path
        self._setup_qt(path)
        self.positionChanged.emit(0.0)

    def _setup_qt(self, path: str) -> None:
        self._qt = None
        # SAUTLAB_NO_AUDIO=1 forces the simulated clock (tests, machines without sound)
        if os.environ.get("SAUTLAB_NO_AUDIO") or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer

            if not QMediaDevices.audioOutputs():
                return
            self._out = QAudioOutput()
            self._out.setVolume(self.volume)
            qt = QMediaPlayer()
            qt.setAudioOutput(self._out)
            qt.setSource(QUrl.fromLocalFile(path))
            qt.setPlaybackRate(self.rate)
            qt.positionChanged.connect(self._on_qt_position)
            self._qt = qt
        except Exception:
            self._qt = None

    def _on_qt_position(self, ms: int) -> None:
        if not self.playing:
            return
        self.pos = ms / 1000.0
        self.positionChanged.emit(self.pos)
        self._check_range()

    def _tick_sim(self) -> None:
        dt = self._clock.restart() / 1000.0
        self.pos = min(self.pos + dt * self.rate, self.duration)
        self.positionChanged.emit(self.pos)
        self._check_range()
        if self.pos >= self.duration:
            self.pause()

    def _check_range(self) -> None:
        if self.range and self.pos >= self.range[1]:
            if self.loop:
                self.seek(self.range[0])
                return
            self.pause()
            self.pos = self.range[1]
            self.positionChanged.emit(self.pos)
            self.rangeFinished.emit()

    # controls
    def play(self) -> None:
        if self.duration <= 0 or self.playing:
            return
        if self.pos >= self.duration - 0.05:
            self.pos = 0.0
        self.playing = True
        if self._qt is not None:
            self._qt.setPosition(int(self.pos * 1000))
            self._qt.play()
        else:
            self._clock.start()
            self._timer.start()
        self.playingChanged.emit(True)

    def pause(self) -> None:
        if not self.playing:
            return
        self.playing = False
        if self._qt is not None:
            self._qt.pause()
        else:
            self._timer.stop()
        self.playingChanged.emit(False)

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            if self.range and not (self.range[0] <= self.pos < self.range[1]):
                self.range = None
            self.play()

    def stop(self) -> None:
        self.range = None
        was = self.playing
        self.pause()
        self.seek(0.0)
        if was:
            self.playingChanged.emit(False)

    def seek(self, t: float) -> None:
        t = min(max(t, 0.0), self.duration)
        self.pos = t
        if self._qt is not None:
            self._qt.setPosition(int(t * 1000))
        self.positionChanged.emit(t)

    def play_range(self, start: float, end: float) -> None:
        self.pause()
        self.range = (max(0.0, start), min(self.duration, end))
        self.seek(self.range[0])
        self.play()

    def clear_range(self) -> None:
        self.range = None

    def set_rate(self, rate: float) -> None:
        self.rate = rate
        if self._qt is not None:
            self._qt.setPlaybackRate(rate)

    def set_volume(self, v: float) -> None:
        self.volume = v
        if self._out is not None:
            self._out.setVolume(v)

    def set_loop(self, on: bool) -> None:
        self.loop = on
