"""Contract between the GUI and the speech-processing backend.

The GUI only talks to this interface. When the real backend is ready,
subclass Backend (or adapt your classes to these methods) and pass it to
MainWindow in main.py. Every method is blocking: the GUI runs the slow ones
(download, extract_audio, run_pipeline, reprocess_segment) in a worker thread.

Time values are seconds (float). Audio is 16 kHz mono.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..models import Segment, VideoInfo

# progress callback: (fraction 0..1, message)
Progress = Callable[[float, str], None]


class Backend(ABC):
    # names shown for the two ASR systems (model A = first transcript column)
    asr_names: tuple[str, str] = ("Whisper", "Audar")
    # clean / noisy cut-off in dB used for the automatic label
    snr_threshold: float = 20.0

    @abstractmethod
    def list_videos(self) -> list[VideoInfo]:
        """All videos in the project, with their pipeline `steps` state."""

    @abstractmethod
    def download(self, url: str, progress: Progress) -> VideoInfo:
        """Step 1: download the video, return its info (title, channel, duration, license)."""

    @abstractmethod
    def extract_audio(self, video_id: str, sample_rate: int, channels: int, progress: Progress) -> str:
        """Step 2: extract audio, return the WAV path."""

    @abstractmethod
    def audio_path(self, video_id: str) -> str:
        """Path of the extracted 16 kHz mono WAV (used for playback)."""

    @abstractmethod
    def load_audio(self, video_id: str) -> tuple[np.ndarray, int]:
        """Mono float32 samples in [-1, 1] and the sample rate."""

    @abstractmethod
    def load_segments(self, video_id: str) -> list[Segment]:
        """Segments saved earlier (or produced by the pipeline), sorted by start."""

    @abstractmethod
    def load_log(self, video_id: str) -> list[dict]:
        """Saved annotator change log: dicts with time, user, segment, action, detail."""

    @abstractmethod
    def save_segments(self, video_id: str, segments: list[Segment], log: list[dict]) -> None:
        """Persist the annotator's work so it can be continued later."""

    @abstractmethod
    def run_pipeline(self, video_id: str, progress: Progress) -> list[Segment]:
        """Steps 3-6: region labels, segmentation, SNR, dual ASR. Returns the new segments."""

    def reprocess_segment(self, video_id: str, segment: Segment) -> Segment:
        """Optional: re-run SNR and both ASR systems after the annotator changed a cut."""
        return segment

    # optional features: the GUI shows a message when a backend does not provide them
    def add_local_file(self, path: str, progress: Progress) -> VideoInfo:
        """Ingest a local video/audio file instead of a URL."""
        raise NotImplementedError("This backend cannot add local files.")

    def import_asr(self, video_id: str, csv_path: str) -> list[Segment]:
        """Load the dual-ASR result CSV (made by the Kaggle runner) for this video."""
        raise NotImplementedError("This backend cannot import ASR results.")

    def mark_exported(self, video_id: str) -> None:
        """Called after the annotator exported the final CSV."""

    def clear_processing(self, video_id: str) -> VideoInfo | None:
        """Optional: delete cached Steps 3-6 processing outputs while keeping the downloaded video/audio."""
        raise NotImplementedError("This backend cannot clear processing caches.")
