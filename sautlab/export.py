"""CSV and segment-audio export, independent of the backend."""
from __future__ import annotations

import wave
from pathlib import Path

from .models import Segment, VideoInfo
from .review_csv import write_review_csv


def write_csv(path: str | Path, segments: list[Segment]) -> None:
    """Final CSV in the backend's review format (the 16 app columns)."""
    write_review_csv(path, segments)


def export_segment_audio(
    wav_path: str | Path, out_dir: str | Path, video: VideoInfo, segments: list[Segment]
) -> int:
    """Writes one .wav and one .txt per accepted speech/mixed segment. Returns the count."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with wave.open(str(wav_path), "rb") as src:
        sr = src.getframerate()
        width = src.getsampwidth()
        ch = src.getnchannels()
        for i, s in enumerate(segments, 1):
            if not (s.has_text and s.status == "accepted" and s.final_text.strip()):
                continue
            src.setpos(int(s.start * sr))
            frames = src.readframes(int((s.end - s.start) * sr))
            name = f"{video.id}_{i:04d}_{int(s.start * 1000):08d}_{int(s.end * 1000):08d}"
            with wave.open(str(out / f"{name}.wav"), "wb") as dst:
                dst.setnchannels(ch)
                dst.setsampwidth(width)
                dst.setframerate(sr)
                dst.writeframes(frames)
            (out / f"{name}.txt").write_text(s.final_text.strip() + "\n", encoding="utf-8")
            count += 1
    return count
