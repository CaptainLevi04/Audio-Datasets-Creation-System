"""CSV and segment-audio export, independent of the backend."""
from __future__ import annotations

import csv
import wave
from pathlib import Path

from .models import Segment, VideoInfo
from .review_csv import write_review_csv

SUMMARY_COLUMNS = [
    "Video",
    "Duration",
    "Speech",
    "Music",
    "Mixed",
    "Segments",
    "Mean SNR",
    "Accepted",
    "Chosen by hand",
    "Difficult",
]


def _fmt_min(seconds: float) -> str:
    if 0.0 < seconds < 60.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{round(seconds / 60.0)} min"


def summarize_video(video: VideoInfo, segments: list[Segment]) -> dict[str, str | int]:
    """Build one Summary Table row matching the corpus report schema."""
    total_dur = float(video.duration) if getattr(video, "duration", 0) > 0 else sum(s.duration for s in segments)
    speech_dur = sum(s.duration for s in segments if s.label == "speech")
    music_dur = sum(s.duration for s in segments if s.label == "music")
    mixed_dur = sum(s.duration for s in segments if s.label == "mixed")

    snr_vals = [s.snr_mean for s in segments if s.snr_mean is not None]
    mean_snr_str = f"{round(sum(snr_vals) / len(snr_vals))} dB" if snr_vals else ""

    base = [s for s in segments if s.has_text] or segments
    n_base = len(base)
    if n_base:
        acc_pct = round(100.0 * sum(1 for s in base if s.status == "accepted") / n_base)
        hand_pct = round(100.0 * sum(1 for s in base if s.status == "needs_review") / n_base)
        diff_pct = round(100.0 * sum(1 for s in base if s.status == "difficult") / n_base)
    else:
        acc_pct = hand_pct = diff_pct = 0

    title = (getattr(video, "title", "") or "").strip()
    label = f"{video.id} — {title}" if title and title != video.id else str(video.id)
    return {
        "Video": label,
        "Duration": _fmt_min(total_dur),
        "Speech": _fmt_min(speech_dur),
        "Music": _fmt_min(music_dur),
        "Mixed": _fmt_min(mixed_dur),
        "Segments": len(segments),
        "Mean SNR": mean_snr_str,
        "Accepted": f"{acc_pct}%",
        "Chosen by hand": f"{hand_pct}%",
        "Difficult": f"{diff_pct}%",
    }


def write_summary_csv(path: str | Path, items: list[tuple[VideoInfo, list[Segment]]]) -> Path:
    """Write the corpus Summary Table CSV (UTF-8-SIG for Excel compatibility)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for video, segs in items:
            writer.writerow(summarize_video(video, segs))
    return out


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
