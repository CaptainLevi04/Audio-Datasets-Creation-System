"""Video segment clipping utilities.
Cuts video chunks corresponding to the 5–15s speech segments.
Supports lossless ffmpeg stream copy and MoviePy.
"""

from pathlib import Path
import subprocess
import pandas as pd
from .audio import check_ffmpeg


def cut_video_segments_ffmpeg(video_path: str | Path, segments_df: pd.DataFrame,
                             out_dir: str | Path) -> list[Path]:
    ffmpeg_bin = check_ffmpeg()
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    segments_dir = out_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    output_clips = []
    for row in segments_df.itertuples():
        seg_id = getattr(row, "segment_id", f"{row.Index:04d}")
        out_clip = segments_dir / f"segment_{seg_id}.mp4"
        
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-ss", str(row.start),
            "-to", str(row.end),
            "-i", str(video_path),
            "-c", "copy",
            str(out_clip)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out_clip.exists():
            output_clips.append(out_clip)

    return output_clips


def cut_video_segments_moviepy(video_path: str | Path, segments_df: pd.DataFrame,
                               out_dir: str | Path) -> list[Path]:
    try:
        import moviepy.editor as mpy
    except ImportError:
        try:
            import moviepy
            mpy = moviepy
        except ImportError:
            raise ImportError("moviepy is not installed. Install with: pip install moviepy")

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    segments_dir = out_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    clip = mpy.VideoFileClip(str(video_path))
    output_clips = []

    try:
        for row in segments_df.itertuples():
            seg_id = getattr(row, "segment_id", f"{row.Index:04d}")
            out_clip = segments_dir / f"segment_{seg_id}.mp4"
            
            sub = clip.subclip(float(row.start), float(row.end))
            sub.write_videofile(
                str(out_clip),
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            output_clips.append(out_clip)
    finally:
        clip.close()

    return output_clips


def cut_video_from_timestamp_file(video_path: str | Path, timestamps_file: str | Path,
                                  out_dir: str | Path) -> list[Path]:
    with open(timestamps_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    rows = []
    for i, parts in enumerate(lines, start=1):
        if len(parts) >= 2:
            rows.append({
                "segment_id": f"{i:04d}",
                "start": float(parts[0]),
                "end": float(parts[1])
            })
    df = pd.DataFrame(rows)
    return cut_video_segments_ffmpeg(video_path, df, out_dir)
