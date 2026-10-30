"""Step 1: Video downloader and metadata extractor.
Supports online video URLs (via yt-dlp) and local files.
Collects link, title, channel, duration, license, and creative_commons status.
"""

from pathlib import Path
import re
import shutil


def is_creative_commons(license_str: str | None) -> bool:
    """Check if the license string indicates Creative Commons or Public Domain."""
    if not license_str:
        return False
    norm = license_str.lower()
    return "creative commons" in norm or "cc-by" in norm or "public domain" in norm


def download_video(url: str, out_dir: str | Path, max_height: int | None = 480) -> tuple[Path, dict]:
    """Download a video via yt-dlp and extract full metadata.
    
    Args:
        url: Direct video URL (YouTube, Vimeo, etc.).
        out_dir: Destination folder.
        max_height: Maximum video vertical resolution (e.g., 480).
                    Restricting height significantly saves bandwidth and disk space
                    while preserving 100% of the original audio quality.
                    
    Returns:
        (video_path, metadata_dict)
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required for downloading videos. Install it with: pip install yt-dlp"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_height:
        fmt = (f"bv*[vcodec^=avc1][height<={max_height}]+ba"
               f"/bv*[height<={max_height}]+ba/bv*+ba/b")
    else:
        fmt = "bv*[vcodec^=avc1]+ba/bv*+ba/b"

    opts = {
        "format": fmt,
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_download_progress],
    }

    print(f"Fetching metadata and downloading: {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = Path(ydl.prepare_filename(info))

    if not video_path.exists():
        found = sorted(out_dir.glob("video.*"))
        if not found:
            raise RuntimeError(f"Download completed but output video file was not found in {out_dir}")
        video_path = found[0]

    fmts = info.get("requested_formats") or [info]
    src_audio = next((f for f in fmts if f.get("acodec") not in (None, "none")), {})

    meta = {
        "video_id": info.get("id") or "custom_video",
        "url": url,
        "webpage_url": info.get("webpage_url") or url,
        "title": info.get("title") or "Unknown Title",
        "channel": info.get("channel") or info.get("uploader") or "Unknown Channel",
        "duration_sec": info.get("duration") or 0,
        "license": info.get("license") or "Standard YouTube License",
        "creative_commons": is_creative_commons(info.get("license")),
        "upload_date": info.get("upload_date"),
        "language": info.get("language"),
        "source_audio_codec": src_audio.get("acodec"),
        "source_audio_bitrate_kbps": src_audio.get("abr"),
        "source_audio_sample_rate": src_audio.get("asr"),
    }
    return video_path, meta


def process_local_file(file_path: str | Path, out_dir: str | Path) -> tuple[Path, dict]:
    """Ingest a local video or audio file for the pipeline without downloading.
    
    Args:
        file_path: Path to local media file.
        out_dir: Destination folder.
        
    Returns:
        (copied_video_path, metadata_dict)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Local file not found: {file_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = out_dir / f"video{file_path.suffix}"
    if file_path.resolve() != target_path.resolve():
        shutil.copy2(file_path, target_path)

    clean_stem = re.sub(r"[^\w\-]", "_", file_path.stem)
    meta = {
        "video_id": clean_stem,
        "url": f"file:///{file_path.resolve().as_posix()}",
        "webpage_url": f"file:///{file_path.resolve().as_posix()}",
        "title": file_path.name,
        "channel": "Local User",
        "duration_sec": None,
        "license": "Local / Custom",
        "creative_commons": False,
        "upload_date": None,
        "language": "ara",
        "source_audio_codec": "local",
        "source_audio_bitrate_kbps": None,
        "source_audio_sample_rate": None,
    }
    return target_path, meta


def _download_progress(d):
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "").strip()
        speed = d.get("_speed_str", "").strip()
        eta = d.get("_eta_str", "").strip()
        print(f"\r  [yt-dlp] Downloading: {pct} | Speed: {speed} | ETA: {eta}    ", end="", flush=True)
    elif d["status"] == "finished":
        print("\r  [yt-dlp] Download finished successfully.                            ")
