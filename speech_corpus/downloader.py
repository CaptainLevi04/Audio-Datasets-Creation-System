"""Step 1: Video downloader and metadata extractor.
Supports online video URLs (via yt-dlp) and local files.
Collects link, title, channel, duration, license, and creative_commons status.
Saves info.json compliant with project specification.
"""

from pathlib import Path
import json
import re
import shutil
import time
from typing import Optional, Tuple

from .audio import DEFAULT_SR, extract_audio, extract_audio_hq, read_wav, write_wav


def is_creative_commons(license_str_or_meta) -> bool:
    """True only if yt-dlp's metadata indicates Creative Commons (no loose substring match)."""
    if isinstance(license_str_or_meta, dict):
        lic = str(license_str_or_meta.get("license") or "").lower()
    else:
        lic = str(license_str_or_meta or "").lower()
    return "creative commons" in lic or "cc-by" in lic or "public domain" in lic


def extract_video_id(url_or_path: str) -> str:
    """Extract YouTube 11-char ID or sanitize local filename."""
    m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$|\/)", url_or_path)
    if m:
        return m.group(1)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", Path(url_or_path).stem)
    return name[:32] if name else "sample_video"


def _save_info(out_dir: Path, info: dict) -> dict:
    """Write info.json to out_dir and return info dict."""
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def find_cached_video(url: str, search_dirs: list = None) -> Optional[Tuple[Path, dict]]:
    """Check if the video URL was already downloaded previously on the local machine."""
    vid_id = extract_video_id(url)
    if search_dirs is None:
        search_dirs = [Path("data"), Path("../data")]

    for s_dir in search_dirs:
        s_dir = Path(s_dir)
        if not s_dir.exists():
            continue
        cand_dir = s_dir / vid_id
        if cand_dir.exists() and (cand_dir / "audio.wav").exists():
            info_file = cand_dir / "info.json"
            if info_file.exists():
                try:
                    meta = json.loads(info_file.read_text(encoding="utf-8"))
                    return cand_dir / "audio.wav", meta
                except Exception:
                    pass
    return None


def download_video(url: str, out_dir: str | Path, max_duration: Optional[int] = None,
                   cookies_file: Optional[str | Path] = None) -> Tuple[Path, dict]:
    """Download best audio stream via yt-dlp, convert to 16 kHz WAV, save info.json.
    
    Args:
        url: YouTube video URL.
        out_dir: Target output directory.
        max_duration: Optional duration limit in seconds.
        cookies_file: Path to cookies.txt to bypass YouTube bot blocks on Colab.
        
    Returns:
        (wav_path, info_dict)
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path, info_path = out_dir / "audio.wav", out_dir / "info.json"

    if wav_path.exists() and info_path.exists():
        print(f"⚡ Cached audio + info found in {out_dir}")
        return wav_path, json.loads(info_path.read_text(encoding="utf-8"))

    base = {"quiet": True, "no_warnings": True, "noplaylist": True}
    if cookies_file:
        base["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(base) as ydl:
        meta = ydl.extract_info(url, download=False)

    dur = meta.get("duration")
    if max_duration and dur and dur > max_duration:
        raise ValueError(f"Video is {dur}s, longer than max_duration={max_duration}s")

    info = {
        "video_id": extract_video_id(url),
        "source": "youtube",
        "url": url,
        "webpage_url": meta.get("webpage_url") or url,
        "title": meta.get("title") or "Unknown Title",
        "channel": meta.get("channel") or meta.get("uploader") or "Unknown Channel",
        "duration_s": dur,
        "upload_date": meta.get("upload_date"),
        "license": meta.get("license") or "Standard YouTube License",
        "is_creative_commons": is_creative_commons(meta),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not info["is_creative_commons"]:
        print(f"⚠️ License is '{info['license']}', not Creative Commons. Document this in the report.")

    if not wav_path.exists():
        opts = dict(base, format="bestaudio/best", outtmpl=str(out_dir / "temp_video.%(ext)s"))
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        temp = list(out_dir.glob("temp_video.*"))
        if not temp:
            raise RuntimeError("yt-dlp produced no file (on Colab: YouTube may block the IP; pass cookies_file or upload audio).")
        extract_audio_hq(temp[0], wav_path)
        for f in temp:
            f.unlink(missing_ok=True)

    _save_info(out_dir, info)
    return wav_path, info


def ingest_local_file(src: str | Path, out_dir: str | Path) -> Tuple[Path, dict]:
    """Ingest a local audio/video file instead of a URL. Returns (wav_path, info_dict)."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = extract_audio_hq(src, out_dir / "audio.wav")
    import soundfile as sf
    dur = sf.info(str(wav_path)).duration
    info = {
        "video_id": out_dir.name,
        "source": "local_file",
        "url": str(src.resolve() if hasattr(src, "resolve") else src),
        "title": Path(src).name,
        "channel": "Local User",
        "duration_s": round(dur, 2),
        "license": "Local / Custom",
        "is_creative_commons": False,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return wav_path, _save_info(out_dir, info)


def process_local_file(file_path: str | Path, out_dir: str | Path) -> Tuple[Path, dict]:
    """Backward-compatible alias for ingest_local_file."""
    return ingest_local_file(file_path, out_dir)
