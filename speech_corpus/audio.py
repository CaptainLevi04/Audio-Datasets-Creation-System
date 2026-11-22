"""Step 2: Audio extraction and processing utilities.
Standardized to 16 kHz, mono, 16-bit WAV as required by the speech corpus project.
Fast polyphase resampling, lossless/standard FFmpeg extraction, and header inspection.
"""

from pathlib import Path
import math
import shutil
import subprocess
import wave
import numpy as np
from scipy import signal
import soundfile as sf

SAMPLE_RATE = 16000
DEFAULT_SR = SAMPLE_RATE


def check_ffmpeg() -> bool:
    """Check if ffmpeg executable is present in PATH or known location."""
    if shutil.which("ffmpeg") is not None:
        return True
    try:
        import imageio_ffmpeg
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if exe.exists():
            return True
    except Exception:
        pass
    return False


def get_ffmpeg_bin() -> str:
    """Get absolute path to ffmpeg binary or raise helpful error."""
    path = shutil.which("ffmpeg")
    if path is not None:
        return path
    try:
        import imageio_ffmpeg, os
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if exe.exists():
            target = exe.parent / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if not target.exists():
                shutil.copy2(exe, target)
            if str(exe.parent) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(target)
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found on PATH. Please install it:\n"
        "  - Windows: winget install ffmpeg (or scoop install ffmpeg)\n"
        "  - Ubuntu / Debian: sudo apt update && sudo apt install ffmpeg\n"
        "  - macOS: brew install ffmpeg"
    )


def read_wav(path: str | Path, target_sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 array in [-1.0, 1.0].
    Performs polyphase resampling if sample rate differs from target_sr (fast, no FFT edge ringing).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        mono = signal.resample_poly(mono, int(target_sr) // g, int(sr) // g).astype(np.float32)
        sr = target_sr
    return np.ascontiguousarray(mono), sr


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = DEFAULT_SR) -> Path:
    """Save 1D float32 or int16 numpy array to standard 16-bit PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dtype != np.int16:
        pcm16 = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        pcm16 = audio
    sf.write(str(path), pcm16, sample_rate, subtype="PCM_16")
    return path


def extract_audio(video_path: str | Path, wav_path: str | Path,
                  sample_rate: int = SAMPLE_RATE, overwrite: bool = False) -> Path:
    """Extract audio from video/media and convert to 16 kHz, mono, 16-bit PCM WAV."""
    wav_path = Path(wav_path)
    if not overwrite and wav_path.exists() and wav_path.stat().st_size > 1000:
        return wav_path

    video_path = Path(video_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if check_ffmpeg():
        bin_path = get_ffmpeg_bin()
        codec = "flac" if Path(wav_path).suffix.lower() == ".flac" else "pcm_s16le"
        cmd = [
            bin_path, "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", codec,
            str(wav_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed:\n{res.stderr.strip()}")
    else:
        mono, _ = read_wav(video_path, target_sr=sample_rate)
        write_wav(wav_path, mono, sample_rate=sample_rate)
    return wav_path


def extract_audio_conditioned(video_path: str | Path, flac_path: str | Path,
                              sample_rate: int = SAMPLE_RATE, overwrite: bool = False) -> Path:
    """Extract conditioned 16 kHz mono lossless FLAC audio.
    NOTE: This file is lossless ONLY relative to the conditioned 16 kHz mono signal;
    it is NOT a higher-fidelity archive of the original source media.
    """
    flac_path = Path(flac_path)
    if not overwrite and flac_path.exists() and flac_path.stat().st_size > 1000:
        return flac_path

    video_path = Path(video_path)
    flac_path.parent.mkdir(parents=True, exist_ok=True)

    if check_ffmpeg():
        bin_path = get_ffmpeg_bin()
        cmd = [
            bin_path, "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "flac",
            str(flac_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg conditioned audio extraction failed:\n{res.stderr.strip()}")
    else:
        mono, _ = read_wav(video_path, target_sr=sample_rate)
        sf.write(str(flac_path), mono, sample_rate, format="FLAC")
    return flac_path


def extract_source_archive(video_path: str | Path, archive_path: str | Path,
                           overwrite: bool = False) -> Path:
    """Extract genuine source-rate, source-channel lossless archive (FLAC) from container
    without downmixing or resampling. This is the only file entitled to the 'HQ' / 'Source Archive' label.
    """
    archive_path = Path(archive_path)
    if not overwrite and archive_path.exists() and archive_path.stat().st_size > 1000:
        return archive_path

    video_path = Path(video_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if check_ffmpeg():
        bin_path = get_ffmpeg_bin()
        cmd = [
            bin_path, "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn",
            "-c:a", "flac",
            str(archive_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            # Fallback if source audio codec cannot be directly packaged to flac
            cmd_pcm = [
                bin_path, "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-vn",
                "-c:a", "pcm_s16le",
                str(archive_path.with_suffix(".wav")),
            ]
            subprocess.run(cmd_pcm, capture_output=True, text=True)
            return archive_path.with_suffix(".wav")
    return archive_path


def extract_audio_hq(src: str | Path, dst_wav: str | Path, sr: int = DEFAULT_SR) -> Path:
    """Legacy compatibility helper.
    Redirects to extract_audio_conditioned if target is 16 kHz mono, or extract_source_archive
    if target is a genuine source archive, ensuring conditioned 16 kHz is never misrepresented as HQ.
    """
    dst_p = Path(dst_wav)
    if dst_p.suffix.lower() == ".flac":
        if "conditioned" in dst_p.name or sr == DEFAULT_SR:
            return extract_audio_conditioned(src, dst_wav, sample_rate=sr)
        return extract_source_archive(src, dst_wav)
    return extract_audio(src, dst_wav, sample_rate=sr)


def wav_info(wav_path: str | Path) -> dict:
    """Inspect WAV audio file header and return audio properties dict."""
    wav_path = Path(wav_path)
    info = sf.info(str(wav_path))
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "num_samples": info.frames,
        "duration_sec": round(info.duration, 3),
    }


def slice_wav(source_wav: str | Path, target_wav: str | Path,
              start_sec: float, end_sec: float, sr: int = DEFAULT_SR) -> Path:
    """Slice an interval [start_sec, end_sec] from a WAV file and write to target_wav."""
    samples, file_sr = read_wav(source_wav, target_sr=sr)
    i0 = max(0, int(start_sec * file_sr))
    i1 = min(len(samples), int(end_sec * file_sr))
    return write_wav(target_wav, samples[i0:i1], sample_rate=file_sr)
