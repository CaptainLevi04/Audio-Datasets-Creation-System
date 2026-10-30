"""Step 2: Audio extraction and processing utilities.
Standardized to 16 kHz, mono, 16-bit WAV as required by the speech corpus project.
"""

from pathlib import Path
import shutil
import subprocess
import wave
import numpy as np

SAMPLE_RATE = 16000
DEFAULT_SR = SAMPLE_RATE


def check_ffmpeg() -> str:
    """Check if ffmpeg is available on PATH or known locations. Returns path to ffmpeg executable."""
    path = shutil.which("ffmpeg")
    if path is not None:
        return path

    # Optional fallback: check if imageio_ffmpeg is installed
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except ImportError:
        pass

    raise RuntimeError(
        "ffmpeg not found on PATH. Please install it:\n"
        "  - Windows: winget install ffmpeg (or scoop install ffmpeg)\n"
        "  - Ubuntu / Debian: sudo apt update && sudo apt install ffmpeg\n"
        "  - macOS: brew install ffmpeg"
    )


def extract_audio(video_path: str | Path, wav_path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Extract audio from video/media and convert to 16 kHz, mono, 16-bit PCM WAV.
    
    Args:
        video_path: Path to source media file (mp4, mkv, webm, mp3, etc.)
        wav_path: Target path for the output WAV file.
        sample_rate: Target sample rate in Hz (default: 16000).
        
    Returns:
        Path to the generated WAV file.
    """
    ffmpeg_bin = check_ffmpeg()
    video_path = Path(video_path)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{res.stderr.strip()}")
    return wav_path


def extract_audio_hq(video_path: str | Path, flac_path: str | Path) -> Path:
    """Lossless audio copy at source sample rate and channels for human annotator listening.
    
    Args:
        video_path: Path to source video file.
        flac_path: Target path for lossless flac file.
        
    Returns:
        Path to the created flac file.
    """
    ffmpeg_bin = check_ffmpeg()
    video_path = Path(video_path)
    flac_path = Path(flac_path)
    flac_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-c:a", "flac",
        str(flac_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg HQ audio extraction failed:\n{res.stderr.strip()}")
    return flac_path


def wav_info(wav_path: str | Path) -> dict:
    """Inspect WAV audio file header and return audio properties dict."""
    wav_path = Path(wav_path)
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth() * 8
        n = w.getnframes()
        dur = n / sr if sr > 0 else 0.0
        return {
            "sample_rate": sr,
            "channels": ch,
            "sample_width_bits": sw,
            "num_samples": n,
            "duration_sec": round(dur, 3),
        }


def read_wav(wav_path: str | Path) -> tuple[np.ndarray, int]:
    """Read standard WAV file into float32 samples in range [-1.0, 1.0].
    
    Returns:
        (samples, sample_rate)
    """
    wav_path = Path(wav_path)
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples.astype(np.float32), sr


def write_wav(wav_path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Save 1D float32 numpy array to 16-bit PCM mono WAV file without external dependencies.
    
    Args:
        wav_path: Output file path.
        samples: Float numpy array with values in [-1.0, 1.0].
        sample_rate: Audio sampling frequency in Hz (default: 16000).
        
    Returns:
        Path to written WAV file.
    """
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    
    clipped = np.clip(samples, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())
        
    return wav_path


def slice_wav(source_wav: str | Path, target_wav: str | Path,
              start_sec: float, end_sec: float) -> Path:
    """Slice an interval [start_sec, end_sec] from a WAV file and write to target_wav."""
    samples, sr = read_wav(source_wav)
    i0 = max(0, int(start_sec * sr))
    i1 = min(len(samples), int(end_sec * sr))
    sliced = samples[i0:i1]
    return write_wav(target_wav, sliced, sample_rate=sr)
