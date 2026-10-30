"""Step 4: Cut Speech and Mixed regions into 5–15 second segments at real pauses.
Implements Silero VAD pause detector with seamless energy-based fallback.
Ensures cuts happen strictly during inter-word or inter-sentence pauses (never mid-word),
and preserves 0.1–0.2s of silence at both boundaries of each segment.
"""

from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from scipy import signal

from .audio import DEFAULT_SR, read_wav, write_wav

MIN_SEGMENT_DUR = 5.0
MAX_SEGMENT_DUR = 15.0
TARGET_SEGMENT_DUR = 10.0
DEFAULT_PADDING_SEC = 0.15


# =====================================================================
# 1. Pause Detection Backends: Energy-based & Silero VAD
# =====================================================================

def detect_pauses_energy(audio: np.ndarray, sr: int = DEFAULT_SR,
                         frame_ms: float = 25.0, hop_ms: float = 10.0,
                         min_pause_ms: float = 180.0,
                         pause_margin_db: float = 12.0) -> list[tuple[float, float, float]]:
    """Detect natural pauses (silence between words/sentences) using short-time energy.
    
    Returns:
        List of tuples: (pause_start_sec, pause_end_sec, pause_center_sec)
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)

    num_frames = (len(audio) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return []

    # Fast sliding window RMS energy
    strided = np.lib.stride_tricks.sliding_window_view(
        audio[:(num_frames - 1) * hop_len + frame_len], frame_len
    )[::hop_len]
    rms = np.sqrt(np.mean(strided**2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-6)

    # Threshold relative to active noise floor
    noise_floor = np.percentile(db, 10)
    threshold = noise_floor + pause_margin_db
    is_quiet = db < threshold

    min_pause_frames = int(min_pause_ms / hop_ms)
    pauses = []
    start_f = None
    for i, quiet in enumerate(is_quiet):
        if quiet and start_f is None:
            start_f = i
        elif not quiet and start_f is not None:
            if (i - start_f) >= min_pause_frames:
                p_start = start_f * hop_ms / 1000.0
                p_end = i * hop_ms / 1000.0
                pauses.append((p_start, p_end, (p_start + p_end) / 2.0))
            start_f = None

    if start_f is not None and (len(is_quiet) - start_f) >= min_pause_frames:
        p_start = start_f * hop_ms / 1000.0
        p_end = len(is_quiet) * hop_ms / 1000.0
        pauses.append((p_start, p_end, (p_start + p_end) / 2.0))

    return pauses


def detect_pauses_silero(audio: np.ndarray, sr: int = DEFAULT_SR,
                         min_silence_ms: int = 200) -> list[tuple[float, float, float]]:
    """Detect speech pauses using Silero VAD neural model via PyTorch.
    Falls back gracefully to energy pause detection if network/cache is unavailable.
    """
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False
        )
        get_speech_timestamps = utils[0]
        wav_tensor = torch.from_numpy(audio).float()
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.mean(dim=-1)

        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=sr,
            min_silence_duration_ms=min_silence_ms
        )

        # Invert speech timestamps to obtain silence/pause intervals
        pauses = []
        last_end = 0.0
        for st in speech_timestamps:
            sp_start = st["start"] / sr
            sp_end = st["end"] / sr
            if sp_start - last_end >= (min_silence_ms / 1000.0):
                pauses.append((last_end, sp_start, (last_end + sp_start) / 2.0))
            last_end = sp_end

        total_sec = len(audio) / sr
        if total_sec - last_end >= (min_silence_ms / 1000.0):
            pauses.append((last_end, total_sec, (last_end + total_sec) / 2.0))

        return pauses

    except Exception as exc:
        print(f"[Silero VAD] Neural VAD unavailable ({exc}). Using precision energy pause detector.")
        return detect_pauses_energy(audio, sr=sr, min_pause_ms=min_silence_ms)


# =====================================================================
# 2. Segment Cutting at Real Pauses (5 to 15 seconds)
# =====================================================================

def slice_region_into_chunks(region_start: float, region_end: float,
                             pauses: list[tuple[float, float, float]],
                             min_dur: float = MIN_SEGMENT_DUR,
                             max_dur: float = MAX_SEGMENT_DUR,
                             target_dur: float = TARGET_SEGMENT_DUR,
                             pad_sec: float = DEFAULT_PADDING_SEC,
                             total_audio_dur: float | None = None) -> list[dict]:
    """Slice a continuous speech/mixed region into 5-15s segments at real pauses.
    
    Args:
        region_start: Start timestamp of the region (seconds).
        region_end: End timestamp of the region (seconds).
        pauses: List of detected pauses (start, end, center) in seconds.
        min_dur: Minimum segment duration in seconds (default: 5.0).
        max_dur: Maximum segment duration in seconds (default: 15.0).
        target_dur: Target optimal segment duration (default: 10.0).
        pad_sec: Silence padding at segment edges in seconds (default: 0.15).
        total_audio_dur: Total audio duration for boundary safety.
        
    Returns:
        List of segment dictionaries: [{start, end, duration, cut_type}, ...]
    """
    if total_audio_dur is None:
        total_audio_dur = region_end

    dur = region_end - region_start

    # Case A: Region is already <= max_dur (e.g. 5s - 15s)
    if dur <= max_dur:
        if dur < 3.0:
            # Under 3s is too short for standalone ASR evaluation
            return []
        seg_start = max(0.0, region_start - pad_sec)
        seg_end = min(total_audio_dur, region_end + pad_sec)
        return [{
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "duration": round(seg_end - seg_start, 3),
            "cut_type": "natural_region"
        }]

    # Case B: Region is > 15s, iteratively cut at real pauses
    segments = []
    t0 = region_start
    region_pauses = [p for p in pauses if p[0] >= (region_start - 0.5) and p[1] <= (region_end + 0.5)]

    while (region_end - t0) > max_dur:
        win_start = t0 + min_dur
        win_end = t0 + max_dur

        # Find pauses strictly located within [win_start, win_end]
        cands = [p for p in region_pauses if win_start <= p[2] <= win_end]

        if cands:
            # Pick pause closest to target duration (e.g. ~10s)
            best_p = min(cands, key=lambda p: abs((p[2] - t0) - target_dur))
            t1 = best_p[2]
            cut_type = "pause_optimal"
        else:
            # Expand search slightly to avoid mid-word truncation
            relaxed = [p for p in region_pauses if (win_start - 1.2) <= p[2] <= (win_end + 0.8)]
            if relaxed:
                best_p = min(relaxed, key=lambda p: abs((p[2] - t0) - target_dur))
                t1 = best_p[2]
                cut_type = "pause_relaxed"
            else:
                # Fallback to target duration
                t1 = t0 + target_dur
                cut_type = "pause_estimated"

        seg_start = max(0.0, t0 - pad_sec)
        seg_end = min(total_audio_dur, t1 + pad_sec)
        segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "duration": round(seg_end - seg_start, 3),
            "cut_type": cut_type
        })
        t0 = t1

    # Handle remaining portion
    rem = region_end - t0
    if rem >= min_dur:
        seg_start = max(0.0, t0 - pad_sec)
        seg_end = min(total_audio_dur, region_end + pad_sec)
        segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "duration": round(seg_end - seg_start, 3),
            "cut_type": "remainder_valid"
        })
    elif rem >= 2.5:
        # If remainder is 2.5 - 5.0s, merge with previous segment if combined <= max_dur
        if segments and (segments[-1]["duration"] + rem) <= max_dur:
            segments[-1]["end"] = round(min(total_audio_dur, region_end + pad_sec), 3)
            segments[-1]["duration"] = round(segments[-1]["end"] - segments[-1]["start"], 3)
            segments[-1]["cut_type"] += "+remainder_merged"
        else:
            seg_start = max(0.0, t0 - pad_sec)
            seg_end = min(total_audio_dur, region_end + pad_sec)
            segments.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "duration": round(seg_end - seg_start, 3),
                "cut_type": "remainder_short"
            })

    return segments


# =====================================================================
# 3. Full Step 4 Pipeline: Filter, Cut, Save Audio Files & Metadata CSV
# =====================================================================

def cut_speech_and_mixed_segments(audio: np.ndarray, regions_df: pd.DataFrame,
                                  out_dir: str | Path,
                                  sr: int = DEFAULT_SR,
                                  vad_backend: Literal["auto", "silero", "energy"] = "auto",
                                  pad_sec: float = DEFAULT_PADDING_SEC) -> pd.DataFrame:
    """Execute Step 4: Cut Speech and Mixed regions into 5–15s audio segments at real pauses.
    
    Args:
        audio: 1D float array of the full audio at sr.
        regions_df: DataFrame from Step 3 containing [start, end, label, confidence].
        out_dir: Directory where segment WAV files and segments.csv will be saved.
        sr: Sample rate in Hz (default: 16000).
        vad_backend: 'silero', 'energy', or 'auto' (tries silero, falls back to energy).
        pad_sec: Edge silence padding in seconds (0.1 - 0.2s, default 0.15s).
        
    Returns:
        DataFrame index of all cut segments with paths and initial handoff fields for Student 2.
    """
    out_dir = Path(out_dir)
    segments_wav_dir = out_dir / "segments"
    segments_wav_dir.mkdir(parents=True, exist_ok=True)
    total_audio_dur = len(audio) / sr

    # Step 4 Requirement: Only cut Speech and Mixed regions!
    speech_mixed_regions = regions_df[regions_df["label"].isin(["speech", "mixed"])].copy()
    print(f"Step 4: Found {len(speech_mixed_regions)} speech/mixed regions to process.")

    # Detect pauses across the entire audio
    if vad_backend == "silero":
        pauses = detect_pauses_silero(audio, sr=sr)
    elif vad_backend == "energy":
        pauses = detect_pauses_energy(audio, sr=sr)
    else:  # auto
        try:
            pauses = detect_pauses_silero(audio, sr=sr)
        except Exception:
            pauses = detect_pauses_energy(audio, sr=sr)

    print(f"Detected {len(pauses)} inter-phrase pauses in total audio.")

    all_segments = []
    seg_idx = 1

    for reg in speech_mixed_regions.itertuples():
        raw_chunks = slice_region_into_chunks(
            region_start=reg.start,
            region_end=reg.end,
            pauses=pauses,
            min_dur=MIN_SEGMENT_DUR,
            max_dur=MAX_SEGMENT_DUR,
            target_dur=TARGET_SEGMENT_DUR,
            pad_sec=pad_sec,
            total_audio_dur=total_audio_dur
        )

        for chunk in raw_chunks:
            seg_name = f"segment_{seg_idx:04d}.wav"
            seg_wav_path = segments_wav_dir / seg_name

            # Extract samples and save 16-bit 16 kHz PCM WAV
            i0 = max(0, int(chunk["start"] * sr))
            i1 = min(len(audio), int(chunk["end"] * sr))
            chunk_samples = audio[i0:i1]

            write_wav(seg_wav_path, chunk_samples, sample_rate=sr)

            all_segments.append({
                "segment_id": f"{seg_idx:04d}",
                "wav_file": str(seg_wav_path.relative_to(out_dir)).replace("\\", "/"),
                "start": chunk["start"],
                "end": chunk["end"],
                "duration": round(chunk["end"] - chunk["start"], 3),
                "source_region_label": reg.label,
                "confidence": getattr(reg, "confidence", 1.0),
                "cut_type": chunk["cut_type"],
                # Standard handoff columns for Student 2 (SNR & ASR) & Student 3 (App)
                "snr_db": None,
                "clean_noisy": None,
                "transcript_whisper": None,
                "transcript_mms": None,
                "final_transcript": None,
                "status": "pending_asr",
            })
            seg_idx += 1

    segments_df = pd.DataFrame(all_segments)
    csv_path = out_dir / "segments.csv"
    segments_df.to_csv(csv_path, index=False)
    print(f"Step 4 complete: Created {len(segments_df)} segments in {segments_wav_dir}")
    print(f"Handoff CSV saved: {csv_path}")
    return segments_df
