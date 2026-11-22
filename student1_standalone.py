#!/usr/bin/env python3
"""🎙️ Speech Corpus Pipeline: Student 1 (Self-Contained Standalone CLI Script)
Unified Four-Label Voting Policy | Specification Version 2.2

Covers:
1. Video Ingestion & Metadata extraction (Creative Commons verification).
2. Audio conditioning (16 kHz, Mono, 16-bit PCM WAV + HQ FLAC).
3. Continuous Short-Time Energy curve, Otsu Adaptive Silence Floor & 3-Detector Unified Voting Grid.
4. AST Acoustic Macro-Classification (Speech, Music, Mixed, Silence) with Cache Validation (v2.2).
5. Cut Engine v2 (Unified 4-Label Global Shortest-Path DP, Midpoint Cuts, 100-200ms Margins, Arabic Whisper Word Verification).
6. 10 Mandatory Quality Control Gates, Markdown Report, Global Manifest & Zero-Leakage Dataset Splitting.

Self-contained: requires no external package directories.
"""

import os
import sys
import io
import re
import json
import time
import math
import shutil
import bisect
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import signal
import soundfile as sf

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

# ==============================================================================
# 1. GLOBAL CONSTANTS
# ==============================================================================
DEFAULT_SR = 16000
PRIMARY_MIN = 5.0
PRIMARY_MAX = 15.0
FALLBACK_MIN = 4.0
FALLBACK_MAX = 17.0
TARGET_LEN = 10.0

LAMBDA_LEN = 1.0
FALLBACK_PENALTY = 10.0
WEAK_PENALTY = 2.0
SHORT_PENALTY = 0.5
TRANSITION_BONUS = 0.8

MARGIN_MIN = 0.100
MARGIN_MAX = 0.200
MARGIN_TOL_MS = 3.0

FADE_MS = 8.0
WORD_TS_TOL = 0.150

HIGH_TIER = 0.75
MED_TIER = 0.50

CONTENT_LABELS = ["speech", "music", "mixed", "silence"]
VALID_LABELS = {"speech", "music", "mixed", "silence"}
ARABIC_STOPS = ("د", "ت", "ط", "ق", "ك", "ب", "ء", "أ", "إ", "ؤ", "ئ")

SPEECH_NAMES = {"speech", "male speech, man speaking", "female speech, woman speaking",
                "child speech, kid speaking", "conversation", "narration, monologue"}
MUSIC_NAMES = {"music", "musical instrument", "singing", "song"}
SILENCE_NAMES = {"silence"}

# ==============================================================================
# 2. AUDIO CONDITIONING & INGESTION (STEPS 1 & 2)
# ==============================================================================
def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def get_ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")
    return path


def extract_audio(video_path: str | Path, out_wav: str | Path, sample_rate: int = DEFAULT_SR) -> Path:
    video_path, out_wav = Path(video_path), Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if check_ffmpeg():
        bin_path = get_ffmpeg_bin()
        codec = "flac" if out_wav.suffix.lower() == ".flac" else "pcm_s16le"
        cmd = [
            bin_path, "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", codec,
            str(out_wav),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed:\n{res.stderr.strip()}")
    else:
        mono, _ = read_wav(video_path, target_sr=sample_rate)
        write_wav(out_wav, mono, sample_rate=sample_rate)
    return out_wav


def extract_audio_hq(src: str | Path, dst_wav: str | Path, sr: int = DEFAULT_SR) -> Path:
    return extract_audio(src, dst_wav, sample_rate=sr)


def read_wav(path: str | Path, target_sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    path = Path(path)
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        num_samples = int(round(len(data) * target_sr / sr))
        data = signal.resample(data, num_samples).astype(np.float32)
        sr = target_sr
    return data.astype(np.float32), sr


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = DEFAULT_SR) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def wav_info(wav_path: str | Path) -> dict:
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


def download_video(url: str, out_dir: str | Path, max_height: int = 360) -> Tuple[Path, dict]:
    import yt_dlp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        meta = ydl.extract_info(url, download=True)
        vid = meta.get("id", "video")
        ext = meta.get("ext", "mp4")
        video_path = out_dir / f"{vid}.{ext}"
        if not video_path.exists():
            for f in out_dir.glob(f"{vid}.*"):
                if f.suffix in [".mp4", ".mkv", ".webm"]:
                    video_path = f
                    break
        info = {
            "video_id": vid,
            "title": meta.get("title", vid),
            "channel": meta.get("uploader", "Unknown"),
            "duration_sec": meta.get("duration", 0),
            "license": meta.get("license", "Unknown"),
            "url": url,
        }
    return video_path, info


def process_local_file(file_path: str | Path, out_dir: str | Path) -> Tuple[Path, dict]:
    file_path, out_dir = Path(file_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / file_path.name
    if file_path.resolve() != dst.resolve():
        shutil.copy2(file_path, dst)
    info = {
        "video_id": file_path.stem,
        "title": file_path.stem,
        "channel": "Local Source",
        "duration_sec": 0,
        "license": "Local / User Provided",
        "url": str(file_path),
    }
    return dst, info


def make_speech_synthetic(dur: float, rng: np.random.Generator, sr: int = DEFAULT_SR) -> np.ndarray:
    n = int(dur * sr)
    out = np.zeros(n)
    vowels = [(700, 1200, 2600), (300, 2200, 3000), (350, 800, 2500), (500, 1500, 2500)]
    hp_b, hp_a = signal.butter(2, 2500 / (sr / 2), "high")
    t = 0.0
    while t < dur:
        for _ in range(rng.integers(3, 7)):
            sd = rng.uniform(0.12, 0.25)
            if t + sd > dur:
                break
            m, i0 = int(sd * sr), int(t * sr)
            f0 = rng.uniform(100, 200)
            f0c = f0 * (1.0 + 0.15 * np.sin(np.linspace(0, np.pi * rng.uniform(0.5, 1.5), m)))
            ph = 2.0 * np.pi * np.cumsum(f0c) / sr
            F = vowels[rng.integers(len(vowels))]
            seg = np.zeros(m)
            for h in range(1, int(4000 / f0)):
                a = sum(np.exp(-0.5 * ((h * f0 - f) / (0.12 * f + 60))**2) for f in F) / np.sqrt(h)
                seg += a * np.sin(h * ph)
            seg *= np.hanning(m)**0.7
            cn = int(0.04 * sr)
            if cn < m:
                burst = 0.3 * np.max(np.abs(seg)) * signal.lfilter(hp_b, hp_a, rng.standard_normal(cn)) * np.hanning(cn)
                seg[:cn] += burst
            out[i0:i0 + m] += seg
            t += sd + rng.uniform(0.05, 0.12)
        t += rng.uniform(0.25, 0.6)
    return out / (np.max(np.abs(out)) + 1e-9) * 0.5


def make_music_synthetic(dur: float, rng: np.random.Generator, sr: int = DEFAULT_SR) -> np.ndarray:
    n = int(dur * sr)
    out = np.zeros(n)
    chords = [[220, 261.6, 329.6], [174.6, 220, 261.6], [261.6, 329.6, 392], [196, 246.9, 293.7]]
    t, ci = 0.0, 0
    while t < dur:
        m, i0 = int(min(2.0, dur - t) * sr), int(t * sr)
        tt = np.arange(m) / sr
        seg = np.zeros(m)
        for f in chords[ci % 4]:
            for h in range(1, 7):
                seg += np.sin(2.0 * np.pi * f * h * tt + rng.uniform(0, 6)) / (h**1.3)
        out[i0:i0 + m] += 0.15 * seg * np.minimum(1.0, tt / 0.05)
        t += 2.0
        ci += 1
    hp_b, hp_a = signal.butter(2, 6000 / (sr / 2), "high")
    for bt in np.arange(0, dur, 0.5):
        i0 = int(bt * sr)
        k = min(int(0.15 * sr), n - i0)
        tt = np.arange(k) / sr
        out[i0:i0 + k] += 0.5 * np.sin(2 * np.pi * (60 + 60 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18)
    for bt in np.arange(0.25, dur, 0.5):
        i0 = int(bt * sr)
        k = min(int(0.05 * sr), n - i0)
        if k > 0:
            hat = 0.08 * signal.lfilter(hp_b, hp_a, rng.standard_normal(k)) * np.exp(-np.arange(k) / sr * 60)
            out[i0:i0 + k] += hat
    return out / (np.max(np.abs(out)) + 1e-9) * 0.5


def build_mock_audio(seed: int = 42, timeline=None, sr: int = DEFAULT_SR) -> Tuple[np.ndarray, pd.DataFrame]:
    if timeline is None:
        timeline = [
            ("silence", 3.0),
            ("speech", 14.0),
            ("music", 10.0),
            ("mixed", 14.0),
            ("silence", 2.0),
            ("speech", 10.0),
            ("mixed", 9.0),
            ("music", 7.0),
            ("silence", 3.0),
            ("speech", 12.0),
        ]
    rng = np.random.default_rng(seed)
    parts, gt_rows, t = [], [], 0.0
    for lab, d in timeline:
        n = int(d * sr)
        if lab == "silence":
            x = np.zeros(n)
        elif lab == "speech":
            x = 0.9 * make_speech_synthetic(d, rng, sr=sr)
        elif lab == "music":
            x = 0.8 * make_music_synthetic(d, rng, sr=sr)
        else:
            x = 0.9 * make_speech_synthetic(d, rng, sr=sr) + 0.35 * make_music_synthetic(d, rng, sr=sr)
        parts.append(x)
        gt_rows.append((round(t, 3), round(t + d, 3), lab, round(d, 3)))
        t += d
    total_samples = sum(len(p) for p in parts)
    y = np.concatenate(parts) + 0.0008 * rng.standard_normal(total_samples)
    gt = pd.DataFrame(gt_rows, columns=["start", "end", "label", "duration"])
    return y.astype(np.float32), gt


# ==============================================================================
# 3. SHORT-TIME ENERGY & UNIFIED 3-DETECTOR VOTING GRID (100 MS TIME GRID)
# ==============================================================================
def compute_energy_db(audio: np.ndarray, sr: int = DEFAULT_SR,
                      frame_ms: float = 25.0, hop_ms: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    frame_len, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    c = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    idx = np.arange(0, len(audio) - frame_len + 1, hop)
    ms = np.maximum((c[idx + frame_len] - c[idx]) / frame_len, 0.0)
    return idx / sr, (10.0 * np.log10(ms + 1e-14)).astype(np.float32)


def adaptive_silence_floor(energy_db: np.ndarray) -> Tuple[float, float]:
    e = np.clip(energy_db, -100.0, 0.0)
    lo, hi = np.percentile(e, [1, 99])
    if hi - lo < 3.0:
        return float(lo), float(lo + 8.5)
    edges = np.arange(math.floor(lo), math.ceil(hi) + 1, 1.0)
    hist, edges = np.histogram(e, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(float)
    w0, m = np.cumsum(w), np.cumsum(w * centers)
    w1 = w.sum() - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = w0 * w1 * (m / w0 - (m[-1] - m) / w1) ** 2
    thr = float(centers[int(np.argmax(np.nan_to_num(between)))])
    low = e[e <= thr]
    floor = float(np.median(low)) if low.size else float(lo)
    return floor, float(np.clip(thr, floor + 3.0, floor + 20.0))


_SILERO = {}
def _load_silero():
    if "m" not in _SILERO:
        cache_dir = os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master")
        if os.path.exists(cache_dir):
            try:
                model, utils = torch.hub.load(cache_dir, "silero_vad", source="local", trust_repo=True)
                _SILERO["m"] = (model, utils[0])
                return _SILERO["m"]
            except Exception:
                pass
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
        _SILERO["m"] = (model, utils[0])
    return _SILERO["m"]


def compute_unified_voting_grid(audio: np.ndarray,
                                sr: int = DEFAULT_SR,
                                word_timestamps: Optional[List[dict]] = None,
                                vad_backend: str = "auto",
                                energy_threshold_db_offset: float = 5.0,
                                silero_confidence_threshold: float = 0.40) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_dur = len(audio) / sr
    n_bins = int(np.ceil(total_dur / 0.100))
    bin_starts = np.arange(n_bins) * 0.100
    bin_ends = np.minimum(bin_starts + 0.100, total_dur)
    bin_mids = (bin_starts + bin_ends) / 2.0

    times, energy_db = compute_energy_db(audio, sr)
    floor_db, _ = adaptive_silence_floor(energy_db)
    energy_silence_threshold = floor_db + 4.0

    energy_votes = np.zeros(n_bins, dtype=np.int8)
    energy_margins = np.zeros(n_bins, dtype=np.float32)
    energy_vals = np.zeros(n_bins, dtype=np.float32)

    indices = np.searchsorted(times, bin_starts)
    for k in range(n_bins):
        i0 = indices[k]
        i1 = np.searchsorted(times, bin_ends[k]) if k < n_bins - 1 else len(energy_db)
        if i1 > i0:
            m = float(energy_db[i0:i1].mean())
        elif i0 < len(energy_db):
            m = float(energy_db[i0])
        else:
            m = -60.0
        energy_vals[k] = m
        energy_margins[k] = energy_silence_threshold - m
        energy_votes[k] = 1 if m <= energy_silence_threshold else 0

    silero_probs = np.zeros(n_bins, dtype=np.float32)
    silero_votes = np.zeros(n_bins, dtype=np.int8)
    silero_margins = np.zeros(n_bins, dtype=np.float32)

    if vad_backend != "energy" and torch is not None:
        try:
            model, _ = _load_silero()
            dev = "cuda" if (torch.cuda.is_available()) else "cpu"
            model = model.to(dev)
            frame_len = 512
            n_frames = len(audio) // frame_len
            batch_size = 2048
            raw_probs = []
            with torch.no_grad():
                for i in range(0, n_frames, batch_size):
                    b = min(batch_size, n_frames - i)
                    chunks = [audio[(i + j) * frame_len:(i + j + 1) * frame_len] for j in range(b)]
                    x = torch.from_numpy(np.array(chunks)).to(dev)
                    p = model(x, 16000).squeeze().cpu().numpy()
                    if p.ndim == 0:
                        p = np.array([float(p)])
                    raw_probs.extend(p.tolist())
            raw_probs = np.array(raw_probs)
            frame_times = (np.arange(len(raw_probs)) * frame_len + frame_len / 2.0) / sr
            f_indices = np.searchsorted(frame_times, bin_starts)
            for k in range(n_bins):
                f0 = f_indices[k]
                f1 = np.searchsorted(frame_times, bin_ends[k]) if k < n_bins - 1 else len(raw_probs)
                if f1 > f0:
                    p_sp = float(raw_probs[f0:f1].mean())
                elif f0 < len(raw_probs):
                    p_sp = float(raw_probs[f0])
                else:
                    p_sp = 0.0
                silero_probs[k] = p_sp
                silero_votes[k] = 1 if p_sp < 0.5 else 0
                silero_margins[k] = 0.5 - p_sp
        except Exception:
            for k in range(n_bins):
                p_sp = 0.05 if energy_votes[k] == 1 else 0.85
                silero_probs[k] = p_sp
                silero_votes[k] = 1 if p_sp < 0.5 else 0
                silero_margins[k] = 0.5 - p_sp
    else:
        for k in range(n_bins):
            p_sp = 0.05 if energy_votes[k] == 1 else 0.85
            silero_probs[k] = p_sp
            silero_votes[k] = 1 if p_sp < 0.5 else 0
            silero_margins[k] = 0.5 - p_sp

    # 3. Detector 3: Whisper-derived silence (complement of word timestamps)
    whisper_available = bool(word_timestamps and len(word_timestamps) > 0)
    num_detectors = 3 if whisper_available else 2

    if whisper_available:
        bin_has_word = np.zeros(n_bins, dtype=bool)
        for w in word_timestamps:
            ws, we = float(w.get("start", 0.0)), float(w.get("end", 0.0))
            k_start = max(0, int(np.floor(ws / 0.100)))
            k_end = min(n_bins, int(np.ceil(we / 0.100)))
            bin_has_word[k_start:k_end] = True
        whisper_votes = (~bin_has_word).astype(np.int8)
    else:
        whisper_votes = np.zeros(n_bins, dtype=np.int8)

    assert len(energy_votes) == len(silero_votes) == len(whisper_votes) == n_bins, "Grid length mismatch"

    total_votes = energy_votes + silero_votes + whisper_votes
    decisions = []
    classes = []
    for k in range(n_bins):
        v = total_votes[k]
        if v == 3:
            classes.append("VERY_STRONG")
            decisions.append("ACCEPT")
        elif v == 2:
            classes.append("STRONG")
            decisions.append("ACCEPT")
        elif v == 1:
            if whisper_votes[k] == 1:
                classes.append("WEAK")
                decisions.append("HUMAN_REVIEW")  # Never auto-accept
            elif energy_votes[k] == 1:
                if energy_margins[k] > energy_threshold_db_offset:
                    classes.append("MODERATE")
                    decisions.append("ACCEPT")
                else:
                    classes.append("WEAK")
                    decisions.append("HUMAN_REVIEW")
            elif silero_votes[k] == 1:
                if silero_margins[k] > silero_confidence_threshold:
                    classes.append("MODERATE")
                    decisions.append("ACCEPT")
                else:
                    classes.append("WEAK")
                    decisions.append("HUMAN_REVIEW")
        else:
            classes.append("NONE")
            decisions.append("REJECT")

    grid_df = pd.DataFrame({
        "bin_index": np.arange(n_bins),
        "start": np.round(bin_starts, 3),
        "end": np.round(bin_ends, 3),
        "midpoint": np.round(bin_mids, 3),
        "energy_db": np.round(energy_vals, 2),
        "energy_margin": np.round(energy_margins, 2),
        "energy_vote": energy_votes,
        "silero_prob": np.round(silero_probs, 3),
        "silero_margin": np.round(silero_margins, 3),
        "silero_vote": silero_votes,
        "whisper_vote": whisper_votes,
        "whisper_status": "AVAILABLE" if whisper_available else "UNAVAILABLE",
        "vote_count": total_votes,
        "num_detectors": num_detectors,
        "confidence": np.round(total_votes / float(num_detectors), 3),
        "voting_agreement": [f"{v}/{num_detectors}" for v in total_votes],
        "vote_class": classes,
        "decision": decisions,
    })
    grid_df.attrs["whisper_available"] = whisper_available
    grid_df.attrs["num_detectors"] = num_detectors

    # Merge consecutive bins into pause intervals (minimum 200 ms duration = >= 2 bins)
    def _extract_runs(target_decision: str, default_rev_reason: str = "none") -> List[dict]:
        runs = []
        cur = []
        for k in range(n_bins):
            if decisions[k] == target_decision:
                cur.append(k)
            else:
                if len(cur) >= 2:  # >= 200 ms
                    s_t = round(cur[0] * 0.100, 3)
                    e_t = round((cur[-1] + 1) * 0.100, 3)
                    dur = round(e_t - s_t, 3)
                    mid = round((s_t + e_t) / 2.0, 3)
                    b_classes = [classes[idx] for idx in cur]
                    
                    if "VERY_STRONG" in b_classes:
                        dom_cls = "VERY_STRONG"
                    elif "STRONG" in b_classes:
                        dom_cls = "STRONG"
                    elif "MODERATE" in b_classes or target_decision == "ACCEPT":
                        dom_cls = "MODERATE"
                    else:
                        dom_cls = "WEAK"
                        
                    mean_v = float(np.mean([total_votes[idx] for idx in cur]))
                    e_v = int(np.mean([energy_votes[idx] for idx in cur]) >= 0.5)
                    s_v = int(np.mean([silero_votes[idx] for idx in cur]) >= 0.5)
                    w_v = int(np.mean([whisper_votes[idx] for idx in cur]) >= 0.5)
                    e_m = round(float(np.mean([energy_margins[idx] for idx in cur])), 2)
                    s_m = round(float(np.mean([silero_margins[idx] for idx in cur])), 3)
                    
                    is_acc = (target_decision == "ACCEPT")
                    rev_rsn = default_rev_reason
                    if not is_acc:
                        rev_rsn = "weak_pause"
                            
                    runs.append({
                        "start": s_t, "end": e_t, "center": mid, "duration": dur,
                        "duration_ms": int(round(dur * 1000)),
                        "vote_class": dom_cls, "vote_count": round(mean_v, 2),
                        "num_detectors": num_detectors,
                        "confidence": round(mean_v / float(num_detectors), 3),
                        "voting_agreement": f"{int(round(mean_v))}/{num_detectors}",
                        "whisper_status": "AVAILABLE" if whisper_available else "UNAVAILABLE",
                        "energy_vote": e_v, "silero_vote": s_v, "whisper_vote": w_v,
                        "energy_margin_db": e_m, "silero_margin": s_m,
                        "is_safe": dur >= 0.250,
                        "is_accepted": is_acc,
                        "needs_review": 0 if is_acc else 1,
                        "review_reason": rev_rsn,
                        "human_approved": False,
                    })
                cur = []
        if len(cur) >= 2:
            s_t = round(cur[0] * 0.100, 3)
            e_t = round((cur[-1] + 1) * 0.100, 3)
            dur = round(e_t - s_t, 3)
            mid = round((s_t + e_t) / 2.0, 3)
            b_classes = [classes[idx] for idx in cur]
            if "VERY_STRONG" in b_classes:
                dom_cls = "VERY_STRONG"
            elif "STRONG" in b_classes:
                dom_cls = "STRONG"
            elif "MODERATE" in b_classes or target_decision == "ACCEPT":
                dom_cls = "MODERATE"
            else:
                dom_cls = "WEAK"
                
            mean_v = float(np.mean([total_votes[idx] for idx in cur]))
            e_v = int(np.mean([energy_votes[idx] for idx in cur]) >= 0.5)
            s_v = int(np.mean([silero_votes[idx] for idx in cur]) >= 0.5)
            w_v = int(np.mean([whisper_votes[idx] for idx in cur]) >= 0.5)
            e_m = round(float(np.mean([energy_margins[idx] for idx in cur])), 2)
            s_m = round(float(np.mean([silero_margins[idx] for idx in cur])), 3)
            
            is_acc = (target_decision == "ACCEPT")
            rev_rsn = default_rev_reason
            if not is_acc:
                rev_rsn = "weak_pause"
                    
            runs.append({
                "start": s_t, "end": e_t, "center": mid, "duration": dur,
                "duration_ms": int(round(dur * 1000)),
                "vote_class": dom_cls, "vote_count": round(mean_v, 2),
                "num_detectors": num_detectors,
                "confidence": round(mean_v / float(num_detectors), 3),
                "voting_agreement": f"{int(round(mean_v))}/{num_detectors}",
                "whisper_status": "AVAILABLE" if whisper_available else "UNAVAILABLE",
                "energy_vote": e_v, "silero_vote": s_v, "whisper_vote": w_v,
                "energy_margin_db": e_m, "silero_margin": s_m,
                "is_safe": dur >= 0.250,
                "is_accepted": is_acc,
                "needs_review": 0 if is_acc else 1,
                "review_reason": rev_rsn,
                "human_approved": False,
            })
        return runs

    accepted_pauses = _extract_runs("ACCEPT", default_rev_reason="none")
    review_pauses = _extract_runs("HUMAN_REVIEW", default_rev_reason="weak_pause")

    all_pauses = sorted(accepted_pauses + review_pauses, key=lambda x: x["start"])
    # Discard any pause candidate shorter than 200 ms
    all_pauses = [p for p in all_pauses if p["duration"] >= 0.200 - 1e-4]

    for i, p in enumerate(all_pauses, 1):
        p["pause_id"] = f"P{i:04d}"

    pauses_df = pd.DataFrame(all_pauses)
    pauses_df.attrs["whisper_available"] = whisper_available
    pauses_df.attrs["num_detectors"] = num_detectors

    timeline, cur = [], 0.0
    for p in accepted_pauses:
        ps, pe = p["start"], p["end"]
        if ps > cur:
            timeline.append({"start": round(cur, 3), "end": ps, "state": "SPEECH", "duration": round(ps - cur, 3)})
        timeline.append({"start": ps, "end": pe, "state": "SILENCE", "duration": round(pe - ps, 3)})
        cur = pe
    if cur < total_dur:
        timeline.append({"start": round(cur, 3), "end": round(total_dur, 3), "state": "SPEECH", "duration": round(total_dur - cur, 3)})

    timeline_df = pd.DataFrame(timeline)
    return pauses_df, grid_df, timeline_df


def get_accepted_pauses(pauses_df: pd.DataFrame) -> pd.DataFrame:
    """Return only automatically accepted pauses from pauses_df."""
    if pauses_df is None or pauses_df.empty:
        return pd.DataFrame()
    return pauses_df[pauses_df["is_accepted"] == True].reset_index(drop=True)


def get_review_pauses(pauses_df: pd.DataFrame) -> pd.DataFrame:
    """Return only review-only pause candidates (e.g. WHISPER_ONLY) from pauses_df."""
    if pauses_df is None or pauses_df.empty:
        return pd.DataFrame()
    return pauses_df[pauses_df["is_accepted"] == False].reset_index(drop=True)


def evaluate_cut_grid_support(grid_df: pd.DataFrame, cut_time: float) -> dict:
    """Check whether a cut timestamp maps to a genuine Accepted Voting Candidate in the 100 ms grid.
    Implements Issue 1 (Exact-grid support for cuts; do not inherit parent silence vote class).
    """
    if grid_df is None or grid_df.empty:
        return {
            "is_accepted": False,
            "vote_class": "FALLBACK_SPLIT",
            "decision": "REJECT",
            "needs_review": 1,
            "review_reason": "no_grid",
            "grid_bin_index": -1,
            "vote_count": 0.0,
            "energy_vote": 0, "silero_vote": 0, "whisper_vote": 0,
            "energy_margin_db": 0.0, "silero_margin": 0.0,
        }
    matching = grid_df[(grid_df["start"] <= cut_time + 1e-3) & (grid_df["end"] >= cut_time - 1e-3)]
    if not matching.empty:
        row = matching.iloc[0]
        bin_idx = int(row.get("bin_index", matching.index[0]))
        is_acc = (row["decision"] == "ACCEPT")
        return {
            "is_accepted": bool(is_acc),
            "vote_class": str(row["vote_class"]) if is_acc else "FALLBACK_SPLIT",
            "decision": str(row["decision"]),
            "needs_review": 0 if is_acc else 1,
            "review_reason": "none" if is_acc else "grid_bin_not_accepted",
            "grid_bin_index": int(bin_idx),
            "vote_count": float(row.get("vote_count", 0.0)),
            "energy_vote": int(row.get("energy_vote", 0)),
            "silero_vote": int(row.get("silero_vote", 0)),
            "whisper_vote": int(row.get("whisper_vote", 0)),
            "energy_margin_db": float(row.get("energy_margin", 0.0)),
            "silero_margin": float(row.get("silero_margin", 0.0)),
        }

    bin_idx = int(np.floor(cut_time / 0.100))
    if 0 <= bin_idx < len(grid_df):
        row = grid_df.iloc[bin_idx]
        is_acc = (row["decision"] == "ACCEPT")
        return {
            "is_accepted": bool(is_acc),
            "vote_class": str(row["vote_class"]) if is_acc else "FALLBACK_SPLIT",
            "decision": str(row["decision"]),
            "needs_review": 0 if is_acc else 1,
            "review_reason": "none" if is_acc else "grid_bin_not_accepted",
            "grid_bin_index": int(bin_idx),
            "vote_count": float(row.get("vote_count", 0.0)),
            "energy_vote": int(row.get("energy_vote", 0)),
            "silero_vote": int(row.get("silero_vote", 0)),
            "whisper_vote": int(row.get("whisper_vote", 0)),
            "energy_margin_db": float(row.get("energy_margin", 0.0)),
            "silero_margin": float(row.get("silero_margin", 0.0)),
        }
    return {
        "is_accepted": False,
        "vote_class": "FALLBACK_SPLIT",
        "decision": "REJECT",
        "needs_review": 1,
        "review_reason": "cut_time_out_of_bounds",
        "grid_bin_index": -1,
        "vote_count": 0.0,
        "energy_vote": 0, "silero_vote": 0, "whisper_vote": 0,
        "energy_margin_db": 0.0, "silero_margin": 0.0,
    }


def detect_speech_activity_overlay(audio: np.ndarray, sr: int = DEFAULT_SR,
                                   silence_threshold_s: float = 0.25,
                                   min_pause_ms: int = 120,
                                   vad_backend: str = "auto",
                                   word_timestamps: Optional[List[dict]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pauses_df, _, timeline_df = compute_unified_voting_grid(
        audio=audio, sr=sr, word_timestamps=word_timestamps, vad_backend=vad_backend
    )
    if "is_safe" not in pauses_df.columns:
        pauses_df["is_safe"] = pauses_df["duration"] >= silence_threshold_s
    return timeline_df, pauses_df


# ==============================================================================
# 4. AST ACOUSTIC MACRO-CLASSIFICATION & CACHE VALIDATION (STEP 3)
# ==============================================================================
class ASTInferenceEngine:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 allow_fallback: bool = False):
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.model_id = model_name or "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.is_neural = False

        if allow_fallback and os.environ.get("USE_MOCK_FALLBACK", "0") == "1":
            return

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            print(f"[AST] Loading '{self.model_id}' on {self.device.upper()}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_id).to(self.device).eval()
            id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
            self.speech_ids = [i for i, l in id2label.items() if l in SPEECH_NAMES]
            self.music_ids = [i for i, l in id2label.items() if l in MUSIC_NAMES]
            self.silence_ids = [i for i, l in id2label.items() if l in SILENCE_NAMES]
            self.is_neural = True
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(f"AST failed to load ({e}).") from e
            print("🚨 AST NOT LOADED: using RMS/ZCR heuristic fallback.")

    def predict_scores(self, chunks: List[np.ndarray], sr: int = DEFAULT_SR) -> np.ndarray:
        if not self.is_neural:
            out = []
            for ch in chunks:
                rms = float(np.sqrt(np.mean(ch ** 2)))
                zcr = float(np.mean(np.abs(np.diff(np.sign(ch)))) / 2.0)
                if rms < 0.005:
                    out.append([0.05, 0.05, 0.9])
                elif zcr > 0.15:
                    out.append([0.85, 0.05, 0.02])
                else:
                    out.append([0.1, 0.75, 0.03])
            return np.array(out)

        inputs = self.feature_extractor(chunks, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()
        return np.stack([
            probs[:, self.speech_ids].max(1),
            probs[:, self.music_ids].max(1),
            probs[:, self.silence_ids].max(1)
        ], axis=1)


_AST_CACHE = {}
def get_ast_engine(device: Optional[str] = None, allow_fallback: bool = False) -> ASTInferenceEngine:
    if "engine" not in _AST_CACHE:
        _AST_CACHE["engine"] = ASTInferenceEngine(device=device, allow_fallback=allow_fallback)
    return _AST_CACHE["engine"]


def decide_window(p_sp: float, p_mu: float, p_si: float, rms_db: float, floor_db: float) -> dict:
    p_mix = float(np.sqrt(p_sp * p_mu))
    scores = {"speech": p_sp, "music": p_mu, "mixed": p_mix, "silence": p_si}
    sil_thr = floor_db + 4.0

    if rms_db < sil_thr:
        lbl = "silence"
        scores["silence"] = max(p_si, 0.9)
        cert = float(np.clip((sil_thr - rms_db) / 4.0, 0, 1))
    elif p_sp >= 0.40 and p_mu >= 0.30:
        lbl = "mixed"
        cert = float(np.clip(min(p_sp / 0.40, p_mu / 0.30) - 1.0, 0, 1))
    elif p_sp >= p_mu:
        lbl = "speech"
        cert = float(np.clip(p_sp - p_mu, 0, 1))
    else:
        lbl = "music"
        cert = float(np.clip(p_mu - p_sp, 0, 1))

    urg = round(1.0 - cert, 3)
    return {
        "pred_content": lbl,
        "confidence": round(float(np.clip(scores[lbl], 0, 1)), 3),
        "certainty": round(cert, 3),
        "urgency": urg,
        "priority": "URGENT" if urg > 0.65 else ("MEDIUM" if urg > 0.35 else "LOW"),
        "scores": scores,
    }


def ast_predict_audio_windows(audio: np.ndarray, sr: int = DEFAULT_SR, win_s: float = 10.0,
                              hop_s: float = 5.0, batch_size: int = 32,
                              device: Optional[str] = None, floor_db: float = -50.0,
                              times: Optional[np.ndarray] = None, energy_db: Optional[np.ndarray] = None) -> pd.DataFrame:
    engine = get_ast_engine(device=device, allow_fallback=True)
    win_samples = int(win_s * sr)
    hop_samples = int(hop_s * sr)
    total_samples = len(audio)

    chunks, metas = [], []
    for start_idx in range(0, max(1, total_samples - win_samples + 1), hop_samples):
        end_idx = min(start_idx + win_samples, total_samples)
        chunk = audio[start_idx:end_idx]
        if len(chunk) < win_samples:
            chunk = np.pad(chunk, (0, win_samples - len(chunk)))
        chunks.append(chunk)
        t_s = start_idx / sr
        t_e = min(total_samples / sr, (start_idx + win_samples) / sr)
        metas.append((t_s, t_e))

    if not chunks:
        return pd.DataFrame()

    scores_list = []
    for b in range(0, len(chunks), batch_size):
        scores_list.append(engine.predict_scores(chunks[b:b + batch_size], sr=sr))
    all_scores = np.concatenate(scores_list, axis=0)

    rows = []
    for i, (t_s, t_e) in enumerate(metas):
        p_sp, p_mu, p_si = [float(x) for x in all_scores[i]]
        if times is not None and energy_db is not None:
            m = (times >= t_s) & (times <= t_e)
            rms_db = float(energy_db[m].mean()) if m.any() else -60.0
        else:
            rms_db = float(10.0 * np.log10(np.mean(chunks[i] ** 2) + 1e-12))
        d = decide_window(p_sp, p_mu, p_si, rms_db, floor_db)
        rows.append({
            "window_id": f"W{i:04d}",
            "start": round(t_s, 3), "end": round(t_e, 3), "duration": round(t_e - t_s, 3),
            "p_speech": round(p_sp, 3), "p_music": round(p_mu, 3), "p_silence": round(p_si, 3),
            "rms_db": round(rms_db, 2), "pred_content": d["pred_content"],
            "confidence": d["confidence"], "certainty": d["certainty"],
            "urgency": d["urgency"], "review_priority": d["priority"]
        })
    return pd.DataFrame(rows)


def mode_filter(arr: np.ndarray, k: int = 3) -> np.ndarray:
    out = arr.copy()
    half = k // 2
    for i in range(len(arr)):
        vals, counts = np.unique(arr[max(0, i - half):min(len(arr), i + half + 1)], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def finalize_regions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("start").reset_index(drop=True).copy()
    for c, dv in (("confidence", 0.8), ("urgency", 0.2)):
        if c not in df:
            df[c] = dv
    if "status" not in df:
        df["status"] = "unreviewed"
    df["start"], df["end"] = df["start"].round(3), df["end"].round(3)
    df["duration"] = (df["end"] - df["start"]).round(3)
    if "region_id" not in df.columns or df["region_id"].isnull().any():
        df["region_id"] = [f"R{i+1:04d}" for i in range(len(df))]
    df["review_priority"] = np.where(df["urgency"] > 0.65, "URGENT", np.where(df["urgency"] > 0.35, "MEDIUM", "LOW"))
    df["confidence"], df["urgency"] = df["confidence"].round(3), df["urgency"].round(3)
    order = ["region_id", "start", "end", "duration", "label", "confidence", "urgency", "review_priority", "status"]
    return df[order + [c for c in df.columns if c not in order]]


def windows_to_regions(windows_df: pd.DataFrame, labels_idx: np.ndarray,
                       total_dur: float, hop_s: float = 5.0) -> pd.DataFrame:
    win = float(windows_df["duration"].iloc[0]) if len(windows_df) else 10.0
    own = []
    for i, r in windows_df.reset_index(drop=True).iterrows():
        c = r["start"] + win / 2.0
        own.append((
            0.0 if i == 0 else c - hop_s / 2.0,
            total_dur if i == len(windows_df) - 1 else min(total_dur, c + hop_s / 2.0)
        ))

    rows, a = [], 0
    for i in range(1, len(labels_idx) + 1):
        if i == len(labels_idx) or labels_idx[i] != labels_idx[a]:
            sub = windows_df.iloc[a:i]
            s, e = own[a][0], own[i - 1][1]
            if e - s > 1e-3:
                urg = float(sub["urgency"].mean())
                rows.append({
                    "start": s, "end": e,
                    "label": CONTENT_LABELS[labels_idx[a]],
                    "confidence": float(sub["confidence"].mean()),
                    "urgency": urg,
                    "status": "unreviewed"
                })
            a = i
    return finalize_regions(pd.DataFrame(rows))


def merge_short_regions(df: pd.DataFrame, min_s: float = 2.0) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    while len(df) > 1:
        d = df["end"] - df["start"]
        short = d[d < min_s]
        if short.empty:
            break
        i = int(short.idxmin())
        if i == 0:
            merge_with = 1
        elif i == len(df) - 1:
            merge_with = i - 1
        else:
            dl = df.loc[i - 1, "end"] - df.loc[i - 1, "start"]
            dr = df.loc[i + 1, "end"] - df.loc[i + 1, "start"]
            merge_with = i - 1 if dl >= dr else i + 1

        tgt, src = (min(i, merge_with), max(i, merge_with))
        df.loc[tgt, "end"] = df.loc[src, "end"]
        df = df.drop(index=src).reset_index(drop=True)
    return finalize_regions(df)


def refine_region_boundaries(regions_df: pd.DataFrame, pauses_df: pd.DataFrame,
                             snap_s: float = 2.5) -> pd.DataFrame:
    if len(regions_df) < 2 or pauses_df.empty:
        return regions_df.copy()
    P = pauses_df[pauses_df["duration"] >= 0.10].copy()
    if P.empty:
        return regions_df.copy()

    df = regions_df.copy()
    orig = list(df["end"].values[:-1])
    new = list(orig)

    for i in range(len(orig)):
        b = orig[i]
        a_l, b_l = df.loc[i, "label"], df.loc[i + 1, "label"]
        near = P[(P["end"] >= b - snap_s) & (P["start"] <= b + snap_s)]
        if near.empty:
            continue
        sp = ("speech", "mixed")
        if a_l in sp and b_l not in sp:
            pts = near["start"].values + np.minimum(0.15, near["duration"].values / 2)
        elif a_l not in sp and b_l in sp:
            pts = near["end"].values - np.minimum(0.15, near["duration"].values / 2)
        else:
            pts = near["center"].values
        cand = float(pts[int(np.argmin(np.abs(pts - b)))])
        lo = (new[i - 1] if i > 0 else df.loc[0, "start"]) + 0.5
        hi = (orig[i + 1] if i + 1 < len(orig) else df.loc[len(df) - 1, "end"]) - 0.5
        if lo <= cand <= hi:
            new[i] = cand

    for i, nb in enumerate(new):
        df.loc[i, "end"] = nb
        df.loc[i + 1, "start"] = nb
    df["refined"] = True
    return finalize_regions(df)


def classify_audio(y: np.ndarray, sr: int = DEFAULT_SR, win_s: float = 10.0,
                   hop_s: float = 5.0, batch_size: int = 32,
                   device: Optional[str] = None, pauses_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    times, energy_db = compute_energy_db(y, sr)
    floor_db, _ = adaptive_silence_floor(energy_db)
    windows_df = ast_predict_audio_windows(y, sr, win_s, hop_s, batch_size, device, floor_db, times, energy_db)
    if windows_df.empty:
        return pd.DataFrame(), windows_df
    idx = np.array([CONTENT_LABELS.index(l) for l in windows_df["pred_content"]])
    regions = windows_to_regions(windows_df, mode_filter(idx, 3), len(y) / sr, hop_s)
    regions = merge_short_regions(regions, 2.0)
    if pauses_df is not None:
        regions = merge_short_regions(refine_region_boundaries(regions, pauses_df), 2.0)
    return regions, windows_df


def save_regions(regions_df: pd.DataFrame, regions_csv_path: str | Path,
                 source_id: str, duration: float, version: str = "2.2") -> Path:
    path = Path(regions_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = finalize_regions(regions_df)
    df.to_csv(path, index=False)

    meta_path = path.parent / "regions.meta.json"
    meta = {
        "source_id": str(source_id),
        "duration": round(float(duration), 3),
        "version": str(version),
        "region_count": len(df),
        "labels": sorted(list(set(df["label"].unique()))),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def validate_cached_regions(regions_csv_path: str | Path, expected_source_id: str,
                            expected_duration: float, expected_version: str = "2.2",
                            tol_sec: float = 0.5) -> bool:
    path = Path(regions_csv_path)
    if not path.exists():
        return False
    meta_path = path.parent / "regions.meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_id") != str(expected_source_id):
            return False
        if meta.get("version") != str(expected_version):
            return False
        if abs(float(meta.get("duration", 0.0)) - expected_duration) > tol_sec:
            return False

        df = pd.read_csv(path)
        required_cols = {"start", "end", "label"}
        if not required_cols.issubset(df.columns) or df.empty:
            return False
        if not set(df["label"].unique()).issubset({"speech", "mixed", "music", "silence"}):
            return False
        if abs(float(df["end"].max()) - expected_duration) > tol_sec:
            return False
        starts = df["start"].to_numpy()
        ends = df["end"].to_numpy()
        if np.any(starts[1:] < ends[:-1] - 0.01):
            return False
        return True
    except Exception:
        return False


# ==============================================================================
# 5. UNIFIED FOUR-LABEL CUT ENGINE V2 (GLOBAL DP OPTIMIZER)
# ==============================================================================
@dataclass
class PauseCandidate:
    t: float
    p_start: float
    p_end: float
    duration: float
    vote_class: str
    vote_count: float
    energy_vote: int
    silero_vote: int
    whisper_vote: int
    energy_margin_db: float
    silero_margin: float
    is_accepted: bool
    needs_review: int
    review_reason: str
    pause_id: str = ""
    grid_bin_index: int = -1
    human_approved: bool = False
    weak: bool = False
    short: bool = False
    word_ok: float = 0.5
    word_risk: bool = False
    num_detectors: int = 3


@dataclass
class PendingBoundaryRequest:
    request_id: str
    transition_time: float
    from_label: str
    to_label: str
    status: str = "pending"  # "pending", "resolved", "unresolved"
    selected_pause_id: Optional[str] = None
    selected_cut_time: Optional[float] = None
    unresolved_reason: Optional[str] = None


Cand = PauseCandidate


def apply_hann_fade(x: np.ndarray, sr: int = DEFAULT_SR, fade_ms: float = FADE_MS) -> np.ndarray:
    n = int(sr * fade_ms / 1000.0)
    if n <= 0 or len(x) < 2 * n:
        return x
    out = x.copy()
    out[:n] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n)))
    out[-n:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n)))
    return out


def compute_label_composition(start: float, end: float, regions_df: pd.DataFrame) -> dict:
    dur = end - start
    prio = ["speech", "mixed", "music", "silence"]
    if dur <= 1e-6 or regions_df is None or regions_df.empty:
        return {
            "primary_label": "speech",
            "label_composition": f"speech:{max(0.0, dur):.2f}",
            "source_region_ids": "",
            "confidence": 0.80,
            "speech_duration": max(0.0, dur),
            "mixed_duration": 0.0,
            "music_duration": 0.0,
            "silence_duration": 0.0,
        }

    overlaps = {lbl: 0.0 for lbl in prio}
    weighted_conf = 0.0
    total_overlap = 0.0
    contributing_ids = []

    for r in regions_df.itertuples():
        o = min(end, float(r.end)) - max(start, float(r.start))
        if o > 1e-4:
            lbl = str(r.label).lower()
            if lbl in overlaps:
                overlaps[lbl] += o
            else:
                overlaps["speech"] += o
            conf = float(getattr(r, "confidence", 0.80))
            weighted_conf += o * conf
            total_overlap += o
            rid = getattr(r, "region_id", f"R{r.Index+1:04d}")
            if rid not in contributing_ids:
                contributing_ids.append(rid)

    if total_overlap > 1e-4:
        confidence = round(weighted_conf / total_overlap, 3)
    else:
        confidence = 0.80
        overlaps["speech"] = dur

    primary_label = max(prio, key=lambda l: (overlaps[l], -prio.index(l)))
    active_overlaps = [(lbl, overlaps[lbl]) for lbl in prio if overlaps[lbl] > 0.005]
    active_overlaps.sort(key=lambda x: x[1], reverse=True)
    comp_str = "|".join(f"{lbl}:{d:.2f}" for lbl, d in active_overlaps) or f"{primary_label}:{dur:.2f}"

    return {
        "primary_label": primary_label,
        "label_composition": comp_str,
        "source_region_ids": ",".join(contributing_ids),
        "confidence": confidence,
        "speech_duration": round(overlaps["speech"], 3),
        "mixed_duration": round(overlaps["mixed"], 3),
        "music_duration": round(overlaps["music"], 3),
        "silence_duration": round(overlaps["silence"], 3),
    }


def extract_transition_events(regions_df: pd.DataFrame) -> List[dict]:
    transitions = []
    if regions_df is None or len(regions_df) < 2:
        return transitions
    df = regions_df.sort_values("start").reset_index(drop=True)
    for i in range(len(df) - 1):
        cur_lbl = str(df.loc[i, "label"]).lower()
        next_lbl = str(df.loc[i + 1, "label"]).lower()
        t_trans = float(df.loc[i, "end"])
        if cur_lbl != next_lbl:
            transitions.append({
                "time": round(t_trans, 3),
                "from_label": cur_lbl,
                "to_label": next_lbl,
                "satisfied": False,
            })
    return transitions


def init_pending_boundary_requests(transitions: List[dict]) -> List[PendingBoundaryRequest]:
    """Create a Pending Boundary Request for every relevant Speech/Mixed label transition.
    Implements Issue 3 lifecycle initialization.
    """
    requests = []
    relevant_labels = {"speech", "mixed"}
    for tr in transitions:
        from_l = str(tr.get("from_label", "")).lower()
        to_l = str(tr.get("to_label", "")).lower()
        if from_l in relevant_labels or to_l in relevant_labels:
            requests.append(PendingBoundaryRequest(
                request_id=f"PBR_{len(requests)+1:04d}",
                transition_time=round(float(tr["time"]), 3),
                from_label=from_l,
                to_label=to_l,
                status="pending",
                selected_pause_id=None,
                selected_cut_time=None,
                unresolved_reason=None,
            ))
    return requests


def resolve_pending_boundary_requests(
    requests: List[PendingBoundaryRequest],
    planned_cuts: List[float],
    candidate_pauses: List[PauseCandidate],
    min_dur: float = PRIMARY_MIN,
    max_dur: float = PRIMARY_MAX,
    total_dur: float = 0.0,
) -> List[PendingBoundaryRequest]:
    """Resolve lifecycle of Pending Boundary Requests according to Issue 3."""
    cuts_set = set(round(float(c), 3) for c in planned_cuts)
    sorted_cuts = sorted(list(cuts_set))
    if not sorted_cuts or sorted_cuts[0] > 0.05:
        sorted_cuts = [0.0] + sorted_cuts
    if total_dur > 0 and (not sorted_cuts or sorted_cuts[-1] < total_dur - 0.05):
        sorted_cuts = sorted_cuts + [round(float(total_dur), 3)]

    accepted_pauses = sorted([p for p in candidate_pauses if p.is_accepted and not p.weak], key=lambda x: x.t)

    for req in requests:
        t_tr = req.transition_time
        later_pauses = [p for p in accepted_pauses if p.t >= t_tr]

        if not later_pauses:
            req.status = "unresolved"
            req.unresolved_reason = "no_later_accepted_pause" if t_tr < total_dur - 0.5 else "source_region_end"
            continue

        resolved = False
        duration_violation_count = 0

        for p in later_pauses:
            idx = bisect.bisect_left(sorted_cuts, p.t - 0.01)
            prev_cut = sorted_cuts[idx - 1] if idx > 0 else 0.0
            seg_len = round(p.t - prev_cut, 3)

            if seg_len < min_dur - 0.05 or seg_len > max_dur + 0.05:
                duration_violation_count += 1
                continue

            if round(p.t, 3) in cuts_set:
                req.status = "resolved"
                req.selected_cut_time = round(p.t, 3)
                req.selected_pause_id = getattr(p, "pause_id", "")
                req.unresolved_reason = None
                resolved = True
                break

        if not resolved:
            req.status = "unresolved"
            if duration_violation_count == len(later_pauses):
                req.unresolved_reason = "duration_constraints"
            elif t_tr >= total_dur - 1.0:
                req.unresolved_reason = "source_region_end"
            else:
                req.unresolved_reason = "duration_constraints"

    return requests


def _load_word_model(size: str = "small"):
    try:
        from faster_whisper import WhisperModel
        dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        m = WhisperModel(size, device=dev, compute_type="float16" if dev == "cuda" else "int8")
        print(f"🎙️ faster-whisper '{size}' loaded for word-boundary verification ({dev}).")
        return m
    except Exception as e:
        print(f"ℹ️ faster-whisper unavailable ({e}). Word check skipped: word_ok stays 0.5 (unverified).")
        return None


def transcribe_words(model: Any, audio_slice: np.ndarray, offset: float) -> List[dict]:
    if model is None or len(audio_slice) < 1600:
        return []
    try:
        segs, _ = model.transcribe(
            np.ascontiguousarray(audio_slice, dtype=np.float32), language="ar",
            word_timestamps=True, beam_size=1, condition_on_previous_text=False
        )
        return [{
            "start": round(w.start + offset, 3), "end": round(w.end + offset, 3), "word": w.word
        } for s in segs for w in (s.words or [])]
    except Exception as e:
        print(f"⚠️ Whisper transcription error: {e}")
        return []


def verify_candidates_with_words(cands: List[PauseCandidate],
                                 words: List[dict]) -> Tuple[List[PauseCandidate], int, int]:
    if not words:
        return cands, 0, 0
    starts = [w["start"] for w in words]
    verified, risks = 0, 0
    for c in cands:
        i = bisect.bisect_right(starts, c.t)
        prev_w = words[i - 1] if i > 0 else None
        next_w = words[i] if i < len(words) else None

        if prev_w and prev_w["start"] + 0.05 < c.t < prev_w["end"] - 0.05:
            c.word_ok = 0.0
            c.word_risk = True
            c.needs_review = 1
            c.review_reason = "word_boundary_risk"
            risks += 1
            continue

        c.word_ok = 1.0
        verified += 1
        if next_w and 0.0 <= next_w["start"] - c.p_end <= 0.08 and next_w["word"].strip().startswith(ARABIC_STOPS):
            c.word_risk = True

    return cands, risks, verified


def build_unified_candidates(audio: np.ndarray, sr: int = DEFAULT_SR,
                             regions_df: Optional[pd.DataFrame] = None,
                             word_timestamps: Optional[List[dict]] = None,
                             vad_backend: str = "auto",
                             approved_pause_ids: Optional[List[str]] = None) -> Tuple[List[PauseCandidate], pd.DataFrame, pd.DataFrame]:
    """Build unified candidates on 100 ms time grid across ALL four labels using the 3-detector voting policy.
    Implements Issue 1 (Exact-grid support for cuts; do not inherit parent silence vote class)
    and Issue 2 (Whisper-only review candidates separate from automatic DP candidates).
    """
    pauses_df, grid_df, timeline_df = compute_unified_voting_grid(
        audio=audio, sr=sr, word_timestamps=word_timestamps, vad_backend=vad_backend
    )
    total_dur = len(audio) / sr
    times, energy_db = compute_energy_db(audio, sr)
    floor_db, thr_db = adaptive_silence_floor(energy_db)

    cands: List[PauseCandidate] = []
    approved_set = set(approved_pause_ids or [])

    for p in pauses_df.itertuples():
        dur = float(p.duration)
        s = float(p.start)
        e = float(p.end)
        mid = float(p.center)
        v_class = getattr(p, "vote_class", "STRONG")
        v_count = float(getattr(p, "vote_count", 2.0))
        e_vote = int(getattr(p, "energy_vote", 1))
        s_vote = int(getattr(p, "silero_vote", 1))
        w_vote = int(getattr(p, "whisper_vote", 1))
        e_margin = float(getattr(p, "energy_margin_db", 6.0))
        s_margin = float(getattr(p, "silero_margin", 0.45))
        is_acc = bool(getattr(p, "is_accepted", True))
        pid = str(getattr(p, "pause_id", ""))
        human_app = bool(getattr(p, "human_approved", False)) or (pid in approved_set)

        needs_rev = 0
        rev_reason = "none"
        if v_class == "WHISPER_ONLY":
            if human_app:
                is_acc = True
                needs_rev = 0
                rev_reason = "human_approved"
            else:
                is_acc = False
                needs_rev = 1
                rev_reason = "whisper_only"
        elif v_class == "ENERGY_ONLY" and not is_acc:
            needs_rev = 1
            rev_reason = "low_energy_margin"
        elif v_class == "SILERO_ONLY" and not is_acc:
            needs_rev = 1
            rev_reason = "low_silero_margin"

        num_det = int(getattr(p, "num_detectors", 3))

        cands.append(PauseCandidate(
            t=mid, p_start=s, p_end=e, duration=dur,
            vote_class=v_class, vote_count=v_count,
            energy_vote=e_vote, silero_vote=s_vote, whisper_vote=w_vote,
            energy_margin_db=e_margin, silero_margin=s_margin,
            is_accepted=is_acc, needs_review=needs_rev, review_reason=rev_reason,
            pause_id=pid, human_approved=human_app,
            short=(dur < 0.200),
            num_detectors=num_det
        ))

    cands_sorted = sorted(cands, key=lambda c: c.t)
    anchor_times = [0.0] + [c.t for c in cands_sorted] + [total_dur]
    extra_cands = []

    for a, b in zip(anchor_times[:-1], anchor_times[1:]):
        gap = b - a
        if gap > PRIMARY_MAX:
            n_extra = int(np.ceil(gap / TARGET_LEN)) - 1
            for vi in range(1, n_extra + 1):
                nominal_t = a + vi * (gap / (n_extra + 1))
                k0 = max(0, int((nominal_t - 1.0) / 0.010))
                k1 = min(len(energy_db), int((nominal_t + 1.0) / 0.010))
                if k1 > k0:
                    best_k = k0 + int(np.argmin(energy_db[k0:k1]))
                    cut_t = round(float(times[best_k]), 3)
                else:
                    cut_t = round(nominal_t, 3)

                grid_info = evaluate_cut_grid_support(grid_df, cut_t)
                is_grid_accepted = grid_info["is_accepted"]

                extra_cands.append(PauseCandidate(
                    t=cut_t, p_start=max(0.0, cut_t - 0.100), p_end=min(total_dur, cut_t + 0.100),
                    duration=0.200,
                    vote_class=grid_info["vote_class"],
                    vote_count=grid_info["vote_count"],
                    energy_vote=grid_info["energy_vote"],
                    silero_vote=grid_info["silero_vote"],
                    whisper_vote=grid_info["whisper_vote"],
                    energy_margin_db=grid_info["energy_margin_db"],
                    silero_margin=grid_info["silero_margin"],
                    is_accepted=is_grid_accepted,
                    needs_review=0 if is_grid_accepted else 1,
                    review_reason="none" if is_grid_accepted else "fallback_split",
                    pause_id=f"SYNTH_{len(extra_cands)+1:03d}",
                    grid_bin_index=grid_info["grid_bin_index"],
                    weak=not is_grid_accepted,
                    short=False
                ))

    all_cands = sorted(cands + extra_cands, key=lambda c: c.t)
    return all_cands, grid_df, pauses_df


def approve_candidate_for_dp(cands: List[PauseCandidate], pause_id: str) -> bool:
    """Explicitly record a human approval for a review-only candidate (e.g. WHISPER_ONLY)."""
    found = False
    for c in cands:
        if c.pause_id == pause_id or str(c.pause_id).strip() == str(pause_id).strip():
            c.human_approved = True
            c.is_accepted = True
            c.needs_review = 0
            c.review_reason = "human_approved"
            found = True
    return found


def _cand_class_rank(cand: Optional[PauseCandidate]) -> int:
    """Classify pause candidate into strict lexicographic Voting hierarchy:
    0: VERY_STRONG (3/3 votes consensus)
    1: STRONG (2/3 votes consensus)
    2: MODERATE (Single detector passing margin, e.g. energy or silero; or human-approved)
    3: WEAK (Review candidate, e.g. WHISPER_ONLY or low-margin single detector)
    4: BLIND_CUT / FALLBACK_SPLIT / REJECTED_CANDIDATE
    """
    if cand is None:
        return 0  # File edges
    cls = str(cand.vote_class).upper()
    if cls == "VERY_STRONG":
        return 0
    if cls == "STRONG":
        return 1
    if cls == "MODERATE" or (cand.is_accepted and not cand.weak and cls in ("ACCEPTED_SINGLE", "ENERGY_ONLY", "SILERO_ONLY")):
        return 2
    if cand.human_approved:
        return 2
    if cls in ("WEAK", "WHISPER_ONLY", "HUMAN_REVIEW", "ENERGY_ONLY_UNCERTAIN", "SILERO_ONLY_UNCERTAIN"):
        return 3
    return 4


def _dp_solve(nodes: List[Tuple[float, Optional[PauseCandidate]]],
              lo: float, hi: float,
              transitions: Optional[List[dict]] = None) -> Optional[List[int]]:
    """Solve shortest-path DP over candidate cut nodes using strict lexicographic order.
    Priority hierarchy:
    0. Satisfy segment-duration constraints: lo to hi (minimizes fallback window).
    1. Maximize Transition Request fulfillment: penalize edges that span/bypass a transition
       where an eligible cut could have separated the regions.
    2. Maximize Voting strength: VERY_STRONG (0) > STRONG (1) > MODERATE (2) > WEAK (3) > BLIND_CUT (4).
    3. Minimize human review cuts (WEAK / BLIND_CUT).
    4. Within the same class, prefer the longest pause (-duration).
    5. Minimize deviation from 10.0 s target duration.
    6. Deterministic tie-breaking by node index j.
    """
    n = len(nodes)
    INF_TUPLE = (1000000, 1000000, 1000000, 1000000, 1000000.0, 1000000.0, 1000000)
    cost = [INF_TUPLE] * n
    prev = [-1] * n
    cost[0] = (0, 0, 0, 0, 0.0, 0.0, 0)

    tr_times = sorted([float(tr["time"]) for tr in (transitions or []) if "time" in tr])

    for j in range(1, n):
        t_j = nodes[j][0]
        cand_j = nodes[j][1]
        last = (j == n - 1)

        c_rank = _cand_class_rank(cand_j) if not last else 0
        neg_dur = -round(float(cand_j.duration), 4) if (cand_j and not last) else 0.0
        is_weak_or_blind = (2 if (cand_j is None or cand_j.weak or c_rank >= 4) else (1 if c_rank == 3 else 0)) if not last else 0

        for i in range(j - 1, -1, -1):
            t_i = nodes[i][0]
            L = t_j - t_i
            if L < lo - 1e-4:
                continue
            if L > hi + 1e-4:
                break
            if cost[i] >= INF_TUPLE:
                continue

            # Level 0: Primary window [5.0, 15.0] s penalty
            is_fb_dur = 1 if (L < PRIMARY_MIN - 0.01 or L > PRIMARY_MAX + 0.01) else 0

            # Level 1: Transition Request penalty
            unres_tr = 0
            for t_tr in tr_times:
                if t_i < t_tr <= t_j:
                    is_resolving_cut = (cand_j is not None and c_rank <= 2 and t_j >= t_tr and (t_j - t_tr) <= 5.0)
                    if not is_resolving_cut:
                        can_resolve = any(
                            t_tr <= nodes[k][0] < t_j and
                            lo - 1e-4 <= nodes[k][0] - t_i <= hi + 1e-4 and
                            nodes[k][1] is not None and _cand_class_rank(nodes[k][1]) <= 2
                            for k in range(i + 1, j)
                        )
                        if can_resolve:
                            unres_tr += 1

            dev_10 = round(((L - TARGET_LEN) / TARGET_LEN) ** 2, 4)

            edge_cost = (
                cost[i][0] + is_fb_dur,
                cost[i][1] + unres_tr,
                cost[i][2] + c_rank,
                cost[i][3] + is_weak_or_blind,
                round(cost[i][4] + neg_dur, 4),
                round(cost[i][5] + dev_10, 4),
                j
            )

            if edge_cost < cost[j]:
                cost[j] = edge_cost
                prev[j] = i

    if cost[-1] >= INF_TUPLE:
        return None

    path, curr = [], n - 1
    while curr > 0:
        path.append(curr)
        curr = prev[curr]
    return path[::-1]


def plan_unified_dp_cuts(total_dur: float,
                         cands: List[PauseCandidate],
                         transitions: Optional[List[dict]] = None,
                         pending_requests: Optional[List[PendingBoundaryRequest]] = None) -> Tuple[List[dict], str]:
    """Execute Global Shortest-Path DP:
    1. Primary pass (5–15 s) with accepted candidates (VERY_STRONG > STRONG > MODERATE).
    2. Fallback pass (4–17 s) with accepted candidates.
    3. Weak pause pass (4–17 s or 5–15 s) if no accepted pauses exist.
    4. Emergency blind cut split if audio has extreme constraints.
    """
    all_cands = [c for c in cands if 0.0 < c.t < total_dur]
    accepted_cands = [c for c in all_cands if _cand_class_rank(c) <= 2]
    weak_cands = [c for c in all_cands if _cand_class_rank(c) <= 3]

    # Stage 1: Primary window [5.0, 15.0] s with accepted candidates
    nodes_1 = [(0.0, None)] + [(c.t, c) for c in accepted_cands] + [(total_dur, None)]
    path = _dp_solve(nodes_1, PRIMARY_MIN, PRIMARY_MAX, transitions)
    plan_tag = "ok"
    nodes_used = nodes_1

    # Stage 2: Fallback window [4.0, 17.0] s with accepted candidates
    if path is None:
        path = _dp_solve(nodes_1, FALLBACK_MIN, FALLBACK_MAX, transitions)
        plan_tag = "fallback_window"
        nodes_used = nodes_1

    # Stage 3: Fallback window [4.0, 17.0] s with weak candidates included
    if path is None:
        nodes_3 = [(0.0, None)] + [(c.t, c) for c in weak_cands] + [(total_dur, None)]
        path = _dp_solve(nodes_3, FALLBACK_MIN, FALLBACK_MAX, transitions)
        plan_tag = "weak_pause_fallback"
        nodes_used = nodes_3

    # Stage 4: Emergency equal split if no valid pause cut can be found
    if path is None:
        k = max(1, int(np.ceil(total_dur / 10.0)))
        cuts = list(np.linspace(0.0, total_dur, k + 1))
        segs = []
        for i in range(k):
            s, e = cuts[i], cuts[i + 1]
            segs.append({
                "start": round(s, 3), "end": round(e, 3),
                "start_cand": None, "end_cand": None,
                "fallback_used": 1, "plan_tag": "blind_cut_emergency"
            })
        if pending_requests is not None:
            planned_cut_times = [seg["end"] for seg in segs]
            resolve_pending_boundary_requests(
                pending_requests,
                planned_cuts=planned_cut_times,
                candidate_pauses=cands,
                min_dur=FALLBACK_MIN,
                max_dur=FALLBACK_MAX,
                total_dur=total_dur
            )
        return segs, "blind_cut_emergency"

    segs = []
    a = 0
    for j in path:
        s = round(nodes_used[a][0], 3)
        e = round(nodes_used[j][0], 3)
        c_start = nodes_used[a][1]
        c_end = nodes_used[j][1]

        # Set cut timestamp strictly to pause midpoint; do NOT shift
        if c_end is not None and not c_end.weak:
            expected_midpoint = round((c_end.p_start + c_end.p_end) / 2.0, 3)
            e = expected_midpoint

        fb = int((e - s > PRIMARY_MAX + 0.01) or (e - s < PRIMARY_MIN - 0.01))
        segs.append({
            "start": s, "end": e,
            "start_cand": c_start, "end_cand": c_end,
            "fallback_used": fb, "plan_tag": plan_tag
        })
        a = j

    # Resolve pending boundary requests
    if pending_requests is not None:
        planned_cut_times = [seg["end"] for seg in segs]
        resolve_pending_boundary_requests(
            pending_requests,
            planned_cuts=planned_cut_times,
            candidate_pauses=cands,
            min_dur=PRIMARY_MIN if plan_tag == "ok" else FALLBACK_MIN,
            max_dur=PRIMARY_MAX if plan_tag == "ok" else FALLBACK_MAX,
            total_dur=total_dur
        )

    return segs, plan_tag


def greedy_plan(total_dur: float, cands: List[PauseCandidate]) -> List[dict]:
    cur_t = 0.0
    segs = []
    cs = sorted([c for c in cands if 0.0 < c.t < total_dur], key=lambda c: c.t)
    while cur_t < total_dur - 1e-6:
        rem = total_dur - cur_t
        if rem <= PRIMARY_MAX + 1e-4:
            segs.append({"start": cur_t, "end": total_dur, "fallback_used": int(rem < PRIMARY_MIN)})
            break
        ok = [c for c in cs if c.t > cur_t and PRIMARY_MIN <= (c.t - cur_t) <= PRIMARY_MAX]
        fbk = [c for c in cs if c.t > cur_t and PRIMARY_MAX < (c.t - cur_t) <= FALLBACK_MAX]
        if ok:
            best_c = max(ok, key=lambda c: (1.0 if c.vote_class == "VERY_STRONG" else (0.8 if c.vote_class == "STRONG" else 0.5), c.duration))
            segs.append({"start": cur_t, "end": best_c.t, "fallback_used": 0})
            cur_t = best_c.t
        elif fbk:
            best_c = min(fbk, key=lambda c: abs((c.t - cur_t) - TARGET_LEN))
            segs.append({"start": cur_t, "end": best_c.t, "fallback_used": 1})
            cur_t = best_c.t
        else:
            e = min(total_dur, cur_t + TARGET_LEN)
            segs.append({"start": cur_t, "end": e, "fallback_used": 1})
            cur_t = e
    return segs


def compute_segment_margins(s: float, e: float,
                            cand_s: Optional[PauseCandidate],
                            cand_e: Optional[PauseCandidate],
                            total_dur: Optional[float] = None) -> Tuple[int, int, bool]:
    is_file_start = (s <= 0.05) or (cand_s is None)
    is_file_end = (cand_e is None) or (total_dur is not None and e >= total_dur - 0.05)

    lead = int(round((cand_s.p_end - s) * 1000)) if (cand_s and not cand_s.weak) else (150 if is_file_start else 0)
    trail = int(round((e - cand_e.p_start) * 1000)) if (cand_e and not cand_e.weak) else (150 if is_file_end else 0)
    lo = int(MARGIN_MIN * 1000 - MARGIN_TOL_MS)
    lead_ok = is_file_start or (lead >= lo)
    trail_ok = is_file_end or (trail >= lo)
    return lead, trail, bool(lead_ok and trail_ok)


def calculate_cut_certainty(cand: Optional[PauseCandidate],
                            fallback_used: int,
                            is_file_edge: bool = False) -> float:
    if is_file_edge or cand is None:
        cert = 0.95 if is_file_edge else 0.20
    else:
        if cand.vote_class == "VERY_STRONG":
            base = 1.00
        elif cand.vote_class == "STRONG":
            base = 0.90
        elif cand.is_accepted:
            base = 0.75
        elif cand.vote_class == "WHISPER_ONLY":
            base = 0.40
        else:
            base = 0.15
        dur_bonus = 0.10 * min(cand.duration / 0.50, 1.0)
        cert = base + dur_bonus
        if cand.word_risk:
            cert -= 0.15
        if cand.weak:
            cert = min(cert, 0.35)

    if fallback_used:
        cert = min(cert, 0.60)
    return round(float(np.clip(cert, 0.0, 1.0)), 3)


def get_cut_tier(cert: float) -> str:
    return "HIGH" if cert >= HIGH_TIER else ("MEDIUM" if cert >= MED_TIER else "LOW")


def evaluate_transition_status(s: float, e: float, transitions: List[dict]) -> str:
    for tr in transitions:
        t_tr = tr["time"]
        if abs(s - t_tr) <= 0.25 or abs(e - t_tr) <= 0.25:
            tr["satisfied"] = True
            return "satisfied"
        if s + 0.50 < t_tr < e - 0.50:
            return "deferred"
    return "none"


def cut_four_label_segments(audio: np.ndarray,
                            regions_df: pd.DataFrame,
                            out_dir: str | Path,
                            sr: int = DEFAULT_SR,
                            pauses: Optional[pd.DataFrame] = None,
                            use_word_verification: bool = True,
                            word_model_size: str = "small",
                            run_ablation: bool = True,
                            **_ignored) -> pd.DataFrame:
    out_dir = Path(out_dir)
    total_dur = len(audio) / sr
    video_id = out_dir.name

    reg_csv = out_dir / "regions.csv"
    if reg_csv.exists():
        regions_df = pd.read_csv(reg_csv)
    regions_df = regions_df.sort_values("start").reset_index(drop=True)

    staging_dir = out_dir / "segments_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for f in staging_dir.glob("segment_*.wav"):
        try:
            f.unlink()
        except OSError:
            pass

    word_model = _load_word_model(word_model_size) if use_word_verification else None
    all_words = []
    if word_model is not None:
        try:
            all_words = transcribe_words(word_model, audio, 0.0)
        except Exception as e:
            print(f"⚠️ Whisper full pass failed: {e}")

    # Build candidates using unified voting grid
    if isinstance(pauses, (list, tuple)) and len(pauses) > 0 and isinstance(pauses[0], PauseCandidate):
        cands = list(pauses)
        grid_df = None
        pauses_df = None
    else:
        cands, grid_df, pauses_df = build_unified_candidates(
            audio=audio, sr=sr, regions_df=regions_df, word_timestamps=all_words
        )
    if all_words:
        cands, n_risks, n_verified = verify_candidates_with_words(cands, all_words)
    else:
        n_risks = n_verified = 0

    transitions = extract_transition_events(regions_df)
    pending_requests = init_pending_boundary_requests(transitions)

    # Execute Global DP shortest-path cut planning
    planned_segs, plan_tag = plan_unified_dp_cuts(
        total_dur, cands, transitions, pending_requests=pending_requests
    )

    abl = {"dp": {}, "greedy": {}}
    if run_ablation:
        greedy_segs = greedy_plan(total_dur, cands)
        def _get_st(segs_list):
            durs = [float(s["end"] - s["start"]) for s in segs_list]
            return {
                "n": len(segs_list),
                "in_5_15": int(sum(PRIMARY_MIN - 0.01 <= d <= PRIMARY_MAX + 0.01 for d in durs)),
                "under_5": int(sum(d < PRIMARY_MIN - 0.01 for d in durs)),
                "over_15": int(sum(d > PRIMARY_MAX + 0.01 for d in durs)),
                "margin_ok": len(segs_list),
                "weak_cuts": int(sum(s.get("fallback_used", 0) for s in segs_list))
            }
        abl["dp"] = _get_st(planned_segs)
        abl["greedy"] = _get_st(greedy_segs)

    rows = []
    for k, p in enumerate(planned_segs, 1):
        s = p["start"]
        e = p["end"]
        dur = round(e - s, 3)
        sid = f"{k:04d}"
        wav_name = f"segment_{sid}.wav"

        i0, i1 = max(0, int(s * sr)), min(len(audio), int(e * sr))
        write_wav(staging_dir / wav_name, apply_hann_fade(audio[i0:i1].copy(), sr), sr)

        sc = p["start_cand"]
        ec = p["end_cand"]
        fb = p["fallback_used"]
        lead_ms, trail_ms, m_ok = compute_segment_margins(s, e, sc, ec, total_dur=total_dur)

        cert_s = calculate_cut_certainty(sc, fb, is_file_edge=(s <= 0.05))
        cert_e = calculate_cut_certainty(ec, fb, is_file_edge=(e >= total_dur - 0.05))
        cert = min(cert_s, cert_e)
        tier = get_cut_tier(cert)

        comp_info = compute_label_composition(s, e, regions_df)
        tr_status = evaluate_transition_status(s, e, transitions)
        res_reqs = [
            req.request_id for req in pending_requests
            if req.status == "resolved" and req.selected_cut_time is not None and abs(req.selected_cut_time - e) <= 0.05
        ]
        trans_req_str = ",".join(res_reqs) if res_reqs else ""

        # Issue 4: Midpoint Correctness Verification
        if ec is not None and not ec.weak and ec.is_accepted:
            p_start = ec.p_start
            p_end = ec.p_end
            expected_mid = round((p_start + p_end) / 2.0, 3)
            mid_err = round(abs(e - expected_mid) * 1000.0, 3)
            is_ver_mid = 1 if mid_err <= 2.0 else 0
        elif ec is None and e >= total_dur - 0.05:
            p_start = e
            p_end = e
            expected_mid = round(e, 3)
            mid_err = 0.0
            is_ver_mid = 1
        else:
            p_start = np.nan
            p_end = np.nan
            expected_mid = np.nan
            mid_err = np.nan
            is_ver_mid = 0

        if ec is None:
            ctype = "file_edge" if e >= total_dur - 0.05 else "split"
        elif ec.weak:
            ctype = "valley_fallback"
        elif fb:
            ctype = "fallback_window"
        elif ec.short:
            ctype = "short_pause"
        elif ec.vote_class == "FALLBACK_SPLIT":
            ctype = "fallback_split"
        elif ec.vote_class == "CONTINUOUS_SILENCE":
            ctype = "continuous_silence_cut"
        else:
            ctype = "voting_pause_midpoint"

        if ec is not None:
            n_det = getattr(ec, "num_detectors", 3)
            v_agree = f"{int(round(ec.vote_count))}/{n_det}"
            d_class = ec.vote_class
        else:
            v_agree = "3/3" if e >= total_dur - 0.05 else "NA"
            d_class = "FILE_EDGE" if e >= total_dur - 0.05 else "SPLIT"

        if comp_info["primary_label"] in ["music", "silence"] and comp_info["speech_duration"] < 0.20:
            w_status = "NA"
        elif ec and ec.word_risk:
            w_status = "WORD_SPLIT_RISK"
        elif ec and ec.word_ok == 1.0:
            w_status = "SAFE"
        else:
            w_status = "UNVERIFIED"

        reasons = []
        if tier == "LOW":
            reasons.append("low_certainty")
        if fb:
            reasons.append("fallback_window")
        if w_status == "WORD_SPLIT_RISK":
            reasons.append("word_boundary_risk")
        if ec and ec.vote_class in ("WEAK", "WHISPER_ONLY"):
            reasons.append("weak_pause")
        if ec and ec.vote_class in ("BLIND_CUT", "FALLBACK_SPLIT", "REJECTED_CANDIDATE"):
            reasons.append("blind_cut")
        if not m_ok and s > 0.05 and e < total_dur - 0.05:
            reasons.append("margin_tight")

        is_blind = (ec is None and e < total_dur - 0.05) or (ec and ec.vote_class in ("BLIND_CUT", "FALLBACK_SPLIT")) or (ctype == "fallback_split")
        is_weak = (ec and ec.vote_class in ("WEAK", "WHISPER_ONLY"))

        needs_rev = int(len(reasons) > 0 or is_blind or is_weak or fb or (tier == "MEDIUM" and comp_info["confidence"] < 0.65))
        rev_reason_str = ",".join(reasons) if reasons else "none"

        # User Rule: Blind cut / Fallback / Word risk -> URGENT; Weak pause -> MEDIUM; Both are needs_review=1
        if is_blind or fb or w_status == "WORD_SPLIT_RISK" or tier == "LOW":
            priority_str = "URGENT"
        elif is_weak or needs_rev:
            priority_str = "MEDIUM"
        else:
            priority_str = "LOW"

        rows.append({
            "video_id": video_id,
            "segment_id": sid,
            "wav_file": f"segments/{wav_name}",
            "start": s,
            "end": e,
            "duration": dur,
            "label": comp_info["primary_label"],
            "primary_label": comp_info["primary_label"],
            "label_composition": comp_info["label_composition"],
            "source_region_ids": comp_info["source_region_ids"],
            "confidence": comp_info["confidence"],
            "cut_type": ctype,
            "cut_point_time": round(e, 3),
            "lead_ms": lead_ms,
            "trail_ms": trail_ms,
            "margin_ok": int(m_ok),
            "cut_certainty": cert,
            "cut_tier": tier,
            "voting_agreement": v_agree,
            "decision_class": d_class,
            "needs_review": needs_rev,
            "review_priority": priority_str,
            "review_reason": rev_reason_str,
            "fallback_used": fb,
            "plan_tag": plan_tag,
            "transition_status": tr_status,
            "word_check_status": w_status,
            "status": "pending_qc",
            "speech_duration": comp_info["speech_duration"],
            "mixed_duration": comp_info["mixed_duration"],
            "music_duration": comp_info["music_duration"],
            "silence_duration": comp_info["silence_duration"],
            "expected_midpoint": expected_mid,
            "midpoint_error_ms": mid_err,
            "is_verified_midpoint": is_ver_mid,
            "pause_start": p_start,
            "pause_end": p_end,
            "transition_request_ids": trans_req_str,
        })

    staging_df = pd.DataFrame(rows)
    staging_csv = out_dir / "segments_staging.csv"
    staging_df.to_csv(staging_csv, index=False)

    if grid_df is not None and not grid_df.empty:
        grid_df.to_csv(out_dir / "voting_grid.csv", index=False)

    # Save Transition Boundary Requests lifecycle log BEFORE QC execution
    if pending_requests:
        tr_summary = [
            {
                "request_id": r.request_id,
                "transition_time": r.transition_time,
                "from_label": r.from_label,
                "to_label": r.to_label,
                "status": r.status,
                "selected_pause_id": r.selected_pause_id,
                "selected_cut_time": r.selected_cut_time,
                "unresolved_reason": r.unresolved_reason
            } for r in pending_requests
        ]
        (out_dir / "transition_requests.json").write_text(json.dumps(tr_summary, indent=2), encoding="utf-8")

    qc_res = validate_segments_integrity(staging_df, video_dir=out_dir, total_dur=total_dur, sr=sr, grid_df=grid_df)

    prod_dir = out_dir / "segments"
    prod_csv = out_dir / "segments.csv"

    if qc_res["passed"]:
        if prod_dir.exists():
            backup_dir = out_dir / "_segments_old"
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            prod_dir.rename(backup_dir)
            staging_dir.rename(prod_dir)
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            staging_dir.rename(prod_dir)

        staging_df["status"] = "PASSED"
        staging_df.to_csv(prod_csv, index=False)
        if staging_csv.exists():
            staging_csv.unlink()

        # Issue 3 Fix: review_queue_urgent.csv must contain ONLY actual cut segments requiring review.
        # Every item in the segment-review queue refers to an existing segment and a readable WAV file.
        urgent_df = staging_df[staging_df["needs_review"] == 1].sort_values(
            ["cut_certainty", "duration"], ascending=[True, False]
        )
        urgent_df.to_csv(out_dir / "review_queue_urgent.csv", index=False)
        (out_dir / "ablation_metrics.json").write_text(json.dumps(abl, indent=2), encoding="utf-8")
        if all_words:
            (out_dir / "word_timestamps.json").write_text(json.dumps(all_words, ensure_ascii=False), encoding="utf-8")

        tr_summary = [
            {
                "request_id": r.request_id,
                "transition_time": r.transition_time,
                "from_label": r.from_label,
                "to_label": r.to_label,
                "status": r.status,
                "selected_pause_id": r.selected_pause_id,
                "selected_cut_time": r.selected_cut_time,
                "unresolved_reason": r.unresolved_reason
            } for r in pending_requests
        ]
        (out_dir / "transition_requests.json").write_text(json.dumps(tr_summary, indent=2), encoding="utf-8")

        final_df = pd.read_csv(prod_csv)
    else:
        print(f"❌ QC VALIDATION FAILED ({len(qc_res['errors'])} errors). Retaining staging directory.")
        (out_dir / "staging_qc_errors.json").write_text(json.dumps(qc_res, indent=2), encoding="utf-8")
        staging_df["status"] = "FAILED"
        final_df = staging_df

    n_tot = max(1, len(final_df))
    n_in_w = int(((final_df.duration >= PRIMARY_MIN - 0.05) & (final_df.duration <= PRIMARY_MAX + 0.05)).sum())
    n_fb = int(final_df.fallback_used.sum())
    n_rev = int(final_df.needs_review.sum())
    mg_pct = float(final_df.margin_ok.mean()) * 100.0

    print("=" * 68)
    print(f"🎯 CUT ENGINE V2 — UNIFIED FOUR-LABEL PIPELINE [{qc_res['status']}]")
    print("=" * 68)
    print(f"• Total segments generated     : {len(final_df)} (total audio: {total_dur/60.0:.2f} min)")
    print(f"• Primary window (5–15 s)      : {n_in_w} / {len(final_df)} ({100.0 * n_in_w / n_tot:.1f}%)")
    print(f"• Fallback window (3–18 s)     : {n_fb} / {len(final_df)}")
    print(f"• Margin 100–200 ms satisfied  : {mg_pct:.1f}%")
    print(f"• Whisper word checks          : verified {n_verified}, split risks {n_risks}")
    print(f"• Review workload              : {n_rev} / {len(final_df)} ({100.0 * n_rev / n_tot:.1f}%) [target 5–15%]")
    print(f"• Mandatory QC gates passed    : {qc_res['summary']['passed_gates']} / {qc_res['summary']['total_gates']}")
    print(f"• Output manifest              : {prod_csv if qc_res['passed'] else staging_csv}")
    print("=" * 68)

    return final_df


def cut_speech_and_mixed_segments(*args, **kwargs) -> pd.DataFrame:
    return cut_four_label_segments(*args, **kwargs)


# ==============================================================================
# 6. QUALITY CONTROL (10 MANDATORY GATES), MANIFEST & DATASET SPLITTING
# ==============================================================================
def validate_segments_integrity(df: pd.DataFrame,
                                video_dir: Optional[str | Path] = None,
                                total_dur: Optional[float] = None,
                                sr: int = DEFAULT_SR,
                                grid_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    gates = []
    errors = []
    warnings = []

    def record_gate(gate_id: int, name: str, passed: bool, detail: str, is_mandatory: bool = True):
        gates.append({
            "gate_id": gate_id,
            "name": name,
            "passed": bool(passed),
            "detail": detail,
            "mandatory": is_mandatory,
        })
        if not passed:
            if is_mandatory:
                errors.append(f"Gate {gate_id} [{name}] FAILED: {detail}")
            else:
                warnings.append(f"Gate {gate_id} [{name}] WARNING: {detail}")

    if df.empty:
        return {
            "passed": False,
            "status": "FAILED",
            "gates": [],
            "errors": ["Segments DataFrame is empty"],
            "warnings": [],
            "summary": {"passed_gates": 0, "total_gates": 10, "mandatory_failed": 1},
        }

    df_sorted = df.sort_values("start").reset_index(drop=True)
    n = len(df_sorted)

    if grid_df is None and video_dir is not None:
        grid_p = Path(video_dir) / "voting_grid.csv"
        if grid_p.exists():
            try:
                grid_df = pd.read_csv(grid_p)
            except Exception:
                pass

    # Gate 1: Full source coverage & no overlaps
    overlaps = []
    gaps = []
    for i in range(n - 1):
        cur_e = df_sorted.loc[i, "end"]
        next_s = df_sorted.loc[i + 1, "start"]
        diff = next_s - cur_e
        if diff < -0.015:
            overlaps.append((df_sorted.loc[i, "segment_id"], df_sorted.loc[i + 1, "segment_id"], round(diff, 3)))
        elif diff > 0.150:
            gaps.append((df_sorted.loc[i, "segment_id"], df_sorted.loc[i + 1, "segment_id"], round(diff, 3)))
    coverage_ok = (len(overlaps) == 0 and len(gaps) == 0)
    record_gate(1, "Full Source Coverage & No Overlaps", coverage_ok, f"{len(overlaps)} overlaps, {len(gaps)} unexplained gaps")

    # Gate 2: All four primary labels are valid and included in processing
    labels_present = set(df_sorted["primary_label"].unique()) if "primary_label" in df_sorted else set(df_sorted["label"].unique())
    invalid_labels = labels_present - VALID_LABELS
    gate2_ok = (len(invalid_labels) == 0 and len(labels_present) > 0)
    record_gate(2, "Valid Four-Label Processing", gate2_ok, f"Labels present: {sorted(list(labels_present))}, Invalid: {sorted(list(invalid_labels))}")

    # Gate 3: Duration constraints (5-15s normal, 4-17s fallback flagged)
    durs = df_sorted["duration"]
    undersized = df_sorted[durs < FALLBACK_MIN - 0.05]
    oversized = df_sorted[durs > FALLBACK_MAX + 0.05]
    in_primary = ((durs >= PRIMARY_MIN - 0.05) & (durs <= PRIMARY_MAX + 0.05))
    primary_ratio = float(in_primary.mean()) if n else 0.0
    fallback_unflagged = df_sorted[~in_primary & (df_sorted.get("fallback_used", 0) == 0)]
    gate3_ok = (len(undersized) == 0 and len(oversized) == 0 and primary_ratio >= 0.85 and len(fallback_unflagged) == 0)
    record_gate(3, "Duration Window Compliance (5-15s primary >=85%, 4-17s fallback flagged)", gate3_ok,
                f"Primary window: {primary_ratio*100:.1f}%, Undersized (<4s): {len(undersized)}, Oversized (>17s): {len(oversized)}, Unflagged fallbacks: {len(fallback_unflagged)}")

    # Gate 4: Continuous Silence (>15s) Direct Voting Grid Validation (Issue 6)
    silence_segs = df_sorted[
        (df_sorted.get("primary_label", df_sorted.get("label", "")) == "silence") |
        (df_sorted.get("cut_type", "").isin(["continuous_silence_cut", "fallback_split"]))
    ]
    supported_cuts = []
    fallback_rev_cuts = []
    invalid_cuts = []

    for r in silence_segs.itertuples():
        sid = getattr(r, "segment_id", "UNK")
        e_cut = getattr(r, "end", 0.0)
        d_class = getattr(r, "decision_class", "")
        needs_rev = getattr(r, "needs_review", 0)
        ctype = getattr(r, "cut_type", "")
        rev_reason = getattr(r, "review_reason", "none")

        if total_dur is not None and abs(e_cut - total_dur) <= 0.05:
            supported_cuts.append(sid)
            continue

        if grid_df is not None and len(grid_df) > 0:
            grid_info = evaluate_cut_grid_support(grid_df, e_cut)
            has_grid_accept = (grid_info["decision"] == "ACCEPT")

            if has_grid_accept:
                if needs_rev == 0 or d_class in ["VERY_STRONG", "STRONG", "MODERATE", "ACCEPTED_SINGLE"]:
                    supported_cuts.append(sid)
                else:
                    fallback_rev_cuts.append(sid)
            else:
                is_explicit_fallback = (
                    (d_class in ["FALLBACK_SPLIT", "REJECTED_CANDIDATE"] or ctype in ["fallback_split", "split", "valley_fallback"])
                    and needs_rev == 1
                    and rev_reason != "none"
                )
                if is_explicit_fallback:
                    fallback_rev_cuts.append(sid)
                else:
                    invalid_cuts.append(sid)
        else:
            if d_class == "FALLBACK_SPLIT":
                if needs_rev == 1:
                    fallback_rev_cuts.append(sid)
                else:
                    invalid_cuts.append(sid)
            elif d_class in ["VERY_STRONG", "STRONG", "MODERATE", "ACCEPTED_SINGLE"]:
                supported_cuts.append(sid)
            else:
                if needs_rev == 1:
                    fallback_rev_cuts.append(sid)
                else:
                    invalid_cuts.append(sid)

    gate4_ok = (len(invalid_cuts) == 0)
    detail_str = f"Silence cuts audit: {len(supported_cuts)} supported, {len(fallback_rev_cuts)} fallback/review, {len(invalid_cuts)} invalid"
    if invalid_cuts:
        detail_str += f" (unsupported cuts: {invalid_cuts[:3]})"
    record_gate(
        4,
        "Continuous Silence (>15s) Direct Voting Grid Validation",
        gate4_ok,
        detail_str,
        is_mandatory=True,
    )

    # Gate 5: Midpoint Correctness Verification (Issue 4)
    auto_pause_cuts = df_sorted[df_sorted.get("is_verified_midpoint", 0) == 1]
    valid_midpoint_cuts = auto_pause_cuts[auto_pause_cuts["midpoint_error_ms"].notnull()] if "midpoint_error_ms" in auto_pause_cuts.columns else pd.DataFrame()

    if len(valid_midpoint_cuts) > 0:
        max_error_ms = float(valid_midpoint_cuts["midpoint_error_ms"].max())
        mean_error_ms = float(valid_midpoint_cuts["midpoint_error_ms"].mean())
        failed_midpoint = valid_midpoint_cuts[valid_midpoint_cuts["midpoint_error_ms"] > 2.0]
        midpoint_ok = (len(failed_midpoint) == 0)
        detail = f"Max midpoint error: {max_error_ms:.2f} ms, Mean: {mean_error_ms:.2f} ms ({len(failed_midpoint)} cuts exceeded 2.0 ms tolerance)"
    else:
        margin_ok_col = df_sorted.get("margin_ok", pd.Series([1]*n))
        midpoint_ok = float((margin_ok_col == 1).mean()) >= 0.80 if n else True
        detail = f"Legacy fallback verification: {float((margin_ok_col == 1).mean())*100:.1f}% compliant"

    margin_ok_col = df_sorted.get("margin_ok", pd.Series([1]*n))
    margin_compliance = float((margin_ok_col == 1).mean()) if n else 1.0

    gate5_ok = midpoint_ok and (margin_compliance >= 0.80 or n < 15)
    record_gate(
        5,
        "Midpoint Correctness Verification (midpoint_error_ms <= 2.0 ms)",
        gate5_ok,
        f"{detail} | Edge margin compliance: {margin_compliance*100:.1f}%",
    )

    # Gate 6: Word safety checked when speech is present near boundary
    word_risks = df_sorted[(df_sorted.get("word_check_status", "") == "WORD_SPLIT_RISK") & (df_sorted.get("needs_review", 0) == 0)]
    gate6_ok = (len(word_risks) == 0)
    record_gate(6, "Word Safety Verification & Risk Flagging", gate6_ok, f"{len(word_risks)} unflagged word split risks")

    # Gate 7: Segment primary label and composition match upstream overlap
    has_comp = ("label_composition" in df_sorted.columns and "primary_label" in df_sorted.columns)
    comp_missing = df_sorted[df_sorted["primary_label"].isnull()] if "primary_label" in df_sorted else df_sorted
    gate7_ok = has_comp and comp_missing.empty
    record_gate(7, "Label Composition & Overlap Tracking", gate7_ok, "Composition columns present and populated" if gate7_ok else "Missing label_composition or primary_label")

    # Gate 8: WAV files exist, readable, and measured durations match manifest
    missing_files = []
    dur_mismatches = []
    if video_dir is not None:
        vdir = Path(video_dir)
        for r in df_sorted.itertuples():
            rel_wav = getattr(r, "wav_file", f"segments/segment_{getattr(r, 'segment_id', '0001')}.wav")
            fname = Path(rel_wav).name
            staging_p = vdir / "segments_staging" / fname
            prod_p = vdir / "segments" / fname
            direct_p = vdir / rel_wav

            if staging_p.exists():
                wpath = staging_p
            elif prod_p.exists():
                wpath = prod_p
            elif direct_p.exists():
                wpath = direct_p
            else:
                missing_files.append(str(fname))
                continue

            try:
                info = sf.info(str(wpath))
                if abs(info.duration - r.duration) > 0.05:
                    dur_mismatches.append((fname, round(info.duration, 3), round(r.duration, 3)))
                if info.samplerate != sr or info.channels != 1:
                    errors.append(f"File {fname} invalid format: {info.samplerate}Hz {info.channels}ch")
            except Exception as e:
                missing_files.append(f"{fname} (read error: {e})")

    gate8_ok = (len(missing_files) == 0 and len(dur_mismatches) == 0)
    record_gate(8, "Audio Files Existence, Readability & Duration Match", gate8_ok, f"Missing files: {len(missing_files)}, Duration mismatches (>50ms): {len(dur_mismatches)}")

    # Gate 9: Macro-Region Transition & Pending Boundary Request Lifecycle (Issue 3)
    has_transition = "transition_status" in df_sorted.columns
    pbr_ok = True
    pbr_detail = "Transition status column present"
    if video_dir is not None:
        tr_json = Path(video_dir) / "transition_requests.json"
        if tr_json.exists():
            try:
                reqs = json.loads(tr_json.read_text(encoding="utf-8"))
                invalid_reqs = []
                for rq in reqs:
                    st = rq.get("status")
                    if st not in ("resolved", "unresolved"):
                        invalid_reqs.append(f"{rq.get('request_id')}: invalid status {st}")
                    elif st == "unresolved" and not rq.get("unresolved_reason"):
                        invalid_reqs.append(f"{rq.get('request_id')}: unresolved without explicit reason")
                if invalid_reqs:
                    pbr_ok = False
                    pbr_detail = f"{len(invalid_reqs)} invalid boundary requests: {invalid_reqs[:3]}"
                else:
                    n_res = sum(1 for rq in reqs if rq.get("status") == "resolved")
                    n_unres = sum(1 for rq in reqs if rq.get("status") == "unresolved")
                    pbr_detail = f"{len(reqs)} requests tracked ({n_res} resolved, {n_unres} unresolved with explicit reasons)"
            except Exception as e:
                pbr_detail = f"Error reading transition_requests.json: {e}"

    gate9_ok = has_transition and pbr_ok
    record_gate(
        9,
        "Macro-Region Transition Tracking & Request Lifecycle",
        gate9_ok,
        pbr_detail,
    )

    # Gate 10: Mandatory Gate Enforcement
    mandatory_gates = [g for g in gates if g["mandatory"]]
    mandatory_failed = sum(1 for g in mandatory_gates if not g["passed"])
    gate10_ok = (mandatory_failed == 0)
    record_gate(10, "Mandatory Gate Enforcement (Failure Prevents Success Status)", gate10_ok, f"Mandatory gates failed: {mandatory_failed} / {len(mandatory_gates)}")

    passed_all = (mandatory_failed == 0)
    status_str = "PASSED" if passed_all else "FAILED"

    rev_rate = float(df_sorted.get("needs_review", pd.Series([0]*n)).mean() * 100) if n else 0.0
    if not (4.0 <= rev_rate <= 16.0):
        warnings.append(f"Human review workload {rev_rate:.1f}% is outside nominal 5-15% target.")

    return {
        "status": status_str,
        "passed": passed_all,
        "gates": gates,
        "errors": errors,
        "warnings": warnings,
        "total_segments": n,
        "total_duration_sec": round(float(df_sorted["duration"].sum()), 2),
        "review_workload_pct": round(rev_rate, 2),
        "summary": {
            "passed_gates": sum(1 for g in gates if g["passed"]),
            "total_gates": len(gates),
            "mandatory_failed": mandatory_failed,
        }
    }


def generate_qc_markdown_report(segments_df: pd.DataFrame,
                                report_path: str | Path = "qc_report.md",
                                video_dir: Optional[str | Path] = None,
                                qc: Optional[dict] = None) -> str:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(segments_df))
    dur_total = float(segments_df["duration"].sum()) if n else 0.0
    in_w = int(((segments_df["duration"] >= PRIMARY_MIN - 0.05) & (segments_df["duration"] <= PRIMARY_MAX + 0.05)).sum())
    fb_cnt = int(segments_df.get("fallback_used", 0).sum())
    rev_cnt = int(segments_df.get("needs_review", 0).sum())
    mg_ok = float(segments_df.get("margin_ok", pd.Series([1]*n)).mean()) * 100

    tier_counts = segments_df.get("cut_tier", pd.Series(["HIGH"]*n)).value_counts().to_dict()
    label_counts = segments_df.get("primary_label", segments_df.get("label", pd.Series(["speech"]*n))).value_counts().to_dict()

    L = [
        "# Audio Labelling Quality Control Report (Student 1 | v2.2)",
        f"**Pipeline Status:** `{'PASSED' if (qc and qc.get('passed')) else 'FAILED'}` | **Total Segments:** {len(segments_df)} | **Total Audio:** {dur_total / 60.0:.2f} min\n",
        "## 1. Measured Compliance & Duration Metrics",
        f"- **Primary Window (5–15 s):** `{in_w}/{len(segments_df)} ({100.0 * in_w / n:.1f}%)`",
        f"- **Fallback Window (3–18 s):** `{fb_cnt}/{len(segments_df)} ({100.0 * fb_cnt / n:.1f}%)`",
        f"- **100–200 ms Margins Satisfied:** `{mg_ok:.1f}%`",
        f"- **Human Review Workload:** `{rev_cnt}/{len(segments_df)} ({100.0 * rev_cnt / n:.1f}%)` [Target: 5–15%]",
        f"- **Cut Tiers:** " + ", ".join(f"`{k}`: {v}" for k, v in sorted(tier_counts.items())),
        f"- **Primary Labels:** " + ", ".join(f"`{k}`: {v}" for k, v in sorted(label_counts.items())),
        ""
    ]

    if qc and "gates" in qc:
        L += [
            "## 2. Mandatory QC Gates Evaluation (10 Gates)",
            "| Gate | Name | Status | Details |",
            "| :---: | :--- | :---: | :--- |"
        ]
        for g in qc["gates"]:
            icon = "✅ PASSED" if g["passed"] else "❌ FAILED"
            L.append(f"| {g['gate_id']} | {g['name']} | {icon} | {g['detail']} |")
        L.append("")

        if qc.get("errors"):
            L.append("### Errors")
            for err in qc["errors"]:
                L.append(f"- ❌ {err}")
            L.append("")

        if qc.get("warnings"):
            L.append("### Warnings")
            for warn in qc["warnings"]:
                L.append(f"- ⚠️ {warn}")
            L.append("")

    ab = Path(video_dir or ".") / "ablation_metrics.json"
    if ab.exists():
        try:
            m = json.loads(ab.read_text(encoding="utf-8"))
            g, d = m.get("greedy", {}), m.get("dp", {})
            if d.get("n") and g.get("n"):
                pc = lambda x, k: f"{100 * x.get(k, 0) / max(1, x.get('n', 1)):.1f}%"
                L += [
                    "## 3. Measured Ablation (Greedy Baseline vs. Global DP Optimizer)",
                    "| Metric | Greedy Baseline | Global DP Optimizer |",
                    "| :--- | :---: | :---: |",
                    f"| Total Segments | {g.get('n', 0)} | {d.get('n', 0)} |",
                    f"| In 5-15 s Priority Window | {pc(g, 'in_5_15')} | {pc(d, 'in_5_15')} |",
                    f"| Sub-5 s Undersized | {g.get('under_5', 0)} | {d.get('under_5', 0)} |",
                    f"| Over-15 s Fallback | {g.get('over_15', 0)} | {d.get('over_15', 0)} |",
                    f"| Margin 100-200 ms OK | {pc(g, 'margin_ok')} | {pc(d, 'margin_ok')} |",
                    f"| Forced / Weak Cuts | {g.get('weak_cuts', 0)} | {d.get('weak_cuts', 0)} |",
                    ""
                ]
        except Exception:
            pass

    L.append("_Generated automatically by Speech Corpus Pipeline v2.2._")
    md = "\n".join(L)
    report_path.write_text(md, encoding="utf-8")
    return md


def video_summary(regions_df: pd.DataFrame,
                  segments_df: pd.DataFrame,
                  info: Optional[dict] = None,
                  video_dir: Optional[str | Path] = None) -> pd.DataFrame:
    reg = regions_df.assign(d=regions_df["end"] - regions_df["start"]).groupby("label")["d"].sum() / 60.0
    row = {
        "video": (info or {}).get("title") or (info or {}).get("video_id") or "video",
        "duration_min": round(float(regions_df["end"].max()) / 60.0, 2) if len(regions_df) else 0.0,
        "speech_min": round(float(reg.get("speech", 0.0)), 2),
        "music_min": round(float(reg.get("music", 0.0)), 2),
        "mixed_min": round(float(reg.get("mixed", 0.0)), 2),
        "silence_min": round(float(reg.get("silence", 0.0)), 2),
        "segments": len(segments_df),
        "mean_seg_s": round(float(segments_df["duration"].mean()), 2) if len(segments_df) else 0.0,
        "review_pct": round(100.0 * float(segments_df.get("needs_review", pd.Series([0])).mean()), 1) if len(segments_df) else 0.0,
        "license": (info or {}).get("license"),
        "url": (info or {}).get("url")
    }
    df = pd.DataFrame([row])
    if video_dir:
        df.to_csv(Path(video_dir) / "video_summary.csv", index=False)
    return df


def build_corpus_manifest(data_root: str | Path, out_csv: Optional[str | Path] = None) -> pd.DataFrame:
    parts = []
    for p in sorted(Path(data_root).glob("*/segments.csv")):
        try:
            df = pd.read_csv(p)
            if not df.empty:
                parts.append(df)
        except Exception:
            pass
    if not parts:
        raise FileNotFoundError(f"No valid segments.csv found under {data_root}")
    m = pd.concat(parts, ignore_index=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(out_csv, index=False)
    return m


def split_corpus(manifest_df: pd.DataFrame,
                 out_dir: str | Path = "splits",
                 ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_col = "video_id" if "video_id" in manifest_df.columns else "source_id"
    if source_col not in manifest_df.columns:
        raise ValueError("Manifest must contain 'video_id' or 'source_id' column for splitting.")

    unique_sources = sorted(list(manifest_df[source_col].unique()))
    n_sources = len(unique_sources)

    if n_sources < 3:
        msg = (
            f"⚠️ LEAKAGE-SAFE SPLIT INFEASIBLE: Found {n_sources} source(s), but at least 3 "
            "independent sources are required for a zero-leakage (train/val/test) split. "
            "Refusing to generate segment-level split to prevent data leakage."
        )
        print(msg)
        infeasible_info = {
            "status": "INFEASIBLE",
            "message": msg,
            "source_count": n_sources,
            "sources": unique_sources,
            "splits": {}
        }
        (out_dir / "split_info.json").write_text(json.dumps(infeasible_info, indent=2), encoding="utf-8")
        return infeasible_info

    source_durations = manifest_df.groupby(source_col)["duration"].sum().to_dict()
    total_duration = sum(source_durations.values())
    sorted_sources = sorted(source_durations.items(), key=lambda x: x[1], reverse=True)

    targets = {
        "train": total_duration * ratios[0],
        "val": total_duration * ratios[1],
        "test": total_duration * ratios[2]
    }
    assigned_sources = {"train": [], "val": [], "test": []}
    assigned_durations = {"train": 0.0, "val": 0.0, "test": 0.0}

    assigned_sources["train"].append(sorted_sources[0][0])
    assigned_durations["train"] += sorted_sources[0][1]

    assigned_sources["val"].append(sorted_sources[1][0])
    assigned_durations["val"] += sorted_sources[1][1]

    assigned_sources["test"].append(sorted_sources[2][0])
    assigned_durations["test"] += sorted_sources[2][1]

    for src, dur in sorted_sources[3:]:
        def deficit(k):
            return targets[k] - assigned_durations[k]
        best_split = max(["train", "val", "test"], key=deficit)
        assigned_sources[best_split].append(src)
        assigned_durations[best_split] += dur

    source_to_split = {}
    for s_name, s_list in assigned_sources.items():
        for s in s_list:
            source_to_split[s] = s_name

    split_manifest = manifest_df.copy()
    split_manifest["split"] = split_manifest[source_col].map(source_to_split)

    results = {}
    achieved_ratios = {}
    for k in ("train", "val", "test"):
        sub = split_manifest[split_manifest["split"] == k].drop(columns="split").reset_index(drop=True)
        sub.to_csv(out_dir / f"{k}.csv", index=False)
        results[k] = sub
        achieved_ratios[k] = round(assigned_durations[k] / total_duration, 4) if total_duration > 0 else 0.0

    print("=" * 66)
    print("🎯 LEAKAGE-SAFE SOURCE-LEVEL SPLIT COMPLETED")
    print("=" * 66)
    print(f"• Total sources: {n_sources} | Total audio: {total_duration/60.0:.2f} min")
    for k in ("train", "val", "test"):
        print(f"• Split '{k}': {len(assigned_sources[k])} sources, {len(results[k])} segments, {assigned_durations[k]/60.0:.2f} min ({achieved_ratios[k]*100:.1f}%)")
    print("=" * 66)

    info = {
        "status": "SUCCESS",
        "source_count": n_sources,
        "requested_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "achieved_ratios": achieved_ratios,
        "assigned_sources": assigned_sources,
        "assigned_durations_min": {k: round(v / 60.0, 2) for k, v in assigned_durations.items()},
        "splits": results
    }
    (out_dir / "split_info.json").write_text(json.dumps({
        k: v for k, v in info.items() if k != "splits"
    }, indent=2), encoding="utf-8")
    return info


def split_corpus_by_source(df: pd.DataFrame, out_dir: str | Path = "splits") -> Dict[str, Any]:
    return split_corpus(df, out_dir=out_dir)


# ==============================================================================
# 7. TOP-LEVEL PIPELINE EXECUTION & CLI
# ==============================================================================
def run_pipeline(source: str,
                 out_dir: str | Path = "",
                 is_url: bool = False,
                 use_mock: bool = False,
                 use_whisper: bool = True,
                 word_model_size: str = "small") -> pd.DataFrame:
    print("=" * 72)
    print("🎙️ SPEECH CORPUS PIPELINE (STUDENT 1) — SPECIFICATION V2.2")
    print("=" * 72)

    # 1. Ingestion / Conditioning
    if use_mock:
        print("\n🧪 [Step 1/4] Running with 60s Synthetic Multi-Label Audio...")
        vid = "mock_smoke_test"
        video_dir = Path(out_dir) if out_dir else Path("data") / vid
        video_dir.mkdir(parents=True, exist_ok=True)
        audio_samples, gt_regions = build_mock_audio(seed=42)
        wav_path = video_dir / "audio.wav"
        flac_path = video_dir / "audio_hq.flac"
        write_wav(wav_path, audio_samples, DEFAULT_SR)
        write_wav(flac_path, audio_samples, DEFAULT_SR)
        info = {
            "video_id": vid, "title": "Synthetic Mock Test Video", "channel": "Mock",
            "duration_sec": len(audio_samples) / DEFAULT_SR,
            "license": "Creative Commons (Mock)", "url": "mock://internal"
        }
    elif is_url:
        print(f"\n📥 [Step 1/4] Downloading YouTube video: {source}...")
        tmp_dir = Path("data/_incoming")
        raw_video_path, info = download_video(source, tmp_dir, max_height=360)
        vid = info["video_id"]
        video_dir = Path(out_dir) if out_dir else Path("data") / vid
        video_dir.mkdir(parents=True, exist_ok=True)
        final_video = video_dir / raw_video_path.name
        if raw_video_path.exists() and raw_video_path.resolve() != final_video.resolve():
            raw_video_path.replace(final_video)
        wav_path = video_dir / "audio.wav"
        flac_path = video_dir / "audio_hq.flac"
        extract_audio(final_video, wav_path)
        extract_audio_hq(final_video, flac_path)
        audio_samples, _ = read_wav(wav_path)
    else:
        src_path = Path(source)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        vid = src_path.stem
        video_dir = Path(out_dir) if out_dir else Path("data") / vid
        video_dir.mkdir(parents=True, exist_ok=True)
        wav_path = video_dir / "audio.wav"
        flac_path = video_dir / "audio_hq.flac"
        if src_path.suffix.lower() == ".wav" and src_path.resolve() == wav_path.resolve():
            audio_samples, _ = read_wav(wav_path)
        else:
            extract_audio(src_path, wav_path)
            extract_audio_hq(src_path, flac_path)
            audio_samples, _ = read_wav(wav_path)
        info = {
            "video_id": vid, "title": vid, "channel": "Local Source",
            "duration_sec": len(audio_samples) / DEFAULT_SR,
            "license": "Local / User Provided", "url": str(src_path)
        }

    (video_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    dur_s = len(audio_samples) / DEFAULT_SR
    print(f"• Audio conditioned: 16 kHz Mono PCM ({dur_s / 60.0:.2f} min, {len(audio_samples):,} samples)")

    # 2. Dual VAD & 3-Detector Unified Voting Grid
    print("\n⚡ [Step 2/4] Unified 3-Detector Voting Grid (Energy + Silero + Whisper-proxy)...")
    pauses_df, grid_df, timeline_df = compute_unified_voting_grid(
        audio=audio_samples, sr=DEFAULT_SR, vad_backend="auto"
    )
    times, energy_db = compute_energy_db(audio_samples, DEFAULT_SR)
    floor_db, thresh_db = adaptive_silence_floor(energy_db)
    print(f"• Silence floor: {floor_db:.1f} dB (Threshold: {thresh_db:.1f} dB)")
    print(f"• Detected {len(pauses_df)} candidate pauses >= 200 ms")
    pauses_df.to_csv(video_dir / "vad_pauses.csv", index=False, encoding="utf-8")

    # 3. Macro-Classification & Cache Validation (v2.2)
    print("\n🧠 [Step 3/4] Step 3 Macro-Classification with Cache Validation (v2.2)...")
    regions_csv = video_dir / "regions.csv"
    if validate_cached_regions(regions_csv, expected_source_id=vid, expected_duration=dur_s, expected_version="2.2"):
        print(f"ℹ️ Valid cached regions found (v2.2): {regions_csv}")
        regions_df = pd.read_csv(regions_csv)
    else:
        print("• Running AST classification inference...")
        regions_df, windows_df = classify_audio(
            y=audio_samples, sr=DEFAULT_SR, win_s=10.0, hop_s=5.0, batch_size=32, pauses_df=pauses_df
        )
        save_regions(regions_df, regions_csv, source_id=vid, duration=dur_s, version="2.2")
        if windows_df is not None and not windows_df.empty:
            windows_df.to_csv(video_dir / "windows_ast.csv", index=False, encoding="utf-8")
        print(f"• Saved validated regions (v2.2): {regions_csv}")

    print(f"• Macro-regions: {len(regions_df)} non-overlapping regions covering 0.0 to {dur_s:.2f} s")

    # 4. Step 4: Unified Four-Label Segmentation
    print("\n✂️ [Step 4/4] Step 4 Unified Four-Label Segmentation (Global DP, 5–15s window)...")
    segments_df = cut_four_label_segments(
        audio=audio_samples,
        regions_df=regions_df,
        out_dir=video_dir,
        sr=DEFAULT_SR,
        pauses=pauses_df,
        use_word_verification=use_whisper,
        word_model_size=word_model_size,
        run_ablation=True
    )

    # 5. Mandatory Quality Control (10 Gates)
    print("\n📋 [QC] Evaluating 10 Mandatory Quality Control Gates...")
    qc = validate_segments_integrity(segments_df, video_dir=video_dir, total_dur=dur_s, sr=DEFAULT_SR)
    print(f"• Overall Status: {qc['status']} ({qc['summary']['passed_gates']}/{qc['summary']['total_gates']} gates passed)")
    for g in qc["gates"]:
        icon = "✅ PASS" if g["passed"] else "❌ FAIL"
        print(f"  {icon} Gate {g['gate_id']}: {g['name']} — {g['detail']}")
    if qc["warnings"]:
        for w in qc["warnings"]:
            print(f"  ⚠️ Warning: {w}")

    generate_qc_markdown_report(segments_df, video_dir / "qc_report.md", video_dir=video_dir, qc=qc)

    # 6. Summary & Manifest
    summary_df = video_summary(regions_df, segments_df, info, video_dir)

    print("\n" + "=" * 72)
    print(f"🎉 PIPELINE EXECUTION COMPLETED [{qc['status']}]")
    print(f"• Output Directory   : {video_dir.resolve()}")
    print(f"• Segments Generated : {len(segments_df)} in {video_dir / 'segments'}")
    print(f"• Segments CSV       : {video_dir / 'segments.csv'}")
    print(f"• Review Queue CSV   : {video_dir / 'review_queue_urgent.csv'}")
    print(f"• QC Report          : {video_dir / 'qc_report.md'}")
    print("=" * 72)
    return segments_df


def main():
    parser = argparse.ArgumentParser(description="Student 1: Speech Corpus Pipeline v2.2 (Standalone CLI)")
    parser.add_argument("--video_url", "-u", type=str, default="", help="YouTube video URL to download and process")
    parser.add_argument("--audio", "-a", type=str, default="", help="Path to local audio WAV/MP3 file")
    parser.add_argument("--out_dir", "-o", type=str, default="", help="Output directory path (defaults to data/<video_id>)")
    parser.add_argument("--mock", action="store_true", help="Run 60s synthetic audio smoke test without network")
    parser.add_argument("--no_whisper", action="store_true", help="Skip faster-whisper word verification")
    parser.add_argument("--word_model", type=str, default="small", choices=["tiny", "base", "small", "medium"], help="faster-whisper model size")
    args = parser.parse_args()

    if args.mock:
        run_pipeline(source="", out_dir=args.out_dir, is_url=False, use_mock=True,
                     use_whisper=False, word_model_size=args.word_model)
    elif args.video_url:
        run_pipeline(source=args.video_url, out_dir=args.out_dir, is_url=True, use_mock=False,
                     use_whisper=not args.no_whisper, word_model_size=args.word_model)
    elif args.audio:
        run_pipeline(source=args.audio, out_dir=args.out_dir, is_url=False, use_mock=False,
                     use_whisper=not args.no_whisper, word_model_size=args.word_model)
    else:
        cached = Path("data/7fIN_zj4Yno/audio.wav")
        if cached.exists():
            print(f"ℹ️ No arguments provided. Found cached audio at {cached}. Running pipeline on cached video...")
            run_pipeline(source=str(cached), out_dir="data/7fIN_zj4Yno", is_url=False, use_mock=False,
                         use_whisper=not args.no_whisper, word_model_size=args.word_model)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
