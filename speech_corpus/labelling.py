"""Step 3: Speech / Music / Mixed / Silence Region Labelling.
Computes frame-level class probabilities (via Heuristic acoustic features or PANNs CNN),
applies temporal smoothing and duration filtering, and computes confidence scores.
"""

from typing import Literal
import numpy as np
import pandas as pd
from scipy import signal

LABELS = ["speech", "music", "mixed", "silence"]
COLORS = {
    "speech": "#1f77b4",   # Tab Blue
    "music": "#2ca02c",    # Tab Green
    "mixed": "#ff7f0e",    # Tab Orange
    "silence": "#b0b0b0"   # Light Gray
}
DEFAULT_SR = 16000
DEFAULT_HOP_S = 0.25
DEFAULT_WIN_S = 1.0
DEFAULT_MIN_REGION_S = 1.5


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


# =====================================================================
# 1. Acoustic Features & Heuristic Classification Backend
# =====================================================================

def energy_envelope(y: np.ndarray, sr: int = DEFAULT_SR, frame: int = 400, hop: int = 160) -> np.ndarray:
    """Compute short-time log-energy envelope in dB every 10ms."""
    if len(y) < frame:
        return np.array([0.0], dtype=np.float32)
    _, _, z = signal.stft(y, sr, nperseg=frame, noverlap=frame - hop, boundary=None, padded=False)
    spec_energy = (np.abs(z)**2).sum(axis=0)
    return 10.0 * np.log10(spec_energy + 1e-10)


def long_silence_mask(env: np.ndarray, min_sil_s: float = 1.0, margin_db: float = 8.0) -> np.ndarray:
    """Mask sustained quiet periods (>= min_sil_s) close to the file's noise floor.
    Short inter-word pauses are preserved (handled in Step 4 VAD).
    """
    if len(env) == 0:
        return np.array([], dtype=bool)
    floor = np.percentile(env, 3)
    quiet = env < (floor + margin_db)
    mask = np.zeros_like(quiet, dtype=bool)
    s = 0
    min_frames = int(min_sil_s * 100)  # 100 frames/sec (10ms hop)
    for i in range(1, len(quiet) + 1):
        if i == len(quiet) or quiet[i] != quiet[s]:
            if quiet[s] and (i - s) >= min_frames:
                mask[s:i] = True
            s = i
    return mask


def silence_prob_from_mask(mask: np.ndarray, hop_frames: int, win_frames: int) -> np.ndarray:
    """Calculate p(silence) as the fraction of silent frames inside each analysis window."""
    if len(mask) < win_frames:
        return np.array([1.0 if mask.mean() > 0.5 else 0.0])
    fr = [mask[i:i + win_frames].mean() for i in range(0, len(mask) - win_frames + 1, hop_frames)]
    return sigmoid((np.array(fr) - 0.5) / 0.1)


def heuristic_probs(y: np.ndarray, sr: int = DEFAULT_SR,
                    hop_s: float = DEFAULT_HOP_S, win_s: float = DEFAULT_WIN_S) -> np.ndarray:
    """Compute 4-class probabilities [speech, music, mixed, silence] using acoustic dynamic range.
    
    Speech has high dynamic range (pauses between syllables).
    Music has sustained energy (low dynamic range).
    Mixed has intermediate dynamic range.
    Silence is determined from the sustained silence mask.
    """
    env = energy_envelope(y, sr=sr)
    hf, wf = int(hop_s * 100), int(win_s * 100)
    mask = long_silence_mask(env)
    p_sil = silence_prob_from_mask(mask, hf, wf)

    probs = []
    for k, i in enumerate(range(0, len(env) - wf + 1, hf)):
        window_env = env[i:i + wf]
        active_env = window_env[~mask[i:i + wf]]
        if len(active_env) < 20:
            probs.append([0.0, 0.0, 0.0, 1.0])
            continue

        dyn = np.percentile(active_env, 90) - np.percentile(active_env, 10)
        p_speech_like = float(sigmoid((dyn - 22.0) / 4.0))
        p_not_music = float(sigmoid((dyn - 7.0) / 1.5))
        
        p_speech = p_speech_like
        p_mixed = p_not_music * (1.0 - p_speech_like)
        p_music = 1.0 - p_not_music
        
        active_weight = max(0.0, 1.0 - p_sil[k])
        v = np.array([p_speech, p_music, p_mixed]) * active_weight
        probs.append([v[0], v[1], v[2], p_sil[k]])

    p_arr = np.array(probs, dtype=np.float32) + 1e-9
    return p_arr / p_arr.sum(axis=1, keepdims=True)


# =====================================================================
# 2. PANNs Pre-trained Model Backend (Optional / AudioSet CNN)
# =====================================================================

def panns_probs(y16: np.ndarray, sr: int = DEFAULT_SR, device: str = "cpu",
                hop_s: float = DEFAULT_HOP_S, win_s: float = DEFAULT_WIN_S) -> np.ndarray:
    """Compute class probabilities using AudioSet pre-trained CNN (PANNs inference)."""
    try:
        from panns_inference import SoundEventDetection, labels
    except ImportError:
        raise ImportError(
            "panns_inference is not installed. To use PANNs, run: pip install panns_inference\n"
            "Alternatively, use backend='heuristic' which requires zero external neural models."
        )

    sed = SoundEventDetection(checkpoint_path=None, device=device)
    y32 = signal.resample_poly(y16, 2, 1).astype(np.float32)
    fw = sed.inference(y32[None, :])[0]

    i_sp = labels.index("Speech")
    i_mu = labels.index("Music")

    hf, wf = int(hop_s * 100), int(win_s * 100)
    env = energy_envelope(y16, sr=sr)
    p_sil = silence_prob_from_mask(long_silence_mask(env), hf, wf)

    n_frames = min(len(p_sil), (len(fw) - wf) // hf + 1)
    probs = []
    for k in range(n_frames):
        seg = fw[k * hf : k * hf + wf]
        ps = float(seg[:, i_sp].mean())
        pm = float(seg[:, i_mu].mean())
        v = np.array([ps * (1.0 - pm), pm * (1.0 - ps), ps * pm]) * (1.0 - p_sil[k])
        probs.append([v[0], v[1], v[2], p_sil[k]])

    p_arr = np.array(probs, dtype=np.float32) + 1e-6
    return p_arr / p_arr.sum(axis=1, keepdims=True)


# =====================================================================
# 3. Smoothing, Filtering & Confidence Scores
# =====================================================================

def mode_filter(idx: np.ndarray, k: int = 5) -> np.ndarray:
    """Apply temporal mode (majority voting) filter to suppress spurious label transitions."""
    pad = k // 2
    p = np.pad(idx, pad, mode="edge")
    return np.array([np.bincount(p[i:i + k], minlength=len(LABELS)).argmax() for i in range(len(idx))])


def probs_to_regions(probs: np.ndarray, total_dur: float,
                     hop_s: float = DEFAULT_HOP_S, win_s: float = DEFAULT_WIN_S,
                     min_region_s: float = DEFAULT_MIN_REGION_S,
                     mode_win: int = 5) -> pd.DataFrame:
    """Aggregate frame probabilities into continuous regions with confidence scores.
    
    Args:
        probs: Array of shape (N, 4) with probabilities for [speech, music, mixed, silence].
        total_dur: Total audio duration in seconds.
        hop_s: Analysis hop size in seconds.
        win_s: Analysis window size in seconds.
        min_region_s: Minimum allowable region duration (shorter regions are merged).
        mode_win: Window length for mode filtering.
        
    Returns:
        DataFrame with columns: [start, end, label, confidence, duration]
    """
    if len(probs) == 0:
        return pd.DataFrame(columns=["start", "end", "label", "confidence", "duration"])

    idx = mode_filter(probs.argmax(axis=1), mode_win)

    # Find contiguous runs
    runs = []
    s = 0
    for i in range(1, len(idx) + 1):
        if i == len(idx) or idx[i] != idx[s]:
            runs.append([s, i, idx[s]])
            s = i

    # Iteratively merge runs shorter than min_region_s into the longer neighbor
    min_n = int(round(min_region_s / hop_s))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, (a, b, l) in enumerate(runs):
            if (b - a) < min_n:
                left = runs[j - 1] if j > 0 else None
                right = runs[j + 1] if j < len(runs) - 1 else None
                
                # Pick adjacent neighbor with larger duration
                if left and right:
                    tgt = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
                else:
                    tgt = left or right

                tgt_j = runs.index(tgt)
                runs[tgt_j] = [min(tgt[0], a), max(tgt[1], b), tgt[2]]
                del runs[j]
                changed = True
                break

        # Merge consecutive runs if they now share the same label
        if len(runs) > 0:
            merged = [runs[0]]
            for r in runs[1:]:
                if r[2] == merged[-1][2]:
                    merged[-1][1] = r[1]
                else:
                    merged.append(r)
            runs = merged

    # Map frame indices to physical time (seconds) and compute confidence scores
    rows = []
    for k, (a, b, l) in enumerate(runs):
        start = 0.0 if k == 0 else a * hop_s + win_s / 2.0
        end = total_dur if k == len(runs) - 1 else b * hop_s + win_s / 2.0
        conf = float(probs[a:b, l].mean()) if b > a else 0.5
        dur = end - start
        rows.append({
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "label": LABELS[l],
            "confidence": round(float(np.clip(conf, 0.0, 1.0)), 3),
            "duration": round(float(dur), 3),
        })

    return pd.DataFrame(rows)


def classify_audio(y: np.ndarray, sr: int = DEFAULT_SR,
                   backend: Literal["heuristic", "panns"] = "heuristic",
                   device: str = "cpu") -> tuple[pd.DataFrame, np.ndarray]:
    """Top-level classifier function for Step 3.
    
    Returns:
        (regions_df, frame_probabilities)
    """
    total_dur = len(y) / sr
    if backend == "panns":
        probs = panns_probs(y, sr=sr, device=device)
    else:
        probs = heuristic_probs(y, sr=sr)

    regions = probs_to_regions(probs, total_dur=total_dur)
    return regions, probs


# =====================================================================
# 4. Mock Audio Generator (for testing & verification)
# =====================================================================

def make_speech_synthetic(dur: float, rng: np.random.Generator, sr: int = DEFAULT_SR) -> np.ndarray:
    """Generate synthetic speech-like harmonic signal with vowels and pauses."""
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
    """Generate synthetic music-like chords, kick, and hi-hat."""
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


def build_mock_audio(seed: int = 42, timeline=None, sr: int = DEFAULT_SR) -> tuple[np.ndarray, pd.DataFrame]:
    """Build mock audio signal with ground truth timeline for rapid validation."""
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
        else:  # mixed: speech over music
            x = 0.9 * make_speech_synthetic(d, rng, sr=sr) + 0.35 * make_music_synthetic(d, rng, sr=sr)
        parts.append(x)
        gt_rows.append((round(t, 3), round(t + d, 3), lab, round(d, 3)))
        t += d

    total_samples = sum(len(p) for p in parts)
    y = np.concatenate(parts) + 0.0008 * rng.standard_normal(total_samples)
    gt = pd.DataFrame(gt_rows, columns=["start", "end", "label", "duration"])
    return y.astype(np.float32), gt


# =====================================================================
# 5. Evaluation against Ground Truth
# =====================================================================

def evaluate_ground_truth(pred_df: pd.DataFrame, gt_df: pd.DataFrame, total_dur: float, step: float = 0.01) -> dict:
    """Evaluate predicted regions against ground truth annotations."""
    t = np.arange(0, total_dur, step) + step / 2.0
    p = np.empty(len(t), dtype=object)
    g = np.empty(len(t), dtype=object)

    for r in pred_df.itertuples():
        p[(t >= r.start) & (t < r.end)] = r.label
    for r in gt_df.itertuples():
        g[(t >= r.start) & (t < r.end)] = r.label

    acc = float(np.mean(p == g))
    cm = pd.crosstab(
        pd.Series(g, name="truth"),
        pd.Series(p, name="predicted")
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)
    cm_pct = (cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100).round(1)

    pb = pred_df["start"].values[1:] if len(pred_df) > 1 else np.array([])
    gb = gt_df["start"].values[1:] if len(gt_df) > 1 else np.array([])
    berr = [float(np.min(np.abs(pb - b))) if len(pb) else np.nan for b in gb]

    return {
        "frame_accuracy": round(acc, 4),
        "confusion_matrix_pct": cm_pct,
        "boundary_error_median_sec": round(float(np.nanmedian(berr)), 3) if len(berr) else None,
        "boundary_error_max_sec": round(float(np.nanmax(berr)), 3) if len(berr) else None,
    }
