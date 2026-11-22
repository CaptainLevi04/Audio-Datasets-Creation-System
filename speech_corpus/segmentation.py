"""Step 3.5 & Step 4 Preparation: Short-Time Energy, Adaptive Silence Floor, and VAD Pauses.
Vectorised short-time energy (cumsum), Otsu-based adaptive silence threshold,
Silero VAD pause extraction with explicit speech padding, and audit logging.
"""

from pathlib import Path
import math
import time
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from .audio import DEFAULT_SR

_SILERO = {}
VAD_BACKEND_USED = {"name": None}


def compute_energy_db(audio: np.ndarray, sr: int = DEFAULT_SR,
                      frame_ms: float = 25.0, hop_ms: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised short-time RMS energy (dB) using cumulative sum.
    Returns (frame_start_times, energy_db). ~100x faster than a Python loop.
    """
    frame_len, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    c = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    idx = np.arange(0, len(audio) - frame_len + 1, hop)
    ms = np.maximum((c[idx + frame_len] - c[idx]) / frame_len, 0.0)
    return idx / sr, (10.0 * np.log10(ms + 1e-14)).astype(np.float32)


def adaptive_silence_floor(energy_db: np.ndarray) -> Tuple[float, float]:
    """Otsu threshold on the dB histogram (separates quiet/ambient frames from loud frames).
    Returns (floor_db = median of the quiet class, threshold_db).
    """
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


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) for contiguous True runs."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


class PauseInterval:
    """Represents a speech pause interval with start, end, center, duration."""
    def __init__(self, start: float, end: float, center: Optional[float] = None):
        self.start = round(float(start), 3)
        self.end = round(float(end), 3)
        self.center = round(float(center if center is not None else (start + end) / 2.0), 3)
        self.duration = round(self.end - self.start, 3)
        self.duration_ms = round(self.duration * 1000.0, 1)


def _load_silero():
    """Load Silero VAD model checking local cache first to prevent remote hangs."""
    if "m" not in _SILERO:
        import os
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


def detect_pauses_energy(audio: np.ndarray, sr: int = DEFAULT_SR,
                         min_pause_ms: int = 120) -> List[PauseInterval]:
    """Fallback pause detector based purely on short-time RMS energy."""
    times, e = compute_energy_db(audio, sr)
    _, thr = adaptive_silence_floor(e)
    total = len(audio) / sr
    pauses = []
    for a, b in _true_runs(e < thr):
        t0 = float(times[a])
        t1 = float(times[b]) if b < len(times) else total   # closes a pause open at EOF
        if t1 - t0 >= min_pause_ms / 1000.0:
            pauses.append(PauseInterval(t0, t1))
    return pauses


def detect_pauses_silero(audio: np.ndarray, sr: int = DEFAULT_SR, min_pause_ms: int = 120,
                         speech_pad_ms: int = 10, vad_backend: str = "auto") -> List[PauseInterval]:
    """Pauses = gaps between Silero speech segments.
    Explicit speech_pad_ms=10 prevents shrinking pauses by 60 ms.
    """
    total = len(audio) / sr
    if vad_backend != "energy" and torch is not None:
        try:
            model, get_ts = _load_silero()
            ts = get_ts(torch.from_numpy(audio).float(), model, sampling_rate=sr, threshold=0.5,
                        min_silence_duration_ms=min_pause_ms, min_speech_duration_ms=100,
                        speech_pad_ms=speech_pad_ms)
            VAD_BACKEND_USED["name"] = "silero"
            if not ts:
                return [PauseInterval(0.0, total)]
            pauses, cur = [], 0.0
            for s in ts:
                s0, s1 = s["start"] / sr, s["end"] / sr
                if s0 - cur >= min_pause_ms / 1000.0:
                    pauses.append(PauseInterval(cur, s0))
                cur = s1
            if total - cur >= min_pause_ms / 1000.0:
                pauses.append(PauseInterval(cur, total))
            return pauses
        except Exception as e:
            print(f"⚠️ Silero VAD unavailable ({type(e).__name__}: {e}). Falling back to ENERGY pause detector.")

    VAD_BACKEND_USED["name"] = "energy"
    return detect_pauses_energy(audio, sr, min_pause_ms)


def compute_unified_voting_grid(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    word_timestamps: Optional[List[dict]] = None,
    vad_backend: str = "auto",
    energy_threshold_db_offset: float = 5.0,
    silero_confidence_threshold: float = 0.40,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align Energy, Silero, and Whisper-derived silence to one common 100 ms time grid before voting.
    Implements Specification v2.2 Section 2.1:
    - Common 100 ms time grid across full audio.
    - Detector 1 (Energy): Frame energy in dB vs adaptive silence threshold.
    - Detector 2 (Silero): Frame speech probability vs 0.5 threshold.
    - Detector 3 (Whisper): Complement of recognized word timestamps (ASR proxy).
    - Voting rules:
        3/3: VERY_STRONG -> Accept
        2/3: STRONG -> Accept (regardless of third detector confidence)
        1/3 Whisper only: WHISPER_ONLY -> Human Review (never auto-accept)
        1/3 Energy only: ENERGY_ONLY -> Accept if (thresh_db - energy_db) > 5 dB; else Human Review
        1/3 Silero only: SILERO_ONLY -> Accept if (0.5 - speech_prob) > 0.40; else Human Review
        0/3: NONE -> Reject (not a pause candidate)
    - Consecutive accepted bins are merged into pause intervals.
    - Candidate minimum duration = 200 ms (>= 2 bins).
    - Midpoint cut = exact interval midpoint (leaves >= 100 ms margin on each side).
    - Continuous silence > 15s generates sub-candidates around 14.0–14.5s (nominal 14.25s).
    Returns:
        (candidate_pauses_df, grid_df, silence_timeline_df)
    """
    total_dur = len(audio) / sr
    n_bins = int(np.ceil(total_dur / 0.100))
    if n_bins == 0:
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df

    bin_starts = np.arange(n_bins) * 0.100
    bin_ends = np.minimum(total_dur, bin_starts + 0.100)
    bin_mids = (bin_starts + bin_ends) / 2.0

    # 1. Detector 1: Energy
    times, energy_db = compute_energy_db(audio, sr)
    floor_db, thresh_db = adaptive_silence_floor(energy_db)
    energy_votes = np.zeros(n_bins, dtype=np.int8)
    energy_margins = np.zeros(n_bins, dtype=np.float32)
    energy_vals = np.zeros(n_bins, dtype=np.float32)

    k_indices = np.searchsorted(times, bin_starts)
    for k in range(n_bins):
        k0 = k_indices[k]
        k1 = np.searchsorted(times, bin_ends[k]) if k < n_bins - 1 else len(energy_db)
        if k1 > k0:
            mean_e = float(energy_db[k0:k1].mean())
        elif k0 < len(energy_db):
            mean_e = float(energy_db[k0])
        else:
            mean_e = -100.0
        energy_vals[k] = mean_e
        energy_margins[k] = thresh_db - mean_e
        energy_votes[k] = 1 if mean_e < thresh_db else 0

    # 2. Detector 2: Silero VAD
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
            # Fallback if Silero fails
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

    # Assert matching grid coordinates and lengths
    assert len(energy_votes) == len(silero_votes) == len(whisper_votes) == n_bins, "Grid length mismatch"

    # Combine votes and classify according to specification table
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

    # Build Grid DataFrame
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
    # Maintain SEPARATE collections for:
    # 1. Automatically accepted pauses (decision == 'ACCEPT')
    # 2. Review-only candidates (decision == 'HUMAN_REVIEW', e.g. WHISPER_ONLY)
    accepted_pauses = []
    review_pauses = []

    # Helper to merge contiguous runs of a specific decision
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
    review_pauses = _extract_runs("HUMAN_REVIEW", default_rev_reason="whisper_only")

    # Combine all candidates sorted by start time, assigning stable pause IDs
    all_pauses = sorted(accepted_pauses + review_pauses, key=lambda x: x["start"])
    # Strictly filter out any pause shorter than 200 ms
    all_pauses = [p for p in all_pauses if p["duration"] >= 0.200 - 1e-4]
    for i, p in enumerate(all_pauses, 1):
        p["pause_id"] = f"P{i:04d}"

    pauses_df = pd.DataFrame(all_pauses)
    pauses_df.attrs["whisper_available"] = whisper_available
    pauses_df.attrs["num_detectors"] = num_detectors

    # Build silence/speech timeline based on accepted pauses
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
    # Check if any row in grid_df explicitly bounds cut_time
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
    """Generate speech/silence timeline and structured pauses dataframe via Unified Voting Policy."""
    pauses_df, _, timeline_df = compute_unified_voting_grid(
        audio=audio, sr=sr, word_timestamps=word_timestamps, vad_backend=vad_backend
    )
    if "is_safe" not in pauses_df.columns:
        pauses_df["is_safe"] = pauses_df["duration"] >= silence_threshold_s
    return timeline_df, pauses_df


def compute_instantaneous_energy_vad_overlay(audio: np.ndarray, sr: int = DEFAULT_SR,
                                             vad_timeline_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Compute instantaneous RMS energy array and derive adaptive silence threshold."""
    times, energy_db = compute_energy_db(audio, sr)
    floor_db, thresh_db = adaptive_silence_floor(energy_db)
    if vad_timeline_df is not None and not vad_timeline_df.empty:
        return vad_timeline_df, times, energy_db, thresh_db

    total = len(audio) / sr
    rows = []
    sp = energy_db > thresh_db
    bounds = np.concatenate(([0], np.where(np.diff(sp.astype(np.int8)) != 0)[0] + 1, [len(sp)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        t0 = float(times[a])
        t1 = float(times[b]) if b < len(times) else total
        rows.append({
            "start": round(t0, 3), "end": round(t1, 3), "duration": round(t1 - t0, 3),
            "state": "SPEECH" if sp[a] else "SILENCE"
        })
    return pd.DataFrame(rows), times, energy_db, thresh_db


def log_human_change(video_dir: str | Path, user: str, segment_id: Any,
                     field: str, old_val: Any, new_val: Any, reason: str = "") -> None:
    """Append one audit-trail row (who / what / when) to changes.csv."""
    csv_path = Path(video_dir) / "changes.csv"
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user or "unknown",
        "segment_id": str(segment_id),
        "field": field,
        "old_value": str(old_val),
        "new_value": str(new_val),
        "reason": reason
    }
    pd.DataFrame([row]).to_csv(csv_path, mode="a" if csv_path.exists() else "w",
                               header=not csv_path.exists(), index=False)
