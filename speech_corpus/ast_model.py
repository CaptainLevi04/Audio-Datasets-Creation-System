"""Step 3: Audio Spectrogram Transformer (AST) Classifier & Macro-Region Labelling.
Implements:
1. AST pre-trained inference dynamically mapped via model.config.id2label.
2. Energy-assisted silence floor and mixed speech/music probability gating.
3. Overlap-free region tiling via window centers [c - hop/2, c + hop/2].
4. Pause-snapping boundary refinement (snaps coarse edges to VAD pauses).
5. Short-region merging (< 2.0s) and interactive editing operations.
"""

import json
import time
from pathlib import Path
import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from .audio import DEFAULT_SR

CONTENT_LABELS = ["speech", "music", "mixed", "silence"]
AST_WIN_S = 10.0
AST_HOP_S = 5.0
MIXED_SP_THR = 0.40
MIXED_MU_THR = 0.30
SILENCE_DB_ABOVE_FLOOR = 4.0
SILENCE_MAX_DB = -28.0
SILENCE_HARD_DB = -42.0
MIN_REGION_S = 2.0
SNAP_WINDOW_S = 2.5
ALLOW_HEURISTIC_FALLBACK = False

RECITATION_NAMES = {"chant", "mantra"}
SPEECH_NAMES = {
    "speech",
    "male speech, man speaking",
    "female speech, woman speaking",
    "child speech, kid speaking",
    "conversation",
    "narration, monologue",
} | RECITATION_NAMES
MUSIC_NAMES = {"music", "musical instrument", "singing", "song"}
INSTRUMENT_NAMES = {
    "musical instrument",
    "plucked string instrument",
    "keyboard (musical)",
    "rattle (instrument)",
    "brass instrument",
    "bowed string instrument",
    "wind instrument, woodwind instrument",
    "drum",
    "drum machine",
    "bass drum",
    "guitar",
    "piano",
    "synthesizer",
    "electronic music",
    "background music",
    "pop music",
    "rock music",
    "hip hop music",
    "classical music",
    "jazz",
    "dubstep",
}
SILENCE_NAMES = {"silence"}


class ASTInferenceEngine:
    """Pre-trained Audio Spectrogram Transformer. AudioSet class ids are resolved from model.config.id2label."""
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 allow_fallback: bool = False):
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.model_id = model_name or "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.is_neural = False

        if allow_fallback and os.environ.get("USE_MOCK_FALLBACK", "0") == "1":
            print("⚡ Offline Smoke Test: using instantaneous RMS/ZCR heuristic predictions.")
            return

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            print(f"[AST] Loading '{self.model_id}' on {self.device.upper()}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_id).to(self.device).eval()
            id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
            self.speech_ids = [i for i, l in id2label.items() if l in SPEECH_NAMES]
            self.recitation_ids = [i for i, l in id2label.items() if l in RECITATION_NAMES]
            self.music_ids = [i for i, l in id2label.items() if l in MUSIC_NAMES]
            self.instrument_ids = [
                i for i, l in id2label.items() if "instrument" in l or l in INSTRUMENT_NAMES
            ]
            self.silence_ids = [i for i, l in id2label.items() if l in SILENCE_NAMES]
            if not (self.speech_ids and self.music_ids and self.silence_ids):
                raise RuntimeError("Could not find Speech/Music/Silence classes in model.config.id2label")
            print(f"[AST] class ids -> speech:{self.speech_ids} music:{self.music_ids} silence:{self.silence_ids}")
            self.is_neural = True
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(
                    f"AST failed to load ({e}). Refusing to silently use a heuristic. "
                    f"Fix the install/network or set ALLOW_HEURISTIC_FALLBACK=True for a demo only."
                ) from e
            print("🚨🚨 AST NOT LOADED: using RMS/ZCR heuristic. DO NOT USE THESE LABELS FOR REAL CORPUS. 🚨🚨")

    def predict_scores(self, chunks: List[np.ndarray], sr: int = DEFAULT_SR) -> np.ndarray:
        """Returns array (N, 3): [p_speech, p_music, p_silence]."""
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
        p_sp = probs[:, self.speech_ids].max(1)
        p_mu = probs[:, self.music_ids].max(1)
        p_si = probs[:, self.silence_ids].max(1)
        # AudioSet assigns parent "Music" leakage onto unaccompanied vocal recitation ("Chant"/"Mantra").
        if self.recitation_ids:
            p_recite = probs[:, self.recitation_ids].max(1)
            p_inst = probs[:, self.instrument_ids].max(1) if self.instrument_ids else np.zeros_like(p_recite)
            unaccompanied = (p_recite >= 0.25) & (p_inst < 0.10) & (p_recite >= 0.5 * p_mu)
            p_mu = np.where(unaccompanied, np.minimum(p_mu, 0.25), p_mu)
        return np.stack([p_sp, p_mu, p_si], axis=1)


_AST_CACHE = {}


def get_ast_engine(device: Optional[str] = None, allow_fallback: bool = False) -> ASTInferenceEngine:
    """Cache singleton instance of AST inference engine to prevent redundant reload."""
    if "engine" not in _AST_CACHE:
        _AST_CACHE["engine"] = ASTInferenceEngine(device=device, allow_fallback=allow_fallback)
    return _AST_CACHE["engine"]


def decide_window(p_sp: float, p_mu: float, p_si: float, rms_db: float, floor_db: float) -> dict:
    """Rule-based window label. Silence uses guarded acoustic ENERGY; mixed = speech AND music both present."""
    p_mix = float(np.sqrt(p_sp * p_mu))
    scores = {"speech": p_sp, "music": p_mu, "mixed": p_mix, "silence": p_si}
    sil_thr = floor_db + SILENCE_DB_ABOVE_FLOOR
    is_hard_silence = rms_db < SILENCE_HARD_DB
    is_quiet_silence = (
        rms_db < min(sil_thr, SILENCE_MAX_DB)
        and (p_si >= 0.25 or (p_sp < 0.25 and p_mu < 0.25))
    )

    if is_hard_silence or is_quiet_silence:
        lbl = "silence"
        scores["silence"] = max(p_si, 0.9)
        thr_ref = SILENCE_HARD_DB if is_hard_silence else min(sil_thr, SILENCE_MAX_DB)
        cert = float(np.clip((thr_ref - rms_db) / SILENCE_DB_ABOVE_FLOOR, 0, 1))
    elif p_sp >= MIXED_SP_THR and p_mu >= MIXED_MU_THR:
        lbl = "mixed"
        cert = float(np.clip(min(p_sp / MIXED_SP_THR, p_mu / MIXED_MU_THR) - 1.0, 0, 1))
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


def ast_predict_audio_windows(audio: np.ndarray, sr: int = DEFAULT_SR, win_s: float = AST_WIN_S,
                              hop_s: float = AST_HOP_S, batch_size: int = 32,
                              device: Optional[str] = None, floor_db: float = -50.0,
                              times: Optional[np.ndarray] = None, energy_db: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Slice audio into windows and execute batched AST inference."""
    engine = get_ast_engine(device=device, allow_fallback=ALLOW_HEURISTIC_FALLBACK)
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
    """1D sliding mode filter to remove label flicker."""
    out = arr.copy()
    half = k // 2
    for i in range(len(arr)):
        vals, counts = np.unique(arr[max(0, i - half):min(len(arr), i + half + 1)], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def finalize_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, round, and compute normalized duration and priority for regions DataFrame."""
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
                       total_dur: float, hop_s: float = AST_HOP_S) -> pd.DataFrame:
    """Each window OWNS [center - hop/2, center + hop/2].
    Owned slices tile the file, guaranteeing that regions NEVER overlap!
    """
    win = float(windows_df["duration"].iloc[0]) if len(windows_df) else AST_WIN_S
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


def merge_short_regions(df: pd.DataFrame, min_s: float = MIN_REGION_S) -> pd.DataFrame:
    """Merge regions shorter than min_s into neighboring regions with larger overlap."""
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
        if df.loc[src, "status"] == "human_edited":
            df.loc[tgt, "status"] = "human_edited"
        df = df.drop(index=src).reset_index(drop=True)
    return finalize_regions(df)


def refine_region_boundaries(regions_df: pd.DataFrame, pauses_df: pd.DataFrame,
                             snap_s: float = SNAP_WINDOW_S) -> pd.DataFrame:
    """Snap coarse AST region boundaries to the nearest real pause timestamp."""
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


def classify_audio(y: np.ndarray, sr: int = DEFAULT_SR, win_s: float = AST_WIN_S,
                   hop_s: float = AST_HOP_S, batch_size: int = 32,
                   device: Optional[str] = None, pauses_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top-level classification: AST windows -> energy silence -> non-overlapping regions -> snapping."""
    from .segmentation import compute_energy_db, adaptive_silence_floor
    times, energy_db = compute_energy_db(y, sr)
    floor_db, _ = adaptive_silence_floor(energy_db)
    windows_df = ast_predict_audio_windows(y, sr, win_s, hop_s, batch_size, device, floor_db, times, energy_db)
    if windows_df.empty:
        return pd.DataFrame(), windows_df
    idx = np.array([CONTENT_LABELS.index(l) for l in windows_df["pred_content"]])
    regions = windows_to_regions(windows_df, mode_filter(idx, 3), len(y) / sr, hop_s)
    regions = merge_short_regions(regions, MIN_REGION_S)
    if pauses_df is not None:
        regions = merge_short_regions(refine_region_boundaries(regions, pauses_df), MIN_REGION_S)
    return regions, windows_df


# ---- Pure region-editing operations (used by the spectrogram UI editor) ----
def _renorm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("start").reset_index(drop=True).copy()
    df["duration"] = (df["end"] - df["start"]).round(3)
    return df


def region_set_label(df: pd.DataFrame, idx: int, label: str) -> pd.DataFrame:
    df = df.copy()
    df.loc[idx, "label"] = label
    df.loc[idx, "confidence"] = 1.0
    df.loc[idx, "urgency"] = 0.0
    df.loc[idx, "status"] = "human_edited"
    return _renorm(df)


def region_move_borders(df: pd.DataFrame, idx: int, new_start: float, new_end: float) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if idx == 0:
        new_start = float(df.loc[0, "start"])
    if idx == n - 1:
        new_end = float(df.loc[n - 1, "end"])
    lo = df.loc[idx - 1, "start"] + 0.05 if idx > 0 else 0.0
    hi = df.loc[idx + 1, "end"] - 0.05 if idx < n - 1 else df.loc[idx, "end"]
    if not (lo <= new_start < new_end <= hi):
        raise ValueError(f"Invalid borders: need {lo:.2f} <= start < end <= {hi:.2f}")
    df.loc[idx, ["start", "end"]] = [new_start, new_end]
    df.loc[idx, "status"] = "human_edited"
    if idx > 0:
        df.loc[idx - 1, "end"] = new_start
        df.loc[idx - 1, "status"] = "human_edited"
    if idx < n - 1:
        df.loc[idx + 1, "start"] = new_end
        df.loc[idx + 1, "status"] = "human_edited"
    return _renorm(df)


def region_split(df: pd.DataFrame, idx: int, t: float) -> pd.DataFrame:
    r = df.loc[idx]
    if not (r["start"] + 0.05 < t < r["end"] - 0.05):
        raise ValueError("Split time must be inside the region")
    a, b = r.copy(), r.copy()
    a["end"] = t
    b["start"] = t
    a["status"] = b["status"] = "human_edited"
    return _renorm(pd.concat([df.drop(index=idx), pd.DataFrame([a, b])], ignore_index=True))


def region_merge_with_next(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    if idx >= len(df) - 1:
        raise ValueError("No next region to merge with")
    df = df.copy()
    df.loc[idx, "end"] = df.loc[idx + 1, "end"]
    df.loc[idx, "status"] = "human_edited"
    return _renorm(df.drop(index=idx + 1))


def region_delete(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    if len(df) < 2:
        raise ValueError("Cannot delete the only region")
    df = df.copy()
    if idx > 0:
        df.loc[idx - 1, "end"] = df.loc[idx, "end"]
        df.loc[idx - 1, "status"] = "human_edited"
    else:
        df.loc[1, "start"] = df.loc[0, "start"]
        df.loc[1, "status"] = "human_edited"
    return _renorm(df.drop(index=idx))


def region_paint(df: pd.DataFrame, a: float, b: float, label: str) -> pd.DataFrame:
    """Relabel the interval [a, b] splitting overlapping regions."""
    if b - a < 0.05:
        raise ValueError("Interval too short")
    rows = []
    for r in df.to_dict("records"):
        if r["end"] <= a or r["start"] >= b:
            rows.append(r)
            continue
        if r["start"] < a - 0.05:
            l = dict(r)
            l["end"] = a
            rows.append(l)
        if r["end"] > b + 0.05:
            rr = dict(r)
            rr["start"] = b
            rows.append(rr)
    rows.append({
        "start": a, "end": b, "label": label, "confidence": 1.0,
        "urgency": 0.0, "review_priority": "LOW", "status": "human_edited"
    })
    return _renorm(pd.DataFrame(rows))


def save_regions(regions_df: pd.DataFrame, regions_csv_path: str | Path,
                 source_id: str, duration: float, version: str = "2.2") -> Path:
    """Save regions DataFrame and companion regions.meta.json for cache validation."""
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
    """Validate cached regions against source_id, audio duration, version, and integrity."""
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
        if not required_cols.issubset(df.columns):
            return False
        if df.empty:
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


def load_validated_regions(regions_csv_path: str | Path, expected_source_id: str,
                           expected_duration: float, expected_version: str = "2.2",
                           tol_sec: float = 0.5) -> Optional[pd.DataFrame]:
    """Load cached regions if validation passes, else None."""
    if validate_cached_regions(regions_csv_path, expected_source_id, expected_duration, expected_version, tol_sec):
        return pd.read_csv(regions_csv_path)
    return None

