"""Step 5: SNR estimation and clean / noisy labelling of segments.

Two independent estimators are computed for every segment produced by Step 4:

  A. Energy method : speech-dominated frames vs. the quietest (pause) frames.
  B. WADA-SNR      : Waveform Amplitude Distribution Analysis
                     (C. Kim and R. M. Stern, Interspeech 2008).

The two estimates are combined into one value (`snr`), compared to a
threshold (default 25.0 dB) and turned into a final quality label:

        1 -> noisy        0 -> clean

Place this file at  speech_corpus/snr.py
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .audio import DEFAULT_SR, read_wav

# ---------------------------------------------------------------------
# Defaults (all tunable; document the final values in the report)
# ---------------------------------------------------------------------
DEFAULT_THRESHOLD_DB = 25.0     # assignment's starting point
SNR_MIN_DB = -5.0               # clamp range for the energy method
SNR_MAX_DB = 60.0
DISAGREE_DB = 8.0               # flag if |energy - wada| exceeds this
EPS = 1e-10

NOISY = 1
CLEAN = 0

MIDDLE_6_COLS = [
    "model_A_transcript",
    "model_B_transcript",
    "final_transcript",
    "status",
    "review_status",
    "automatic_status",
]

CHANGE_COLS = [
    "changed_by",
    "changed_at",
    "change_log",
]

OUTPUT_COLUMNS = [
    "start",
    "end",
    "label",
    "snr",
    "quality",
    *MIDDLE_6_COLS,
    "segment_id",
    "audio_file",
    *CHANGE_COLS,
]

# ---------------------------------------------------------------------
# WADA lookup table (Kim & Stern, 2008).
# Maps  v = ln(mean|x|) - mean(ln|x|)  ->  SNR in dB, for SNR = -20 ... 100 dB
# in 1 dB steps.  v rises monotonically with SNR (Gaussian noise ~ 0.4097,
# very peaky clean speech ~ 1.63).
# IMPORTANT: verify against the authors' reference implementation or a
# trusted port before reporting numbers (see validate_wada_table()).
# ---------------------------------------------------------------------
_WADA_G = np.array([
    0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006,
    0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272,
    0.41526426, 0.4178192, 0.42077252, 0.42452799, 0.42918886, 0.43510373,
    0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
    0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496,
    0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985,
    0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349,
    1.01047155, 1.0362095, 1.06136425, 1.08579312, 1.1094819, 1.13277995,
    1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313,
    1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935,
    1.3605727, 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619,
    1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
    1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915,
    1.5229097, 1.52855859, 1.53389835, 1.5391211, 1.5439065, 1.54858517,
    1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767,
    1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
    1.5941969, 1.59693155, 1.599446, 1.60185011, 1.60408668, 1.60627134,
    1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905,
    1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143,
    1.62632463, 1.6274027, 1.62842767, 1.62945532, 1.6303307, 1.63128026,
    1.63204102,
])
_WADA_DB = np.arange(-20, -20 + len(_WADA_G), dtype=np.float64)


def validate_wada_table() -> None:
    """Sanity-check the lookup table (length, range, monotonic trend)."""
    if len(_WADA_G) != 121:
        raise ValueError(f"WADA table should have 121 entries (-20..100 dB), got {len(_WADA_G)}")
    if not (0.40 < _WADA_G[0] < 0.42 and 1.6 < _WADA_G[-1] < 1.7):
        raise ValueError("WADA table end-points look wrong.")
    if np.any(np.diff(_WADA_G) < -0.001):
        raise ValueError("WADA table is not (nearly) monotonic.")


# =====================================================================
# Method A: Energy-based SNR
# =====================================================================

def snr_energy(x: np.ndarray, sr: int = DEFAULT_SR,
               frame_ms: float = 25.0, hop_ms: float = 10.0,
               noise_percentile: float = 10.0,
               speech_margin_db: float = 10.0,
               min_speech_frac: float = 0.10,
               clamp: tuple[float, float] = (SNR_MIN_DB, SNR_MAX_DB)) -> float:
    """Speech-frame power versus quietest-frame (pause) power.

    1. Frame power (25 ms / 10 ms).
    2. Noise power  = mean power (linear) of the quietest `noise_percentile` % frames.
    3. Speech power = mean power (linear) of frames more than `speech_margin_db`
       above the noise level (falls back to the loudest 30 % if too few).
    4. SNR = 10 log10((P_speech - P_noise) / P_noise), clamped.
    """
    x = np.asarray(x, dtype=np.float64)
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if len(x) < frame + hop:
        return float("nan")

    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    power = np.mean(frames ** 2, axis=1) + EPS
    if np.max(power) <= 1e-9:          # digital silence
        return float(clamp[0])

    n_noise = max(1, int(np.ceil(len(power) * noise_percentile / 100.0)))
    p_noise = float(np.mean(np.sort(power)[:n_noise]))

    thr = p_noise * (10.0 ** (speech_margin_db / 10.0))
    speech_frames = power[power > thr]
    if len(speech_frames) < max(1, int(min_speech_frac * len(power))):
        k = max(1, int(0.30 * len(power)))
        speech_frames = np.sort(power)[-k:]
    p_speech = float(np.mean(speech_frames))

    ratio = max(p_speech - p_noise, EPS) / p_noise
    snr = 10.0 * np.log10(ratio)
    return float(np.clip(snr, clamp[0], clamp[1]))


# =====================================================================
# Method B: WADA-SNR (Kim & Stern, 2008)
# =====================================================================

def snr_wada(x: np.ndarray,
             clamp: tuple[float, float] = (SNR_MIN_DB, SNR_MAX_DB)) -> float:
    """Blind SNR estimate from the waveform amplitude distribution.

    Clean speech amplitudes are very peaky (Gamma-like); Gaussian noise is not.
    v = ln(mean|x|) - mean(ln|x|) is mapped to dB through the paper's table.
    The result is clamped to the same range as the energy method so that one
    estimator saturating (e.g. 100 dB on studio-clean audio) cannot dominate
    the combined value.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan")
    a = np.abs(x)
    if a.max() < 1e-9:                 # digital silence
        return float(clamp[0])
    a = np.maximum(a, EPS)

    v = np.log(max(a.mean(), EPS)) - np.log(a).mean()
    # The table is increasing in v; np.interp clamps outside the range.
    snr = float(np.interp(v, _WADA_G, _WADA_DB))
    return float(np.clip(snr, clamp[0], clamp[1]))


# =====================================================================
# Combination + decision
# =====================================================================

def combine_snr(snr_e: float, snr_w: float,
                mode: Literal["mean", "wada", "energy", "min"] = "mean") -> float:
    """Combine the two estimates into the single value used for the decision."""
    vals = [v for v in (snr_e, snr_w) if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    if mode == "wada":
        return float(snr_w) if np.isfinite(snr_w) else float(np.mean(vals))
    if mode == "energy":
        return float(snr_e) if np.isfinite(snr_e) else float(np.mean(vals))
    if mode == "min":
        return float(np.min(vals))
    return float(np.mean(vals))


def decide_label(snr_db: float, threshold_db: float = DEFAULT_THRESHOLD_DB) -> int:
    """1 = noisy, 0 = clean.  Unknown SNR (NaN) is treated conservatively as noisy."""
    if snr_db is None or not np.isfinite(snr_db):
        return NOISY
    return NOISY if snr_db < threshold_db else CLEAN


def estimate_segment_snr(samples: np.ndarray, sr: int = DEFAULT_SR,
                         threshold_db: float = DEFAULT_THRESHOLD_DB,
                         combine: Literal["mean", "wada", "energy", "min"] = "mean",
                         disagree_db: float = DISAGREE_DB) -> dict:
    """All Step 5 quantities for one segment (array in, dict out)."""
    e = snr_energy(samples, sr=sr)
    w = snr_wada(samples)
    s = combine_snr(e, w, mode=combine)
    lbl = decide_label(s, threshold_db)
    snr_val = round(s, 2) if np.isfinite(s) else None
    return {
        "snr": snr_val,
        "quality": "noisy" if lbl == NOISY else "clean",
        "snr_energy_db": round(e, 2) if np.isfinite(e) else None,
        "snr_wada_db": round(w, 2) if np.isfinite(w) else None,
        "snr_db": snr_val,
        "snr_disagree": bool(np.isfinite(e) and np.isfinite(w) and abs(e - w) > disagree_db),
        "noisy_label": lbl,
    }


def _resolve_audio_path(base_dir: Path, row) -> Path:
    """Resolve segment path whether the DataFrame uses `wav_file` or `audio_file`."""
    rel = getattr(row, "wav_file", None)
    if pd.isna(rel) or not str(rel).strip():
        rel = getattr(row, "audio_file", None)
    if pd.isna(rel) or not str(rel).strip():
        raise ValueError("Row is missing both 'wav_file' and 'audio_file'.")

    rel_path = Path(str(rel))
    if rel_path.is_absolute():
        return rel_path

    candidate = base_dir / rel_path
    if candidate.exists():
        return candidate

    parent_candidate = base_dir.parent / rel_path
    if parent_candidate.exists():
        return parent_candidate

    return candidate


def _build_review_schema_df(src_df: pd.DataFrame, snr_res: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Build DataFrame with exact columns:
    start, end, label, snr, quality, [6 blank/existing cols], segment_id, audio_file,
    changed_by, changed_at, change_log.
    """
    out = pd.DataFrame(index=src_df.index)

    out["start"] = src_df["start"] if "start" in src_df.columns else ""
    out["end"] = src_df["end"] if "end" in src_df.columns else ""

    if "label" in src_df.columns:
        out["label"] = src_df["label"]
    elif "source_region_label" in src_df.columns:
        out["label"] = src_df["source_region_label"]
    else:
        out["label"] = ""

    out["snr"] = snr_res["snr"]
    out["quality"] = snr_res["quality"]

    # Leave the 6 middle columns blank unless already present in src_df
    for col in MIDDLE_6_COLS:
        out[col] = src_df[col] if col in src_df.columns else ""

    out["segment_id"] = src_df["segment_id"] if "segment_id" in src_df.columns else ""

    if "audio_file" in src_df.columns:
        out["audio_file"] = src_df["audio_file"]
    elif "wav_file" in src_df.columns:
        # Match review CSV format like "data/<video>/segments/xxx.wav" when relative
        def _fmt_audio(val):
            if pd.isna(val) or not str(val).strip():
                return ""
            p = Path(str(val))
            if p.is_absolute():
                return p.as_posix()
            rel_to_base = (base_dir / p)
            try:
                return rel_to_base.relative_to(Path.cwd()).as_posix()
            except ValueError:
                return rel_to_base.as_posix()
        out["audio_file"] = src_df["wav_file"].map(_fmt_audio)
    else:
        out["audio_file"] = ""

    # Leave change columns for the user/annotator unless already populated
    for col in CHANGE_COLS:
        out[col] = src_df[col] if col in src_df.columns else ""

    return out[OUTPUT_COLUMNS]


# =====================================================================
# Step 5 over a whole segments DataFrame
# =====================================================================

def compute_snr_for_segments(segments_df: pd.DataFrame,
                             base_dir: str | Path,
                             threshold_db: float = DEFAULT_THRESHOLD_DB,
                             combine: Literal["mean", "wada", "energy", "min"] = "mean",
                             disagree_db: float = DISAGREE_DB,
                             save_csv: bool = True,
                             verbose: bool = True) -> tuple[pd.DataFrame, list[int]]:
    """Run Step 5 on the output of Step 4 and write `segments.csv` in review format."""
    base_dir = Path(base_dir)
    src_df = segments_df.copy().reset_index(drop=True)

    if "wav_file" not in src_df.columns and "audio_file" not in src_df.columns:
        raise ValueError("segments table has neither 'wav_file' nor 'audio_file' column")

    rows = []
    for r in src_df.itertuples():
        wav_path = _resolve_audio_path(base_dir, r)
        samples, sr = read_wav(wav_path)
        rows.append(estimate_segment_snr(samples, sr, threshold_db, combine, disagree_db))

    snr_res = pd.DataFrame(rows, index=src_df.index)
    df = _build_review_schema_df(src_df, snr_res, base_dir)
    labels = [int(v) for v in snr_res["noisy_label"].tolist()]

    if save_csv:
        df.to_csv(base_dir / "segments.csv", index=False)

    if verbose:
        n_noisy = sum(labels)
        print(f"Step 5: {len(df)} segments | clean={len(df) - n_noisy} noisy={n_noisy} "
              f"| threshold={threshold_db} dB | mean SNR={snr_res['snr'].mean():.1f} dB "
              f"| methods disagree on {int(snr_res['snr_disagree'].sum())}")
    return df, labels


def plot_snr_histogram(df: pd.DataFrame, threshold_db: float = DEFAULT_THRESHOLD_DB,
                       save_path: str | Path | None = None):
    """SNR histogram (required figure for the report)."""
    import matplotlib.pyplot as plt

    snr_series = pd.to_numeric(df["snr"] if "snr" in df.columns else df["snr_db"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(snr_series.dropna(), bins=30, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.axvline(threshold_db, color="red", ls="--", label=f"threshold = {threshold_db:g} dB")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Number of segments")
    ax.set_title("SNR distribution of segments")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


# =====================================================================
# CLI:  python -m speech_corpus.snr data/<video_id> [--threshold 25]
# =====================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Step 5: SNR + clean/noisy labelling.")
    ap.add_argument("video_dir", help="Folder containing segments.csv and segments/")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_DB)
    ap.add_argument("--combine", choices=["mean", "wada", "energy", "min"], default="mean")
    args = ap.parse_args()

    vdir = Path(args.video_dir)
    seg_df = pd.read_csv(vdir / "segments.csv")
    out_df, lbls = compute_snr_for_segments(seg_df, vdir, args.threshold, args.combine)
    plot_snr_histogram(out_df, args.threshold, vdir / "snr_histogram.png")
    print("Labels (1 = noisy, 0 = clean):", lbls)