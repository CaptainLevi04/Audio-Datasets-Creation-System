"""Interactive & Programmatic Cut-Segment Editor for Speech Corpus Pipeline.

Implements Step 8 human-override operations on cut segments (``segments_df`` /
``segments.csv`` / ``segments/*.wav``):
    - **Move**: Adjust start/end boundaries of a segment (with optional
      neighbor synchronization so adjacent segments remain contiguous).
    - **Split**: Split a segment into two segments at a chosen timestamp.
    - **Merge**: Merge a segment with its previous or next neighbor.
    - **Delete**: Remove a segment (or absorb its span into a neighbor).
    - **Add**: Insert a new segment over ``[start, end]`` with a chosen label.

Every edit can atomically synchronize ``segments.csv``, regenerate affected
``segments/*.wav`` audio clips (16 kHz mono 16-bit PCM with 8 ms Hann fade),
and append an audit-trail entry to ``changes.csv``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from IPython.display import Audio, display

from .audio import DEFAULT_SR, write_wav
from .cut_engine_v2 import (
    FALLBACK_MAX,
    FALLBACK_MIN,
    PRIMARY_MAX,
    PRIMARY_MIN,
    apply_hann_fade,
    compute_label_composition,
)
from .segmentation import log_human_change

VALID_SEGMENT_LABELS = ("speech", "music", "mixed", "silence")
DEFAULT_MIN_SEGMENT_DUR = 0.20

SEGMENT_COLORS = {
    "speech": "#2ca02c",
    "music": "#1f77b4",
    "mixed": "#ff7f0e",
    "silence": "#7f7f7f",
}

_SCHEMA_DEFAULTS: Dict[str, Any] = {
    "video_id": "unknown",
    "segment_id": "0001",
    "wav_file": "segments/segment_0001.wav",
    "start": 0.0,
    "end": 0.0,
    "duration": 0.0,
    "label": "speech",
    "primary_label": "speech",
    "label_composition": "speech:0.00",
    "source_region_ids": "",
    "confidence": 1.0,
    "cut_type": "human_edited",
    "cut_point_time": 0.0,
    "lead_ms": 0,
    "trail_ms": 0,
    "margin_ok": 1,
    "cut_certainty": 1.0,
    "cut_tier": "HIGH",
    "voting_agreement": "HUMAN",
    "decision_class": "HUMAN_OVERRIDE",
    "needs_review": 0,
    "review_priority": "LOW",
    "review_reason": "none",
    "fallback_used": 0,
    "plan_tag": "human_edited",
    "transition_status": "none",
    "word_check_status": "UNVERIFIED",
    "status": "human_edited",
    "speech_duration": 0.0,
    "mixed_duration": 0.0,
    "music_duration": 0.0,
    "silence_duration": 0.0,
    "expected_midpoint": np.nan,
    "midpoint_error_ms": np.nan,
    "is_verified_midpoint": 0,
    "pause_start": np.nan,
    "pause_end": np.nan,
    "transition_request_ids": "",
}


def _resolve_row_index(df: pd.DataFrame, idx: Any) -> Any:
    """Resolve DataFrame row label from index or string segment_id."""
    if isinstance(idx, (int, np.integer)) and idx in df.index:
        return idx
    if idx in df.index:
        return idx
    if "segment_id" in df.columns:
        matches = df.index[df["segment_id"].astype(str) == str(idx)].tolist()
        if matches:
            return matches[0]
    raise KeyError(f"Segment identifier {idx!r} not found in DataFrame.")


def _infer_video_id(df: pd.DataFrame, video_id: Optional[str] = None) -> str:
    if video_id:
        return str(video_id)
    if not df.empty and "video_id" in df.columns:
        vals = df["video_id"].dropna().astype(str)
        if not vals.empty and vals.iloc[0].strip():
            return vals.iloc[0]
    return "unknown"


def _compute_comp_dict(
    start: float,
    end: float,
    regions_df: Optional[pd.DataFrame],
    fallback_label: str = "speech",
    fallback_conf: float = 1.0,
) -> Dict[str, Any]:
    """Recompute label composition fields from regions_df or fallback defaults."""
    dur = round(max(0.0, float(end) - float(start)), 3)
    if regions_df is not None and not regions_df.empty:
        comp = compute_label_composition(start, end, regions_df)
        return comp

    lbl = fallback_label if fallback_label in VALID_SEGMENT_LABELS else "speech"
    return {
        "primary_label": lbl,
        "label_composition": f"{lbl}:{dur:.2f}",
        "source_region_ids": "",
        "confidence": round(float(fallback_conf), 3),
        "speech_duration": dur if lbl == "speech" else 0.0,
        "mixed_duration": dur if lbl == "mixed" else 0.0,
        "music_duration": dur if lbl == "music" else 0.0,
        "silence_duration": dur if lbl == "silence" else 0.0,
    }


def _apply_comp_to_row(row: Dict[str, Any], comp: Dict[str, Any]) -> Dict[str, Any]:
    row["primary_label"] = comp["primary_label"]
    row["label"] = comp["primary_label"]
    row["label_composition"] = comp["label_composition"]
    row["source_region_ids"] = comp["source_region_ids"]
    row["confidence"] = comp["confidence"]
    row["speech_duration"] = comp["speech_duration"]
    row["mixed_duration"] = comp["mixed_duration"]
    row["music_duration"] = comp["music_duration"]
    row["silence_duration"] = comp["silence_duration"]
    return row


def _renorm_segments(df: pd.DataFrame, reindex_ids: bool = True) -> pd.DataFrame:
    """Sort segments chronologically, recompute derived timing/QC fields, and assign clean IDs."""
    if df.empty:
        cols = list(df.columns) if len(df.columns) > 0 else list(_SCHEMA_DEFAULTS.keys())
        return pd.DataFrame(columns=cols)

    out = df.sort_values(["start", "end"]).reset_index(drop=True).copy()

    # Keep label and primary_label synchronized
    if "label" not in out.columns and "primary_label" in out.columns:
        out["label"] = out["primary_label"]
    elif "primary_label" not in out.columns and "label" in out.columns:
        out["primary_label"] = out["label"]
    elif "label" not in out.columns and "primary_label" not in out.columns:
        out["label"] = "speech"
        out["primary_label"] = "speech"
    else:
        missing_lbl = out["label"].isna()
        out.loc[missing_lbl, "label"] = out.loc[missing_lbl, "primary_label"]
        missing_prim = out["primary_label"].isna()
        out.loc[missing_prim, "primary_label"] = out.loc[missing_prim, "label"]

    out["start"] = out["start"].astype(float).round(3)
    out["end"] = out["end"].astype(float).round(3)
    out["duration"] = (out["end"] - out["start"]).round(3)
    out["cut_point_time"] = out["end"]

    # Keep fallback_used consistent with 5-15s primary bounds so QC Gate 3 stays valid
    durs = out["duration"]
    in_primary = (durs >= PRIMARY_MIN - 0.05) & (durs <= PRIMARY_MAX + 0.05)
    out["fallback_used"] = np.where(in_primary, 0, 1).astype(int)

    if reindex_ids:
        sids = [f"{k:04d}" for k in range(1, len(out) + 1)]
        out["segment_id"] = sids
        out["wav_file"] = [f"segments/segment_{sid}.wav" for sid in sids]

    return out


# =====================================================================
# Pure Segment Editing Operations (Move, Split, Merge, Delete, Add)
# =====================================================================

def segment_move(
    df: pd.DataFrame,
    idx: Any,
    new_start: float,
    new_end: float,
    sync_neighbors: bool = True,
    regions_df: Optional[pd.DataFrame] = None,
    total_dur: Optional[float] = None,
    min_dur: float = DEFAULT_MIN_SEGMENT_DUR,
) -> pd.DataFrame:
    """Move a segment's start and/or end boundaries.

    When ``sync_neighbors=True`` (default), adjacent touching segments adjust their
    borders seamlessly so no overlaps or gaps are introduced.
    """
    work = df.sort_values(["start", "end"]).reset_index(drop=True).copy()
    row_key = _resolve_row_index(work, idx)
    pos = work.index.get_loc(row_key)

    new_start = round(float(new_start), 3)
    new_end = round(float(new_end), 3)
    if new_start < 0.0:
        raise ValueError(f"new_start ({new_start:.3f}s) cannot be negative.")
    if total_dur is not None and new_end > float(total_dur) + 1e-3:
        raise ValueError(f"new_end ({new_end:.3f}s) exceeds total audio duration ({total_dur:.3f}s).")
    if new_end - new_start < min_dur - 1e-6:
        raise ValueError(
            f"Segment duration ({new_end - new_start:.3f}s) must be at least {min_dur:.2f}s."
        )

    n = len(work)
    if sync_neighbors:
        if pos > 0:
            prev_s = float(work.loc[pos - 1, "start"])
            if new_start < prev_s + min_dur - 1e-6:
                raise ValueError(
                    f"new_start ({new_start:.3f}s) shrinks previous segment below {min_dur:.2f}s "
                    f"(min allowed: {prev_s + min_dur:.3f}s)."
                )
        if pos < n - 1:
            next_e = float(work.loc[pos + 1, "end"])
            if new_end > next_e - min_dur + 1e-6:
                raise ValueError(
                    f"new_end ({new_end:.3f}s) shrinks next segment below {min_dur:.2f}s "
                    f"(max allowed: {next_e - min_dur:.3f}s)."
                )
    else:
        if pos > 0 and new_start < float(work.loc[pos - 1, "end"]) - 1e-6:
            raise ValueError(
                f"new_start ({new_start:.3f}s) overlaps previous segment ending at "
                f"{float(work.loc[pos - 1, 'end']):.3f}s."
            )
        if pos < n - 1 and new_end > float(work.loc[pos + 1, "start"]) + 1e-6:
            raise ValueError(
                f"new_end ({new_end:.3f}s) overlaps next segment starting at "
                f"{float(work.loc[pos + 1, 'start']):.3f}s."
            )

    def _mark_human_boundary(r_idx: int, s_val: float, e_val: float, keep_type: bool = False):
        row = work.loc[r_idx].to_dict()
        row["start"] = s_val
        row["end"] = e_val
        row["duration"] = round(e_val - s_val, 3)
        row["cut_point_time"] = e_val
        if not keep_type:
            row["cut_type"] = "human_moved"
        row["decision_class"] = "HUMAN_OVERRIDE"
        row["status"] = "human_edited"
        row["needs_review"] = 0
        row["is_verified_midpoint"] = 0
        row["expected_midpoint"] = np.nan
        row["midpoint_error_ms"] = np.nan
        fallback_lbl = str(row.get("primary_label", row.get("label", "speech")))
        fallback_conf = float(row.get("confidence", 1.0)) if pd.notnull(row.get("confidence")) else 1.0
        comp = _compute_comp_dict(s_val, e_val, regions_df, fallback_lbl, fallback_conf)
        _apply_comp_to_row(row, comp)
        for k, v in row.items():
            work.loc[r_idx, k] = v

    _mark_human_boundary(pos, new_start, new_end, keep_type=False)
    if sync_neighbors:
        if pos > 0:
            _mark_human_boundary(pos - 1, float(work.loc[pos - 1, "start"]), new_start, keep_type=True)
        if pos < n - 1:
            _mark_human_boundary(pos + 1, new_end, float(work.loc[pos + 1, "end"]), keep_type=True)

    return _renorm_segments(work, reindex_ids=True)


def segment_split(
    df: pd.DataFrame,
    idx: Any,
    split_time: float,
    regions_df: Optional[pd.DataFrame] = None,
    min_dur: float = DEFAULT_MIN_SEGMENT_DUR,
) -> pd.DataFrame:
    """Split segment ``idx`` into two segments at ``split_time``."""
    work = df.sort_values(["start", "end"]).reset_index(drop=True).copy()
    row_key = _resolve_row_index(work, idx)
    pos = work.index.get_loc(row_key)

    row = work.loc[pos]
    s_t = float(row["start"])
    e_t = float(row["end"])
    split_time = round(float(split_time), 3)

    if not (s_t + min_dur - 1e-6 <= split_time <= e_t - min_dur + 1e-6):
        raise ValueError(
            f"Split time {split_time:.3f}s must be strictly within "
            f"[{s_t + min_dur:.3f}s, {e_t - min_dur:.3f}s]."
        )

    base_lbl = str(row.get("primary_label", row.get("label", "speech")))
    base_conf = float(row.get("confidence", 1.0)) if pd.notnull(row.get("confidence")) else 1.0

    left = row.to_dict()
    right = row.to_dict()

    for part, p_s, p_e in ((left, s_t, split_time), (right, split_time, e_t)):
        part["start"] = p_s
        part["end"] = p_e
        part["duration"] = round(p_e - p_s, 3)
        part["cut_point_time"] = p_e
        part["cut_type"] = "human_split"
        part["decision_class"] = "HUMAN_OVERRIDE"
        part["status"] = "human_edited"
        part["needs_review"] = 0
        part["is_verified_midpoint"] = 0
        part["expected_midpoint"] = np.nan
        part["midpoint_error_ms"] = np.nan
        comp = _compute_comp_dict(p_s, p_e, regions_df, base_lbl, base_conf)
        _apply_comp_to_row(part, comp)

    new_work = pd.concat(
        [work.iloc[:pos], pd.DataFrame([left, right]), work.iloc[pos + 1:]],
        ignore_index=True,
    )
    return _renorm_segments(new_work, reindex_ids=True)


def segment_merge(
    df: pd.DataFrame,
    idx: Any,
    direction: str = "next",
    regions_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge segment ``idx`` with its ``'next'`` or ``'prev'`` neighbor."""
    if direction not in ("next", "prev"):
        raise ValueError("direction must be 'next' or 'prev'.")

    work = df.sort_values(["start", "end"]).reset_index(drop=True).copy()
    row_key = _resolve_row_index(work, idx)
    pos = work.index.get_loc(row_key)

    if direction == "next":
        if pos >= len(work) - 1:
            raise ValueError("No next segment available to merge with.")
        i_a, i_b = pos, pos + 1
    else:
        if pos <= 0:
            raise ValueError("No previous segment available to merge with.")
        i_a, i_b = pos - 1, pos

    ra = work.loc[i_a]
    rb = work.loc[i_b]
    new_s = float(ra["start"])
    new_e = float(rb["end"])

    # Determine merged label when regions_df is unavailable
    dur_a = float(ra["end"]) - float(ra["start"])
    dur_b = float(rb["end"]) - float(rb["start"])
    lbl_a = str(ra.get("primary_label", ra.get("label", "speech")))
    lbl_b = str(rb.get("primary_label", rb.get("label", "speech")))
    dom_lbl = lbl_a if dur_a >= dur_b else lbl_b

    merged = ra.to_dict()
    merged["start"] = new_s
    merged["end"] = new_e
    merged["duration"] = round(new_e - new_s, 3)
    merged["cut_point_time"] = new_e
    merged["cut_type"] = "human_merged"
    merged["decision_class"] = "HUMAN_OVERRIDE"
    merged["status"] = "human_edited"
    merged["needs_review"] = 0
    merged["is_verified_midpoint"] = 0
    merged["expected_midpoint"] = np.nan
    merged["midpoint_error_ms"] = np.nan

    if regions_df is not None and not regions_df.empty:
        comp = _compute_comp_dict(new_s, new_e, regions_df, dom_lbl, 1.0)
    else:
        # Combine per-class durations directly from ra and rb if present
        durs_by_lbl = {
            "speech": float(ra.get("speech_duration", dur_a if lbl_a == "speech" else 0.0) or 0.0)
                      + float(rb.get("speech_duration", dur_b if lbl_b == "speech" else 0.0) or 0.0),
            "mixed": float(ra.get("mixed_duration", dur_a if lbl_a == "mixed" else 0.0) or 0.0)
                     + float(rb.get("mixed_duration", dur_b if lbl_b == "mixed" else 0.0) or 0.0),
            "music": float(ra.get("music_duration", dur_a if lbl_a == "music" else 0.0) or 0.0)
                     + float(rb.get("music_duration", dur_b if lbl_b == "music" else 0.0) or 0.0),
            "silence": float(ra.get("silence_duration", dur_a if lbl_a == "silence" else 0.0) or 0.0)
                       + float(rb.get("silence_duration", dur_b if lbl_b == "silence" else 0.0) or 0.0),
        }
        primary = max(durs_by_lbl, key=durs_by_lbl.get) if sum(durs_by_lbl.values()) > 0 else dom_lbl
        comp_parts = [f"{k}:{round(v, 2):.2f}" for k, v in durs_by_lbl.items() if v > 0.0]
        comp = {
            "primary_label": primary,
            "label_composition": "|".join(comp_parts) if comp_parts else f"{primary}:{new_e - new_s:.2f}",
            "source_region_ids": str(ra.get("source_region_ids", "")),
            "confidence": round(
                (float(ra.get("confidence", 1.0) or 1.0) * dur_a + float(rb.get("confidence", 1.0) or 1.0) * dur_b)
                / max(1e-6, dur_a + dur_b),
                3,
            ),
            "speech_duration": round(durs_by_lbl["speech"], 3),
            "mixed_duration": round(durs_by_lbl["mixed"], 3),
            "music_duration": round(durs_by_lbl["music"], 3),
            "silence_duration": round(durs_by_lbl["silence"], 3),
        }
    _apply_comp_to_row(merged, comp)

    for k, v in merged.items():
        work.loc[i_a, k] = v
    work = work.drop(index=i_b).reset_index(drop=True)
    return _renorm_segments(work, reindex_ids=True)


def segment_delete(
    df: pd.DataFrame,
    idx: Any,
    absorb: bool = False,
    regions_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Delete segment ``idx``.

    If ``absorb=True``, the deleted segment's time span is absorbed into its
    adjacent neighbor (previous if available, otherwise next). If ``absorb=False``,
    the segment is simply removed cleanly from the DataFrame.
    """
    work = df.sort_values(["start", "end"]).reset_index(drop=True).copy()
    row_key = _resolve_row_index(work, idx)
    pos = work.index.get_loc(row_key)

    if absorb:
        if len(work) < 2:
            raise ValueError("Cannot absorb deleted segment when fewer than 2 segments exist.")
        del_row = work.loc[pos]
        if pos > 0:
            nbr = pos - 1
            new_s = float(work.loc[nbr, "start"])
            new_e = float(del_row["end"])
        else:
            nbr = 1
            new_s = float(del_row["start"])
            new_e = float(work.loc[nbr, "end"])

        nbr_dict = work.loc[nbr].to_dict()
        nbr_dict["start"] = new_s
        nbr_dict["end"] = new_e
        nbr_dict["duration"] = round(new_e - new_s, 3)
        nbr_dict["cut_point_time"] = new_e
        nbr_dict["cut_type"] = "human_edited"
        nbr_dict["decision_class"] = "HUMAN_OVERRIDE"
        nbr_dict["status"] = "human_edited"
        nbr_dict["needs_review"] = 0
        nbr_dict["is_verified_midpoint"] = 0
        nbr_dict["expected_midpoint"] = np.nan
        nbr_dict["midpoint_error_ms"] = np.nan
        nbr_lbl = str(nbr_dict.get("primary_label", nbr_dict.get("label", "speech")))
        comp = _compute_comp_dict(new_s, new_e, regions_df, nbr_lbl, 1.0)
        _apply_comp_to_row(nbr_dict, comp)
        for k, v in nbr_dict.items():
            work.loc[nbr, k] = v

    work = work.drop(index=pos).reset_index(drop=True)
    return _renorm_segments(work, reindex_ids=True)


def segment_add(
    df: pd.DataFrame,
    start: float,
    end: float,
    label: Optional[str] = "speech",
    regions_df: Optional[pd.DataFrame] = None,
    video_id: Optional[str] = None,
    total_dur: Optional[float] = None,
    allow_overlap: bool = False,
    min_dur: float = DEFAULT_MIN_SEGMENT_DUR,
) -> pd.DataFrame:
    """Add a new segment covering ``[start, end]`` with an explicit or region-derived label."""
    if label is not None:
        label = str(label).lower().strip()
        if label not in VALID_SEGMENT_LABELS:
            raise ValueError(f"Invalid label {label!r}; expected one of {VALID_SEGMENT_LABELS}.")

    start = round(float(start), 3)
    end = round(float(end), 3)
    if start < 0.0:
        raise ValueError(f"start ({start:.3f}s) cannot be negative.")
    if total_dur is not None and end > float(total_dur) + 1e-3:
        raise ValueError(f"end ({end:.3f}s) exceeds total audio duration ({total_dur:.3f}s).")
    if end - start < min_dur - 1e-6:
        raise ValueError(f"Segment duration ({end - start:.3f}s) must be at least {min_dur:.2f}s.")

    work = df.copy()
    if not allow_overlap and not work.empty:
        overlaps = work[(work["start"].astype(float) < end - 0.015) & (work["end"].astype(float) > start + 0.015)]
        if not overlaps.empty:
            first = overlaps.iloc[0]
            raise ValueError(
                f"Interval [{start:.2f}s, {end:.2f}s] overlaps existing segment "
                f"#{first.get('segment_id', overlaps.index[0])} [{float(first['start']):.2f}s, {float(first['end']):.2f}s]. "
                "Delete/move the overlapping region or set allow_overlap=True."
            )

    vid = _infer_video_id(work, video_id)
    new_row = dict(_SCHEMA_DEFAULTS)
    new_row.update({
        "video_id": vid,
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "cut_point_time": end,
        "cut_type": "human_added",
    })
    if label is not None:
        # Honor the annotator's explicit label while still tracking source regions if available
        comp = _compute_comp_dict(start, end, None, fallback_label=label, fallback_conf=1.0)
        if regions_df is not None and not regions_df.empty:
            reg_comp = _compute_comp_dict(start, end, regions_df, fallback_label=label, fallback_conf=1.0)
            comp["source_region_ids"] = reg_comp.get("source_region_ids", "")
    else:
        comp = _compute_comp_dict(start, end, regions_df, fallback_label="speech", fallback_conf=1.0)
    _apply_comp_to_row(new_row, comp)

    for col in work.columns:
        if col not in new_row:
            new_row[col] = np.nan

    if work.empty:
        combined = pd.DataFrame([new_row])
    else:
        combined = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

    return _renorm_segments(combined, reindex_ids=True)


# =====================================================================
# Disk Synchronization (segments.csv & segments/*.wav)
# =====================================================================

def sync_segments_to_disk(
    new_df: pd.DataFrame,
    audio: Optional[np.ndarray],
    video_dir: str | Path,
    sr: int = DEFAULT_SR,
    old_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Persist updated ``segments.csv`` and atomically synchronize ``segments/*.wav``.

    Unchanged segment spans reuse their existing WAV clip on disk when available;
    new or edited spans are freshly sliced from ``audio`` with an 8 ms Hann fade.
    """
    vdir = Path(video_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    seg_dir = vdir / "segments"
    staging_dir = vdir / "segments_staging"
    csv_path = vdir / "segments.csv"

    synced = _renorm_segments(new_df, reindex_ids=True)

    if audio is not None:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Build lookup from previous unchanged spans to avoid re-writing untouched clips
        reusable: Dict[Tuple[float, float], Path] = {}
        if old_df is not None and not old_df.empty and seg_dir.exists():
            for r in old_df.itertuples():
                rel = getattr(r, "wav_file", None)
                if not rel:
                    continue
                src_p = vdir / str(rel)
                if src_p.exists():
                    reusable[(round(float(r.start), 3), round(float(r.end), 3))] = src_p

        total_samples = len(audio)
        for r in synced.itertuples():
            fname = Path(str(r.wav_file)).name
            dst_p = staging_dir / fname
            key = (round(float(r.start), 3), round(float(r.end), 3))
            src_p = reusable.get(key)
            copied = False
            if src_p is not None and src_p.exists():
                try:
                    info = sf.info(str(src_p))
                    if abs(info.duration - float(r.duration)) <= 0.02 and info.samplerate == sr:
                        shutil.copy2(src_p, dst_p)
                        copied = True
                except Exception:
                    copied = False

            if not copied:
                i0 = max(0, int(round(float(r.start) * sr)))
                i1 = min(total_samples, int(round(float(r.end) * sr)))
                clip = apply_hann_fade(audio[i0:i1].copy(), sr=sr)
                write_wav(dst_p, clip, sample_rate=sr)

        if seg_dir.exists():
            backup_dir = vdir / "_segments_old_editor"
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            seg_dir.rename(backup_dir)
            staging_dir.rename(seg_dir)
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            staging_dir.rename(seg_dir)

    synced.to_csv(csv_path, index=False)
    return synced


# =====================================================================
# Visualization Helper & Interactive ipywidgets UI
# =====================================================================

def plot_segments_window(
    audio: np.ndarray,
    segments_df: pd.DataFrame,
    t0: float,
    t1: float,
    sr: int = DEFAULT_SR,
    selected_pos: Optional[int] = None,
    title: str = "",
) -> plt.Figure:
    """Plot audio waveform overlay with segment boundaries and labels over ``[t0, t1]``."""
    total_dur = len(audio) / sr
    t0, t1 = max(0.0, float(t0)), min(total_dur, float(t1))
    if t1 <= t0:
        t1 = min(total_dur, t0 + 1.0)

    i0, i1 = int(t0 * sr), int(t1 * sr)
    slice_y = audio[i0:i1]
    t_ax = np.linspace(t0, t1, len(slice_y)) if len(slice_y) > 0 else np.array([t0, t1])

    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    if len(slice_y) > 0:
        ax.plot(t_ax, slice_y, color="#2b5c8f", linewidth=0.65, alpha=0.85)

    for pos, r in segments_df.sort_values("start").reset_index(drop=True).iterrows():
        rs, re = float(r["start"]), float(r["end"])
        if re < t0 or rs > t1:
            continue
        a, b = max(t0, rs), min(t1, re)
        lbl = str(r.get("primary_label", r.get("label", "speech"))).lower()
        color = SEGMENT_COLORS.get(lbl, "#888888")
        is_sel = (selected_pos is not None and pos == selected_pos)
        ax.axvspan(a, b, color=color, alpha=0.32 if is_sel else 0.14)
        ax.axvline(rs, color="#dc3545" if is_sel else "#333333", ls="--" if not is_sel else "-", lw=1.8 if is_sel else 1.0)
        ax.axvline(re, color="#dc3545" if is_sel else "#333333", ls="--" if not is_sel else "-", lw=1.8 if is_sel else 1.0)
        ymax = max(0.05, float(np.max(np.abs(slice_y))) if len(slice_y) else 1.0)
        sid = r.get("segment_id", pos + 1)
        ax.text(
            (a + b) / 2.0,
            ymax * 0.85,
            f"Seg #{sid} [{lbl.upper()}]\n{rs:.2f}-{re:.2f}s ({re-rs:.1f}s)",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold" if is_sel else "normal",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
        )

    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"Segments Timeline ({t0:.2f}s – {t1:.2f}s)")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig


def launch_segment_editor(
    audio: np.ndarray,
    segments_df: pd.DataFrame,
    video_dir: str | Path,
    sr: int = DEFAULT_SR,
    regions_df: Optional[pd.DataFrame] = None,
    user: str = "annotator",
) -> Dict[str, Any]:
    """Launch an interactive widget UI to move, split, merge, delete, or add cut segments.

    Every operation synchronizes ``segments.csv``, updates ``segments/*.wav``,
    and logs the edit to ``changes.csv``. Returns a mutable state dict whose
    ``state["df"]`` always holds the latest edited segments DataFrame.
    """
    vdir = Path(video_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    total_dur = len(audio) / sr

    state: Dict[str, Any] = {"df": _renorm_segments(segments_df.copy(), reindex_ids=True)}
    busy = {"flag": False}

    # Controls
    user_w = widgets.Text(value=user, description="Annotator:", layout=widgets.Layout(width="220px"))
    note_w = widgets.Text(placeholder="Reason for edit...", description="Note:", layout=widgets.Layout(width="420px"))
    sel = widgets.Dropdown(description="Segment:", layout=widgets.Layout(width="78%"))

    w_lbl = widgets.Dropdown(options=list(VALID_SEGMENT_LABELS), description="Label:", layout=widgets.Layout(width="200px"))
    b_relabel = widgets.Button(description="Set Label", button_style="info")

    w_start = widgets.FloatText(description="Start (s):", step=0.1, layout=widgets.Layout(width="180px"))
    w_end = widgets.FloatText(description="End (s):", step=0.1, layout=widgets.Layout(width="180px"))
    w_sync_nbr = widgets.Checkbox(value=True, description="Sync neighbor borders", indent=False, layout=widgets.Layout(width="180px"))
    b_move = widgets.Button(description="↔️ Move Segment", button_style="warning")

    w_split = widgets.FloatText(description="Split at (s):", step=0.1, layout=widgets.Layout(width="190px"))
    b_split = widgets.Button(description="✂️ Split", button_style="primary")
    b_merge_prev = widgets.Button(description="🔗 Merge w/ Prev")
    b_merge_next = widgets.Button(description="🔗 Merge w/ Next")

    w_absorb = widgets.Checkbox(value=False, description="Absorb span into neighbor", indent=False, layout=widgets.Layout(width="200px"))
    b_delete = widgets.Button(description="🗑️ Delete", button_style="danger")

    w_add_s = widgets.FloatText(description="Add Start:", step=0.1, layout=widgets.Layout(width="175px"))
    w_add_e = widgets.FloatText(description="End:", step=0.1, layout=widgets.Layout(width="165px"))
    w_add_lbl = widgets.Dropdown(options=list(VALID_SEGMENT_LABELS), value="speech", description="as:", layout=widgets.Layout(width="175px"))
    b_add = widgets.Button(description="➕ Add Segment", button_style="success")

    b_play_seg = widgets.Button(description="▶ Segment")
    b_play_start = widgets.Button(description="▶ Start (±2s)")
    b_play_end = widgets.Button(description="▶ End (±2s)")

    msg_out = widgets.Output()
    plot_out = widgets.Output()
    audio_out = widgets.Output()

    def _build_opts() -> List[Tuple[str, int]]:
        df = state["df"]
        if df.empty:
            return [("(No segments)", -1)]
        opts = []
        for pos, r in df.iterrows():
            lbl = str(r.get("primary_label", r.get("label", "speech"))).upper()
            st_flag = "✔" if r.get("status") == "human_edited" else ""
            opts.append((
                f"#{r['segment_id']} [{lbl}] {r['start']:.2f}s–{r['end']:.2f}s ({r['duration']:.2f}s) {st_flag}".strip(),
                int(pos),
            ))
        return opts

    def _populate_fields(_=None):
        if busy["flag"]:
            return
        df = state["df"]
        plot_out.clear_output(wait=True)
        audio_out.clear_output(wait=True)

        if df.empty or sel.value is None or sel.value < 0 or sel.value not in df.index:
            w_add_s.value, w_add_e.value = 0.0, min(5.0, round(total_dur, 2))
            return

        pos = int(sel.value)
        r = df.loc[pos]
        lbl = str(r.get("primary_label", r.get("label", "speech"))).lower()
        w_lbl.value = lbl if lbl in VALID_SEGMENT_LABELS else "speech"
        w_start.value = float(r["start"])
        w_end.value = float(r["end"])
        w_split.value = round((float(r["start"]) + float(r["end"])) / 2.0, 2)
        w_add_s.value = float(r["end"])
        w_add_e.value = round(min(total_dur, float(r["end"]) + 5.0), 2)

        with plot_out:
            fig = plot_segments_window(
                audio,
                df,
                t0=max(0.0, float(r["start"]) - 3.0),
                t1=min(total_dur, float(r["end"]) + 3.0),
                sr=sr,
                selected_pos=pos,
                title=f"Segment #{r['segment_id']} [{lbl.upper()}] {float(r['start']):.2f}–{float(r['end']):.2f}s",
            )
            display(fig)
            plt.close(fig)

        with audio_out:
            i0 = max(0, int(float(r["start"]) * sr))
            i1 = min(len(audio), int(float(r["end"]) * sr))
            if i1 > i0:
                display(Audio(audio[i0:i1], rate=sr))

    def _commit(new_df: pd.DataFrame, field: str, sid: Any, old_val: Any, new_val: Any, next_pos: int):
        old_df = state["df"]
        synced = sync_segments_to_disk(new_df, audio=audio, video_dir=vdir, sr=sr, old_df=old_df)
        state["df"] = synced
        log_human_change(vdir, user_w.value, f"segment_{sid}", field, old_val, new_val, note_w.value)

        busy["flag"] = True
        sel.options = _build_opts()
        if synced.empty:
            sel.value = -1
        else:
            sel.value = max(0, min(next_pos, len(synced) - 1))
        busy["flag"] = False
        _populate_fields()
        with msg_out:
            print(f"✅ Applied [{field}] -> total segments: {len(synced)}")

    def _guard(fn):
        def _handler(_):
            msg_out.clear_output()
            try:
                fn()
            except Exception as e:
                with msg_out:
                    print(f"❌ {e}")
        return _handler

    def _cur() -> Tuple[int, pd.Series]:
        if state["df"].empty or sel.value is None or sel.value < 0:
            raise ValueError("No segment selected.")
        pos = int(sel.value)
        return pos, state["df"].loc[pos]

    def on_relabel():
        pos, r = _cur()
        df_new = state["df"].copy()
        old_lbl = str(r.get("primary_label", r.get("label", "speech")))
        df_new.loc[pos, "label"] = w_lbl.value
        df_new.loc[pos, "primary_label"] = w_lbl.value
        df_new.loc[pos, "status"] = "human_edited"
        df_new.loc[pos, "needs_review"] = 0
        _commit(df_new, "relabel", r["segment_id"], old_lbl, w_lbl.value, pos)

    def on_move():
        pos, r = _cur()
        new_df = segment_move(
            state["df"], pos, w_start.value, w_end.value,
            sync_neighbors=bool(w_sync_nbr.value),
            regions_df=regions_df, total_dur=total_dur,
        )
        _commit(new_df, "move", r["segment_id"], f"{r['start']}-{r['end']}", f"{w_start.value}-{w_end.value}", pos)

    def on_split():
        pos, r = _cur()
        new_df = segment_split(state["df"], pos, w_split.value, regions_df=regions_df)
        _commit(new_df, "split", r["segment_id"], f"{r['start']}-{r['end']}", f"split@{w_split.value}", pos)

    def on_merge_prev():
        pos, r = _cur()
        new_df = segment_merge(state["df"], pos, direction="prev", regions_df=regions_df)
        _commit(new_df, "merge_prev", r["segment_id"], f"#{r['segment_id']}", "merged_with_prev", max(0, pos - 1))

    def on_merge_next():
        pos, r = _cur()
        new_df = segment_merge(state["df"], pos, direction="next", regions_df=regions_df)
        _commit(new_df, "merge_next", r["segment_id"], f"#{r['segment_id']}", "merged_with_next", pos)

    def on_delete():
        pos, r = _cur()
        new_df = segment_delete(state["df"], pos, absorb=bool(w_absorb.value), regions_df=regions_df)
        _commit(new_df, "delete", r["segment_id"], f"{r['start']}-{r['end']}", "deleted", max(0, pos - 1))

    def on_add():
        new_df = segment_add(
            state["df"], w_add_s.value, w_add_e.value, label=w_add_lbl.value,
            regions_df=regions_df, total_dur=total_dur,
        )
        # Focus newly added segment
        matches = new_df.index[
            (new_df["start"] - round(float(w_add_s.value), 3)).abs() < 0.01
        ].tolist()
        target_pos = int(matches[0]) if matches else len(new_df) - 1
        _commit(new_df, "add", "new", "-", f"{w_add_s.value}-{w_add_e.value}({w_add_lbl.value})", target_pos)

    def _play_range(a: float, b: float):
        audio_out.clear_output(wait=True)
        i0 = max(0, int(max(0.0, a) * sr))
        i1 = min(len(audio), int(min(total_dur, b) * sr))
        with audio_out:
            if i1 > i0:
                display(Audio(audio[i0:i1], rate=sr))

    b_relabel.on_click(_guard(on_relabel))
    b_move.on_click(_guard(on_move))
    b_split.on_click(_guard(on_split))
    b_merge_prev.on_click(_guard(on_merge_prev))
    b_merge_next.on_click(_guard(on_merge_next))
    b_delete.on_click(_guard(on_delete))
    b_add.on_click(_guard(on_add))
    b_play_seg.on_click(lambda _: _play_range(w_start.value, w_end.value))
    b_play_start.on_click(lambda _: _play_range(w_start.value - 2.0, w_start.value + 2.0))
    b_play_end.on_click(lambda _: _play_range(w_end.value - 2.0, w_end.value + 2.0))
    sel.observe(_populate_fields, names="value")

    busy["flag"] = True
    sel.options = _build_opts()
    sel.value = 0 if not state["df"].empty else -1
    busy["flag"] = False

    ui = widgets.VBox([
        widgets.HTML("<h3>✂️ Segment Editor: Move, Split, Merge, Delete & Add Segments</h3>"),
        widgets.HBox([user_w, note_w]),
        sel,
        plot_out,
        widgets.HBox([b_play_seg, b_play_start, b_play_end, w_lbl, b_relabel]),
        audio_out,
        widgets.HBox([w_start, w_end, w_sync_nbr, b_move]),
        widgets.HBox([w_split, b_split, b_merge_prev, b_merge_next]),
        widgets.HBox([w_absorb, b_delete]),
        widgets.HBox([w_add_s, w_add_e, w_add_lbl, b_add]),
        msg_out,
    ])
    display(ui)
    _populate_fields()
    state["ui"] = ui
    return state
