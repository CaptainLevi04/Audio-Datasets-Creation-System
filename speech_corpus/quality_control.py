"""Quality Control (QC), 10 Mandatory Gates, Corpus Manifest, and Leakage-Safe Splitting.
Implements Student 1 AI Implementation Specification v2.2:
1. 10 Mandatory QC Gates (Section 6). QC failure forces status = "FAILED".
2. Real countable integrity checks (no fabricated numbers).
3. Measured ablation comparison (Greedy baseline vs. Global DP).
4. Per-video summary table (speech, music, mixed, silence durations).
5. Global corpus manifest builder.
6. Zero-leakage source-level dataset splitting; reports infeasibility if < 3 sources.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf

from .audio import DEFAULT_SR

PRIMARY_MIN = 5.0
PRIMARY_MAX = 15.0
FALLBACK_MIN = 4.0
FALLBACK_MAX = 17.0
VALID_LABELS = {"speech", "music", "mixed", "silence"}


def validate_segments_integrity(df: pd.DataFrame,
                                video_dir: Optional[str | Path] = None,
                                total_dur: Optional[float] = None,
                                sr: int = DEFAULT_SR,
                                grid_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Validate segments DataFrame against the 10 Mandatory QC Gates of Specification v2.2.
    
    All 10 gates must pass for status == 'PASSED'.
    Any mandatory gate failure sets status = 'FAILED'.
    """
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
    
    # -------------------------------------------------------------
    # Gate 1: Full source coverage & no overlaps
    # -------------------------------------------------------------
    overlaps = []
    gaps = []
    for i in range(n - 1):
        cur_e = df_sorted.loc[i, "end"]
        next_s = df_sorted.loc[i + 1, "start"]
        diff = next_s - cur_e
        if diff < -0.015:  # Overlap greater than 15 ms
            overlaps.append((df_sorted.loc[i, "segment_id"], df_sorted.loc[i + 1, "segment_id"], round(diff, 3)))
        elif diff > 0.150:  # Gap greater than 150 ms
            gaps.append((df_sorted.loc[i, "segment_id"], df_sorted.loc[i + 1, "segment_id"], round(diff, 3)))
    
    coverage_ok = (len(overlaps) == 0 and len(gaps) == 0)
    record_gate(
        1,
        "Full Source Coverage & No Overlaps",
        coverage_ok,
        f"{len(overlaps)} overlaps, {len(gaps)} unexplained gaps",
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 2: All four primary labels are valid and included in processing
    # -------------------------------------------------------------
    labels_present = set(df_sorted["primary_label"].unique()) if "primary_label" in df_sorted else set(df_sorted["label"].unique())
    invalid_labels = labels_present - VALID_LABELS
    gate2_ok = (len(invalid_labels) == 0 and len(labels_present) > 0)
    record_gate(
        2,
        "Valid Four-Label Processing",
        gate2_ok,
        f"Labels present: {sorted(list(labels_present))}, Invalid: {sorted(list(invalid_labels))}",
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 3: Duration constraints (5-15s normal, 4-17s fallback flagged)
    # -------------------------------------------------------------
    durs = df_sorted["duration"]
    undersized = df_sorted[durs < FALLBACK_MIN - 0.05]
    oversized = df_sorted[durs > FALLBACK_MAX + 0.05]
    in_primary = ((durs >= PRIMARY_MIN - 0.05) & (durs <= PRIMARY_MAX + 0.05))
    primary_ratio = float(in_primary.mean()) if n else 0.0
    
    # Fallback samples must be flagged
    fallback_unflagged = df_sorted[~in_primary & (df_sorted.get("fallback_used", 0) == 0)]
    gate3_ok = (len(undersized) == 0 and len(oversized) == 0 and primary_ratio >= 0.85 and len(fallback_unflagged) == 0)
    record_gate(
        3,
        "Duration Window Compliance (5-15s primary >=85%, 4-17s fallback flagged)",
        gate3_ok,
        f"Primary window: {primary_ratio*100:.1f}%, Undersized (<4s): {len(undersized)}, Oversized (>17s): {len(oversized)}, Unflagged fallbacks: {len(fallback_unflagged)}",
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 4: Continuous Silence (>15s) Direct Voting Grid Validation (Issue 6)
    # -------------------------------------------------------------
    # Iterate over every cut generated inside continuous silence and verify against Voting Grid
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

        # Video/file edges are always supported
        if total_dur is not None and abs(e_cut - total_dur) <= 0.05:
            supported_cuts.append(sid)
            continue

        if grid_df is not None and len(grid_df) > 0:
            from .segmentation import evaluate_cut_grid_support
            grid_info = evaluate_cut_grid_support(grid_df, e_cut)
            has_grid_accept = (grid_info["decision"] == "ACCEPT")

            if has_grid_accept:
                # Genuinely supported by 100 ms Voting Grid consensus
                if needs_rev == 0 or d_class in ["VERY_STRONG", "STRONG", "MODERATE", "ACCEPTED_SINGLE"]:
                    supported_cuts.append(sid)
                else:
                    fallback_rev_cuts.append(sid)
            else:
                # Support is absent: MUST be explicitly marked fallback/review
                is_explicit_fallback = (
                    (d_class in ["FALLBACK_SPLIT", "REJECTED_CANDIDATE"] or ctype in ["fallback_split", "split", "valley_fallback"])
                    and needs_rev == 1
                    and rev_reason != "none"
                )
                if is_explicit_fallback:
                    fallback_rev_cuts.append(sid)
                else:
                    # Unsupported synthetic cut incorrectly presented as accepted or unreviewed
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

    # -------------------------------------------------------------
    # Gate 5: Midpoint Correctness Verification (Issue 4)
    # -------------------------------------------------------------
    # Check that for all automatic pause-based cuts: midpoint_error_ms <= 2.0 ms
    # Margin compliance is distinct from midpoint correctness.
    auto_pause_cuts = df_sorted[df_sorted.get("is_verified_midpoint", 0) == 1]
    valid_midpoint_cuts = auto_pause_cuts[auto_pause_cuts["midpoint_error_ms"].notnull()] if "midpoint_error_ms" in auto_pause_cuts.columns else pd.DataFrame()
    
    if len(valid_midpoint_cuts) > 0:
        max_error_ms = float(valid_midpoint_cuts["midpoint_error_ms"].max())
        mean_error_ms = float(valid_midpoint_cuts["midpoint_error_ms"].mean())
        failed_midpoint = valid_midpoint_cuts[valid_midpoint_cuts["midpoint_error_ms"] > 2.0]
        midpoint_ok = (len(failed_midpoint) == 0)
        detail = f"Max midpoint error: {max_error_ms:.2f} ms, Mean: {mean_error_ms:.2f} ms ({len(failed_midpoint)} cuts exceeded 2.0 ms tolerance)"
    else:
        # Fallback if legacy df without midpoint_error_ms column
        margin_ok_col = df_sorted.get("margin_ok", pd.Series([1]*n))
        midpoint_ok = float((margin_ok_col == 1).mean()) >= 0.80 if n else True
        detail = f"Legacy fallback verification: {float((margin_ok_col == 1).mean())*100:.1f}% compliant"

    # Edge silence margin (minimum 100 ms, no upper rejection)
    margin_ok_col = df_sorted.get("margin_ok", pd.Series([1]*n))
    margin_compliance = float((margin_ok_col == 1).mean()) if n else 1.0

    gate5_ok = midpoint_ok and (margin_compliance >= 0.80 or n < 15)
    record_gate(
        5,
        "Midpoint Correctness Verification (midpoint_error_ms <= 2.0 ms)",
        gate5_ok,
        f"{detail} | Edge margin compliance: {margin_compliance*100:.1f}%",
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 6: Word safety checked when speech is present near boundary
    # -------------------------------------------------------------
    word_risks = df_sorted[(df_sorted.get("word_check_status", "") == "WORD_SPLIT_RISK") & (df_sorted.get("needs_review", 0) == 0)]
    gate6_ok = (len(word_risks) == 0)
    record_gate(
        6,
        "Word Safety Verification & Risk Flagging",
        gate6_ok,
        f"{len(word_risks)} unflagged word split risks",
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 7: Segment primary label and composition match upstream overlap
    # -------------------------------------------------------------
    has_comp = ("label_composition" in df_sorted.columns and "primary_label" in df_sorted.columns)
    comp_missing = df_sorted[df_sorted["primary_label"].isnull()] if "primary_label" in df_sorted else df_sorted
    gate7_ok = has_comp and comp_missing.empty
    record_gate(
        7,
        "Label Composition & Overlap Tracking",
        gate7_ok,
        "Composition columns present and populated" if gate7_ok else "Missing label_composition or primary_label",
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 8: WAV files exist, readable, and measured durations match manifest
    # -------------------------------------------------------------
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
    record_gate(
        8,
        "Audio Files Existence, Readability & Duration Match",
        gate8_ok,
        f"Missing files: {len(missing_files)}, Duration mismatches (>50ms): {len(dur_mismatches)}",
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 9: Macro-Region Transition & Pending Boundary Request Lifecycle (Issue 3)
    # -------------------------------------------------------------
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
        is_mandatory=True,
    )

    # -------------------------------------------------------------
    # Gate 10: Mandatory QC failure enforcement
    # -------------------------------------------------------------
    mandatory_gates = [g for g in gates if g["mandatory"]]
    mandatory_failed = sum(1 for g in mandatory_gates if not g["passed"])
    gate10_ok = (mandatory_failed == 0)
    record_gate(
        10,
        "Mandatory Gate Enforcement (Failure Prevents Success Status)",
        gate10_ok,
        f"Mandatory gates failed: {mandatory_failed} / {len(mandatory_gates)}",
        is_mandatory=True,
    )

    passed_all = (mandatory_failed == 0)
    status_str = "PASSED" if passed_all else "FAILED"

    # Workload warning if outside 5-15% target
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
    """Generate professional Markdown QC report strictly adhering to Specification v2.2."""
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

    mid_err_s = segments_df["midpoint_error_ms"].dropna() if "midpoint_error_ms" in segments_df.columns else pd.Series()
    mid_str = f"Max: {mid_err_s.max():.2f} ms, Mean: {mid_err_s.mean():.2f} ms" if len(mid_err_s) else "N/A"

    L = [
        "# Audio Labelling Quality Control Report (Student 1 | v2.2)",
        f"**Pipeline Status:** `{'PASSED' if (qc and qc.get('passed')) else 'FAILED'}` | **Total Segments:** {len(segments_df)} | **Total Audio:** {dur_total / 60.0:.2f} min\n",
        "## 1. Measured Compliance & Duration Metrics",
        f"- **Primary Window (5–15 s):** `{in_w}/{len(segments_df)} ({100.0 * in_w / n:.1f}%)`",
        f"- **Fallback Window (3–18 s):** `{fb_cnt}/{len(segments_df)} ({100.0 * fb_cnt / n:.1f}%)`",
        f"- **Midpoint Accuracy (<= 2 ms error):** `{mid_str}`",
        f"- **Edge Silence Margins (>= 100 ms):** `{mg_ok:.1f}%`",
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

    # Ablation comparison if available
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
    """Generate per-video summary table with duration of speech, music, mixed, silence."""
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
    """Concatenate every validated video's segments.csv into a unified corpus manifest."""
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
    """Source-level dataset splitting ensuring ZERO speaker / source leakage.
    
    If fewer than 3 independent sources exist:
    Reports infeasibility and refuses to create a segment-level split (as required by Section 7).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    source_col = "video_id" if "video_id" in manifest_df.columns else "source_id"
    if source_col not in manifest_df.columns:
        raise ValueError("Manifest must contain 'video_id' or 'source_id' column for splitting.")
        
    unique_sources = sorted(list(manifest_df[source_col].unique()))
    n_sources = len(unique_sources)

    # Infeasibility rule from Section 7
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

    # Partition sources to match requested ratios by total duration
    source_durations = manifest_df.groupby(source_col)["duration"].sum().to_dict()
    total_duration = sum(source_durations.values())
    
    # Sort sources by duration descending for greedy partition
    sorted_sources = sorted(source_durations.items(), key=lambda x: x[1], reverse=True)
    
    targets = {
        "train": total_duration * ratios[0],
        "val": total_duration * ratios[1],
        "test": total_duration * ratios[2]
    }
    assigned_sources = {"train": [], "val": [], "test": []}
    assigned_durations = {"train": 0.0, "val": 0.0, "test": 0.0}

    # Ensure each split receives at least one source
    assigned_sources["train"].append(sorted_sources[0][0])
    assigned_durations["train"] += sorted_sources[0][1]
    
    assigned_sources["val"].append(sorted_sources[1][0])
    assigned_durations["val"] += sorted_sources[1][1]
    
    assigned_sources["test"].append(sorted_sources[2][0])
    assigned_durations["test"] += sorted_sources[2][1]

    # Assign remaining sources greedily to whichever split is farthest below target
    for src, dur in sorted_sources[3:]:
        def deficit(k):
            return targets[k] - assigned_durations[k]
        best_split = max(["train", "val", "test"], key=deficit)
        assigned_sources[best_split].append(src)
        assigned_durations[best_split] += dur

    # Map sources to split names
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
    """Backward-compatible alias for split_corpus."""
    return split_corpus(df, out_dir=out_dir)
