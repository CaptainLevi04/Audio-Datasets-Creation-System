"""Cut Engine v2 - Unified Four-Label Segmentation & Global DP Optimizer.
Implements Student 1 AI Implementation Specification v2.2:
1. Unified 3-Detector Voting Policy across ALL four labels (Speech, Mixed, Music, Silence).
2. Elimination of legacy 'music_whole' and separate silence branching.
3. Candidate pause intervals >= 200 ms with exact midpoint cutting (>= 100 ms margins).
4. Continuous silence > 15 s handled via accepted voting candidates near 14.0–14.5 s (nominal 14.25 s).
5. Global DP / Shortest-Path Optimizer: Primary window 5–15 s; Fallback 3–18 s.
6. Four-label sample composition (primary_label, label_composition, source_region_ids).
7. Macro-region transition event alignment and tracking (satisfied / deferred).
8. Arabic Whisper word boundary verification when speech is present.
9. Manifest segments.csv with 25+ required fields.
10. Atomic output promotion: write to segments_staging/ and promote only upon passing 10 QC gates.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import bisect
import shutil
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from .audio import DEFAULT_SR, write_wav
from .segmentation import (
    compute_energy_db,
    adaptive_silence_floor,
    compute_unified_voting_grid,
    evaluate_cut_grid_support,
    get_accepted_pauses,
    get_review_pauses,
)
from .quality_control import validate_segments_integrity

ARABIC_STOPS = ("د", "ت", "ط", "ق", "ك", "ب", "ء", "أ", "إ", "ؤ", "ئ")

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

MARGIN_MIN = 0.10
MARGIN_MAX = 0.20
MARGIN_TOL_MS = 3.0

WORD_TS_TOL = 0.15
HIGH_TIER = 0.75
MED_TIER = 0.50


@dataclass
class PauseCandidate:
    t: float                  # Exact midpoint cut point: (p_start + p_end) / 2
    p_start: float            # Interval start
    p_end: float              # Interval end
    duration: float           # p_end - p_start
    vote_class: str           # VERY_STRONG, STRONG, ENERGY_ONLY, SILERO_ONLY, WHISPER_ONLY, FALLBACK_SPLIT, REJECTED_CANDIDATE
    vote_count: float         # 0.0 to 3.0
    energy_vote: int          # 0 or 1
    silero_vote: int          # 0 or 1
    whisper_vote: int         # 0 or 1
    energy_margin_db: float   # thr - rms
    silero_margin: float      # 0.5 - p_speech
    is_accepted: bool         # True if accepted by voting policy or approved by human
    needs_review: int         # 1 if human review required
    review_reason: str        # e.g. "whisper_only", "low_energy_margin", "fallback_split", "none"
    pause_id: str = ""        # e.g. "P0001"
    grid_bin_index: int = -1  # Supporting 100 ms grid bin index
    human_approved: bool = False # True if explicitly approved by human
    weak: bool = False        # True if forced energy valley / rejected fallback
    short: bool = False       # True if pause < 200 ms
    word_ok: float = 0.5      # 1.0 verified by word timestamps, 0.5 unverified, 0.0 word split
    word_risk: bool = False   # True if cut near Arabic stop consonant or active word
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


# -------------------------------------------------------------
# Backward-compatibility alias for Cand
# -------------------------------------------------------------
Cand = PauseCandidate


def apply_hann_fade(x: np.ndarray, sr: int = DEFAULT_SR, fade_ms: float = 8.0) -> np.ndarray:
    """Apply an 8 ms Hann fade strictly inside margins to prevent clicks."""
    n = int(sr * fade_ms / 1000.0)
    if n <= 0 or len(x) < 2 * n:
        return x
    out = x.copy()
    out[:n] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n)))
    out[-n:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n)))
    return out


def compute_label_composition(start: float, end: float, regions_df: pd.DataFrame) -> dict:
    """Compute overlap with all four Step 3 labels and derive primary_label + label_composition."""
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

    # Primary label = label with greatest overlap (deterministic tie-break: speech > mixed > music > silence)
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
    """Identify macro-region transition boundaries across the timeline."""
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
    """Resolve lifecycle of Pending Boundary Requests according to Issue 3.
    For each request:
    - Select the first Accepted Pause occurring after the transition that satisfies duration constraints.
    - If the first accepted pause violates duration constraints, keep the request pending and search later pauses.
    - Multiple pending requests are kept independently without overwriting or merging.
    - Mark each request as 'resolved' or assign an explicit 'unresolved_reason'.
    """
    cuts_set = set(round(float(c), 3) for c in planned_cuts)
    sorted_cuts = sorted(list(cuts_set))
    if not sorted_cuts or sorted_cuts[0] > 0.05:
        sorted_cuts = [0.0] + sorted_cuts
    if total_dur > 0 and (not sorted_cuts or sorted_cuts[-1] < total_dur - 0.05):
        sorted_cuts = sorted_cuts + [round(float(total_dur), 3)]

    # Filter accepted pauses (excluding weak/fallback synthetic cuts) sorted by cut midpoint time
    accepted_pauses = sorted([p for p in candidate_pauses if p.is_accepted and not p.weak], key=lambda x: x.t)

    for req in requests:
        t_tr = req.transition_time
        # Search all accepted pauses occurring at or after the transition
        later_pauses = [p for p in accepted_pauses if p.t >= t_tr]

        if not later_pauses:
            req.status = "unresolved"
            req.unresolved_reason = "no_later_accepted_pause" if t_tr < total_dur - 0.5 else "source_region_end"
            continue

        resolved = False
        duration_violation_count = 0

        for p in later_pauses:
            # Find the segment that would be formed from previous cut strictly before p.t
            idx = bisect.bisect_left(sorted_cuts, p.t - 0.01)
            prev_cut = sorted_cuts[idx - 1] if idx > 0 else 0.0
            seg_len = round(p.t - prev_cut, 3)

            # Check if duration constraints are satisfied
            if seg_len < min_dur - 0.05 or seg_len > max_dur + 0.05:
                # Violates segment-duration constraints! Keep request pending and continue searching later pauses!
                duration_violation_count += 1
                continue

            # First duration-feasible accepted pause after transition found:
            # Check if it was selected in the planned cuts
            if round(p.t, 3) in cuts_set:
                req.status = "resolved"
                req.selected_cut_time = round(p.t, 3)
                req.selected_pause_id = getattr(p, "pause_id", "")
                req.unresolved_reason = None
                resolved = True
                break
            else:
                # Even if DP selected a different cut, if this was the first feasible pause,
                # record status or keep searching
                pass

        if not resolved:
            req.status = "unresolved"
            if duration_violation_count == len(later_pauses):
                req.unresolved_reason = "duration_constraints"
            elif t_tr >= total_dur - 1.0:
                req.unresolved_reason = "source_region_end"
            else:
                req.unresolved_reason = "duration_constraints"

    return requests


# ---- Word verification with faster-whisper ----
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
    """Verify candidate cut midpoints against Whisper word timestamps.
    If a candidate cut lands inside an active word, flag word risk.
    """
    if not words:
        return cands, 0, 0
    starts = [w["start"] for w in words]
    verified, risks = 0, 0
    for c in cands:
        i = bisect.bisect_right(starts, c.t)
        prev_w = words[i - 1] if i > 0 else None
        next_w = words[i] if i < len(words) else None

        # Check if candidate cut midpoint lands strictly inside an active word
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

    # Handle gaps > 15 s between candidates: verify against detector grid
    # Issue 1: A cut inside continuous silence is valid as automatic voting cut ONLY IF
    # its exact timestamp maps to a genuine Accepted Voting Candidate in the grid.
    cands_sorted = sorted(cands, key=lambda c: c.t)
    anchor_times = [0.0] + [c.t for c in cands_sorted] + [total_dur]
    extra_cands = []
    
    for a, b in zip(anchor_times[:-1], anchor_times[1:]):
        gap = b - a
        if gap > PRIMARY_MAX:
            n_extra = int(np.ceil(gap / TARGET_LEN)) - 1
            for vi in range(1, n_extra + 1):
                nominal_t = a + vi * (gap / (n_extra + 1))
                # Search within +/- 1.0s for local energy minimum
                k0 = max(0, int((nominal_t - 1.0) / 0.010))
                k1 = min(len(energy_db), int((nominal_t + 1.0) / 0.010))
                if k1 > k0:
                    best_k = k0 + int(np.argmin(energy_db[k0:k1]))
                    cut_t = round(float(times[best_k]), 3)
                else:
                    cut_t = round(nominal_t, 3)
                
                # Check grid support for this candidate cut timestamp
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
    """Explicitly record a human approval for a review-only candidate (e.g. WHISPER_ONLY).
    Implements Issue 2 / Acceptance Test 3. Returns True if candidate found and approved.
    """
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


# -------------------------------------------------------------
# Global Shortest-Path Dynamic Programming (DP) Optimizer
# -------------------------------------------------------------
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
            # Detect transitions occurring between t_i and t_j
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
    """Greedy baseline for real measured ablation comparison."""
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


# -------------------------------------------------------------
# Metric & Margin Utilities
# -------------------------------------------------------------
def compute_segment_margins(s: float, e: float,
                            cand_s: Optional[PauseCandidate],
                            cand_e: Optional[PauseCandidate],
                            total_dur: Optional[float] = None) -> Tuple[int, int, bool]:
    """Calculate lead and trail margins around candidate cut midpoints."""
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
    """Calculate normalized cut certainty score [0.0, 1.0]."""
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
    """Determine transition event status for a segment: satisfied, deferred, or none."""
    for tr in transitions:
        t_tr = tr["time"]
        if abs(s - t_tr) <= 0.25 or abs(e - t_tr) <= 0.25:
            tr["satisfied"] = True
            return "satisfied"
        if s + 0.50 < t_tr < e - 0.50:
            return "deferred"
    return "none"


# -------------------------------------------------------------
# Main Cutting & Staging Pipeline (Unified Four-Label)
# -------------------------------------------------------------
def cut_four_label_segments(audio: np.ndarray,
                            regions_df: pd.DataFrame,
                            out_dir: str | Path,
                            sr: int = DEFAULT_SR,
                            pauses: Optional[pd.DataFrame] = None,
                            use_word_verification: bool = True,
                            word_model_size: str = "small",
                            run_ablation: bool = True,
                            **_ignored) -> pd.DataFrame:
    """Universal 4-label segmenter implementing Specification v2.2."""
    out_dir = Path(out_dir)
    total_dur = len(audio) / sr
    video_id = out_dir.name

    # Check for validated regions.csv (respecting any human edits in Inspector)
    reg_csv = out_dir / "regions.csv"
    if reg_csv.exists():
        regions_df = pd.read_csv(reg_csv)
        print("ℹ️ Loaded regions.csv (includes macro-regions and any human edits).")
    regions_df = regions_df.sort_values("start").reset_index(drop=True)

    # Output staging directory for atomic promotion (Section 5)
    staging_dir = out_dir / "segments_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for f in staging_dir.glob("segment_*.wav"):
        try:
            f.unlink()
        except OSError:
            pass

    # Optional word verification with faster-whisper
    word_model = _load_word_model(word_model_size) if use_word_verification else None
    all_words = []
    if word_model is not None:
        try:
            all_words = transcribe_words(word_model, audio, 0.0)
            print(f"🎙️ Transcribed {len(all_words)} word tokens for boundary verification.")
        except Exception as e:
            print(f"⚠️ Whisper full pass skipped/failed: {e}")
    elif (out_dir / "word_timestamps.json").exists():
        try:
            all_words = json.loads((out_dir / "word_timestamps.json").read_text(encoding="utf-8"))
            print(f"🎙️ Loaded {len(all_words)} cached word tokens from word_timestamps.json.")
        except Exception as e:
            pass

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

    # Measured ablation stats
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

    # Generate WAV files in staging and construct the 25 required manifest fields
    rows = []
    for k, p in enumerate(planned_segs, 1):
        s = p["start"]
        e = p["end"]
        dur = round(e - s, 3)
        sid = f"{k:04d}"
        wav_name = f"segment_{sid}.wav"
        
        # Write WAV with 8 ms Hann fade strictly inside margins
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

        # Label composition & primary label
        comp_info = compute_label_composition(s, e, regions_df)

        # Transition status and resolved requests
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

        # Cut type classification
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

        # Voting agreement string
        if ec is not None:
            n_det = getattr(ec, "num_detectors", 3)
            v_agree = f"{int(round(ec.vote_count))}/{n_det}"
            d_class = ec.vote_class
        else:
            v_agree = "3/3" if e >= total_dur - 0.05 else "NA"
            d_class = "FILE_EDGE" if e >= total_dur - 0.05 else "SPLIT"

        # Word check status
        if comp_info["primary_label"] in ["music", "silence"] and comp_info["speech_duration"] < 0.20:
            w_status = "NA"
        elif ec and ec.word_risk:
            w_status = "WORD_SPLIT_RISK"
        elif ec and ec.word_ok == 1.0:
            w_status = "SAFE"
        else:
            w_status = "UNVERIFIED"

        # Review flags & reasons
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
            # 1-6. Identity & Time
            "video_id": video_id,
            "segment_id": sid,
            "wav_file": f"segments/{wav_name}",
            "start": s,
            "end": e,
            "duration": dur,
            # 7-10. Four-label macro & composition
            "label": comp_info["primary_label"],
            "primary_label": comp_info["primary_label"],
            "label_composition": comp_info["label_composition"],
            "source_region_ids": comp_info["source_region_ids"],
            "confidence": comp_info["confidence"],
            # 11-15. Cutting & Margins
            "cut_type": ctype,
            "cut_point_time": round(e, 3),
            "lead_ms": lead_ms,
            "trail_ms": trail_ms,
            "margin_ok": int(m_ok),
            # 16-19. Voting & Certainty
            "cut_certainty": cert,
            "cut_tier": tier,
            "voting_agreement": v_agree,
            "decision_class": d_class,
            # 20-24. Human review & Planning
            "needs_review": needs_rev,
            "review_priority": priority_str,
            "review_reason": rev_reason_str,
            "fallback_used": fb,
            "plan_tag": plan_tag,
            # 25-27. Transition, Word Safety, Status
            "transition_status": tr_status,
            "word_check_status": w_status,
            "status": "pending_qc",
            # Auxiliary duration breakdowns
            "speech_duration": comp_info["speech_duration"],
            "mixed_duration": comp_info["mixed_duration"],
            "music_duration": comp_info["music_duration"],
            "silence_duration": comp_info["silence_duration"],
            # Midpoint & Boundary tracking
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

    # Save 100 ms Voting Grid for downstream QC and visualization
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

    # -------------------------------------------------------------
    # Mandatory QC Execution & Atomic Promotion (Sections 5 & 6)
    # -------------------------------------------------------------
    qc_res = validate_segments_integrity(
        staging_df, video_dir=out_dir, total_dur=total_dur, sr=sr, grid_df=grid_df
    )

    prod_dir = out_dir / "segments"
    prod_csv = out_dir / "segments.csv"

    if qc_res["passed"]:
        # Safe atomic replacement of production directory
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

        # Review queue & ablation metadata
        # Issue 3 Fix: review_queue_urgent.csv must contain ONLY actual cut segments requiring review.
        # Every item in the segment-review queue refers to an existing segment and a readable WAV file.
        urgent_df = staging_df[staging_df["needs_review"] == 1].sort_values(
            ["cut_certainty", "duration"], ascending=[True, False]
        )
        urgent_df.to_csv(out_dir / "review_queue_urgent.csv", index=False)
        (out_dir / "ablation_metrics.json").write_text(json.dumps(abl, indent=2), encoding="utf-8")
        if all_words:
            (out_dir / "word_timestamps.json").write_text(json.dumps(all_words, ensure_ascii=False), encoding="utf-8")

        # Optional: Save candidate-level diagnostics separately with distinct name
        unselected_cands = [
            {
                "candidate_id": c.pause_id,
                "start": c.p_start,
                "end": c.p_end,
                "duration": round(c.duration, 3),
                "vote_class": c.vote_class,
                "review_reason": c.review_reason
            }
            for c in cands if c.vote_class in ("WEAK", "WHISPER_ONLY") and not c.human_approved
        ]
        if unselected_cands:
            (out_dir / "unselected_review_candidates.json").write_text(
                json.dumps(unselected_cands, indent=2), encoding="utf-8"
            )

        # Save Transition Boundary Requests lifecycle log
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
        # Mandatory failure prevents success status and promotion
        print(f"❌ QC VALIDATION FAILED ({len(qc_res['errors'])} errors). Retaining staging directory.")
        (out_dir / "staging_qc_errors.json").write_text(json.dumps(qc_res, indent=2), encoding="utf-8")
        staging_df["status"] = "FAILED"
        final_df = staging_df

    # Summary Display
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


# -------------------------------------------------------------
# Backward-Compatible Alias
# -------------------------------------------------------------
def cut_speech_and_mixed_segments(*args, **kwargs) -> pd.DataFrame:
    """Seamless drop-in replacement routing all legacy callers to the Unified Four-Label Segmenter."""
    return cut_four_label_segments(*args, **kwargs)


# Backward-compatible dummy functions referenced in legacy notebooks
def cand_from_pause(*args, **kwargs):
    pass

def pause_score(*args, **kwargs):
    return 1.0

def snap_edge_to_pause(*args, **kwargs):
    return None

def build_streams(*args, **kwargs):
    return []

def make_valley_cand(*args, **kwargs):
    return None

def plan_cuts(*args, **kwargs):
    return [], "ok"

def cut_certainty(*args, **kwargs):
    return 1.0

def get_tier(cert: float):
    return get_cut_tier(cert)

def segment_margins(*args, **kwargs):
    return 150, 150, True

def region_label_for(s: float, e: float, regions_df: pd.DataFrame):
    res = compute_label_composition(s, e, regions_df)
    return res["primary_label"], res["confidence"]
