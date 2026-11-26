"""The review CSV of the speech-corpus backend (the 16 "app" columns).

Written by the backend after SNR / dual ASR and read and written by the GUI.
Both CSV variants the backend produces are understood when reading:
  - cut-engine / SNR output (wav_file or audio_file, no transcripts yet)
  - ASR output with model_A_transcript / model_B_transcript

Backend statuses: accepted | human_choice | difficult | no_asr | not_run
GUI statuses:     accepted | needs_review | difficult
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import TEXT_LABELS, Segment

APP_COLS = [
    "start", "end", "label", "snr", "quality",
    "model_A_transcript", "model_B_transcript", "final_transcript",
    "status", "review_status", "automatic_status",
    "segment_id", "audio_file",
    "changed_by", "changed_at", "change_log",
]

VALID_LABELS = ("speech", "music", "mixed", "silence")


def audio_name(index: int) -> str:
    # index is 1-based, same naming as the segment editor of the backend
    return f"segments/segment_{index:04d}.wav"


def status_to_csv(s: Segment) -> tuple[str, str]:
    """(status, review_status) as the backend writes them."""
    if s.label not in TEXT_LABELS:
        if s.status == "difficult":
            return "difficult", "pending"
        return "no_asr", "skipped"
    if s.status == "accepted":
        return "accepted", "accepted"
    if s.status == "difficult":
        return "difficult", "pending"
    has_asr = bool(s.asr_a.strip() or s.asr_b.strip())
    return ("human_choice" if has_asr else "not_run"), "pending"


def status_from_csv(status: str, label: str, has_asr: bool, needs_flag: bool) -> str:
    st = (status or "").strip().lower()
    text = label in TEXT_LABELS
    if not text:
        return "difficult" if st == "difficult" else "accepted"
    if st in ("accepted", "verified"):
        return "accepted"
    if st == "difficult":
        return "difficult"
    if st in ("human_choice", "needs_review", "pending", "not_run"):
        return "needs_review"
    # unknown or empty: judge from what is known
    if needs_flag or not has_asr:
        return "needs_review"
    return "accepted"


def _num(v):
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("nan", "none"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _flag(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "1.0")


def read_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_review_csv(path: str | Path) -> list[Segment]:
    rows = read_rows(path)
    segs: list[Segment] = []
    for row in rows:
        start, end = _num(row.get("start")), _num(row.get("end"))
        if start is None or end is None or end <= start:
            continue
        label = (row.get("label") or row.get("primary_label") or row.get("source_region_label") or "speech").strip().lower()
        if label not in VALID_LABELS:
            label = "speech"
        snr = _num(row.get("snr"))
        if snr is None:
            snr = _num(row.get("snr_db"))
        quality = (row.get("quality") or row.get("clean_noisy") or "").strip().lower() or None
        if quality not in ("clean", "noisy"):
            quality = None
        a = row.get("model_A_transcript") or ""
        b = row.get("model_B_transcript") or ""
        final = row.get("final_transcript") or ""
        conf = _num(row.get("confidence"))
        try:
            log = json.loads(row.get("change_log") or "[]")
            if not isinstance(log, list):
                log = []
        except ValueError:
            log = []
        s = Segment(
            id=len(segs) + 1,
            start=round(start, 3),
            end=round(end, 3),
            label=label,
            confidence=round(conf, 3) if conf is not None else 1.0,
            snr=snr,
            snr_energy=_num(row.get("snr_energy_db")),
            snr_wada=_num(row.get("snr_wada_db")),
            quality=quality,
            quality_auto=quality,
            asr_a=a,
            asr_b=b,
            final_text=final,
            auto_status=(row.get("automatic_status") or "").strip(),
            changed_by=row.get("changed_by") or "",
            changed_at=row.get("changed_at") or "",
            change_log=log,
        )
        s.status = status_from_csv(row.get("status", ""), label, bool(a.strip() or b.strip()), _flag(row.get("needs_review", "")))
        s.decided_by = "edited" if s.changed_by else "auto"
        segs.append(s)
    segs.sort(key=lambda x: x.start)
    for i, s in enumerate(segs, 1):
        s.id = i
    return segs


def review_row(s: Segment, index: int) -> dict:
    status, review = status_to_csv(s)
    return {
        "start": f"{s.start:.3f}",
        "end": f"{s.end:.3f}",
        "label": s.label,
        "snr": "" if s.snr_mean is None else f"{s.snr_mean:.2f}",
        "quality": s.quality or "",
        "model_A_transcript": s.asr_a,
        "model_B_transcript": s.asr_b,
        "final_transcript": s.final_text,
        "status": status,
        "review_status": review,
        "automatic_status": s.auto_status or status,
        "segment_id": f"{index:04d}",
        "audio_file": audio_name(index),
        "changed_by": s.changed_by,
        "changed_at": s.changed_at,
        "change_log": json.dumps(s.change_log, ensure_ascii=False),
    }


def write_review_csv(path: str | Path, segments: list[Segment], extra_cols: dict | None = None) -> None:
    """Writes the 16 app columns in order. `extra_cols` maps a column name to a function Segment -> value
    and adds columns after the app columns (used for segments_full.csv)."""
    ordered = sorted(segments, key=lambda x: x.start)
    cols = list(APP_COLS) + list((extra_cols or {}).keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, s in enumerate(ordered, 1):
            row = review_row(s, i)
            for name, fn in (extra_cols or {}).items():
                v = fn(s)
                row[name] = "" if v is None else v
            w.writerow(row)
