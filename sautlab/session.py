"""Headless annotation session controller built on top of `sautlab.backend.base.Backend`.

Implements all Step 8 human-override operations (boundary editing, split, merge,
add, delete, relabel, SNR quality override, transcript selection/editing, SNR
recomputation, undo/redo, and audit-log persistence) without requiring Qt, so
the same backend contract powers both the desktop PySide6 GUI and the Kaggle
browser UI.
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import numpy as np

from .backend.base import Backend
from .models import LABELS, LABEL_TITLES, STATUS_TITLES, Segment, VideoInfo, fmt_time

MIN_LEN = 0.1
SEAM_EPS = 1e-3


class AnnotationSession:
    """Stateful Step 8 annotation controller decoupled from Qt."""

    def __init__(self, backend: Backend, user: str = "Team"):
        self.backend = backend
        self.user = (user or "Team").strip() or "Team"
        self.snr_threshold = float(getattr(backend, "snr_threshold", 25.0))
        self.asr_names = tuple(getattr(backend, "asr_names", ("Whisper", "Audar")))
        self.videos: list[VideoInfo] = backend.list_videos()
        self.video: VideoInfo | None = None
        self.segments: list[Segment] = []
        self.log: list[dict] = []
        self.selected_id: int | None = None
        self.dirty = False
        self.last_saved: str = ""
        self._undo: list[list[Segment]] = []
        self._redo: list[list[Segment]] = []

    # ------------------------------------------------------------------
    # Lookups & counts
    # ------------------------------------------------------------------
    def seg(self, sid: int | None) -> Segment | None:
        if sid is None:
            return None
        for s in self.segments:
            if s.id == sid:
                return s
        return None

    def index_of(self, sid: int) -> int:
        for i, s in enumerate(self.segments):
            if s.id == sid:
                return i
        return -1

    def number_of(self, sid: int | None) -> int:
        if sid is None:
            return 0
        return self.index_of(sid) + 1

    @property
    def current(self) -> Segment | None:
        return self.seg(self.selected_id)

    def counts(self) -> dict[str, int]:
        c = {"accepted": 0, "needs_review": 0, "difficult": 0}
        for s in self.segments:
            c[s.status] = c.get(s.status, 0) + 1
        return c

    def conflict_ids(self) -> list[int]:
        return [s.id for s in self.segments if s.status in ("needs_review", "difficult")]

    # ------------------------------------------------------------------
    # Video loading & persistence
    # ------------------------------------------------------------------
    def refresh_videos(self) -> list[VideoInfo]:
        self.videos = self.backend.list_videos()
        if self.video is not None:
            self.video = next((v for v in self.videos if v.id == self.video.id), self.video)
        return self.videos

    def load_video(self, video_id: str, auto_save: bool = True) -> None:
        if auto_save and self.dirty and self.video is not None:
            self.save()
        self.refresh_videos()
        self.video = next(v for v in self.videos if v.id == video_id)
        self.segments = sorted(self.backend.load_segments(video_id), key=lambda s: s.start)
        self.log = self.backend.load_log(video_id)
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        self.last_saved = ""
        conflicts = self.conflict_ids()
        if conflicts:
            self.selected_id = conflicts[0]
        elif self.segments:
            self.selected_id = self.segments[0].id
        else:
            self.selected_id = None

    def save(self) -> None:
        if self.video is None:
            return
        self.backend.save_segments(self.video.id, self.segments, self.log)
        self.dirty = False
        self.last_saved = datetime.now().strftime("%H:%M:%S")
        self.refresh_videos()

    def select(self, sid: int | None) -> Segment | None:
        s = self.seg(sid)
        if s is not None:
            self.selected_id = s.id
        return self.current

    def select_relative(self, step: int) -> Segment | None:
        if not self.segments:
            self.selected_id = None
            return None
        cur = self.index_of(self.selected_id) if self.selected_id is not None else -1
        idx = min(max(cur + step, 0), len(self.segments) - 1)
        self.selected_id = self.segments[idx].id
        return self.current

    def select_next_conflict(self, step: int = 1) -> Segment | None:
        ids = self.conflict_ids()
        if not ids:
            return None
        cur = (
            self.index_of(self.selected_id)
            if self.selected_id is not None
            else (-1 if step > 0 else len(self.segments))
        )
        order = sorted((self.index_of(i), i) for i in ids)
        if step > 0:
            nxt = next((i for idx, i in order if idx > cur), order[0][1])
        else:
            nxt = next((i for idx, i in reversed(order) if idx < cur), order[-1][1])
        self.selected_id = nxt
        return self.current

    # ------------------------------------------------------------------
    # Undo / redo & audit logging
    # ------------------------------------------------------------------
    def _snapshot(self) -> list[Segment]:
        return copy.deepcopy(self.segments)

    def _push_undo(self, snap: list[Segment] | None = None) -> None:
        self._undo.append(snap if snap is not None else self._snapshot())
        if len(self._undo) > 200:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.segments = self._undo.pop()
        if self.seg(self.selected_id) is None:
            self.selected_id = self.segments[0].id if self.segments else None
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.segments = self._redo.pop()
        if self.seg(self.selected_id) is None:
            self.selected_id = self.segments[0].id if self.segments else None
        self.dirty = True
        return True

    def _log(self, sid: int | None, action: str, detail: str) -> None:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "user": self.user,
            "segment": self.number_of(sid) if sid else None,
            "segment_id": sid,
            "action": action,
            "detail": detail,
            "kind": "edit",
        }
        self.log.append(entry)
        s = self.seg(sid)
        if s is not None:
            s.changed_by = self.user
            s.changed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            s.change_log.append({"time": s.changed_at, "action": action, "detail": detail})
        self.dirty = True

    # ------------------------------------------------------------------
    # Label / SNR / Transcript overrides
    # ------------------------------------------------------------------
    def set_label(self, sid: int, label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"Invalid label '{label}'. Must be one of {LABELS}.")
        s = self.seg(sid)
        if s is None or s.label == label:
            return
        self._push_undo()
        old = s.label
        s.label = label
        if label in ("music", "silence"):
            s.quality = None
            s.status = "accepted"
            s.decided_by = "human"
        elif old in ("music", "silence"):
            s.status = "needs_review"
            s.decided_by = "human"
        self._log(s.id, "label", f"Label changed from {LABEL_TITLES[old]} to {LABEL_TITLES[label]}")

    def set_status(self, sid: int, status: str) -> None:
        if status not in STATUS_TITLES:
            raise ValueError(f"Invalid status '{status}'.")
        s = self.seg(sid)
        if s is None or s.status == status:
            return
        self._push_undo()
        old = s.status
        s.status = status
        s.decided_by = "human"
        self._log(s.id, "status", f"Status changed from {STATUS_TITLES[old]} to {STATUS_TITLES[status]}")

    def set_quality(self, sid: int, quality: str | None) -> None:
        s = self.seg(sid)
        if s is None:
            return
        new = quality if quality in ("clean", "noisy") else s.quality_auto
        if new == s.quality:
            return
        self._push_undo()
        old = s.quality
        s.quality = new
        self._log(s.id, "quality", f"Clean/Noisy changed from {old or '-'} to {new or '-'}")

    def choose_transcript(self, sid: int, engine: str) -> None:
        s = self.seg(sid)
        if s is None:
            return
        text = s.asr_a if engine == "a" else s.asr_b
        self._push_undo()
        s.final_text = text
        s.status = "accepted"
        s.decided_by = engine
        name = self.asr_names[0] if engine == "a" else self.asr_names[1]
        self._log(sid, "transcript", f"Chose {name} transcript")

    def set_final_text(self, sid: int, text: str) -> None:
        s = self.seg(sid)
        if s is None or text == s.final_text:
            return
        self._push_undo()
        s.final_text = text
        if text.strip():
            s.status = "accepted"
        s.decided_by = "edited"
        self._log(sid, "transcript", "Final transcript edited by hand")

    def set_notes(self, sid: int, text: str) -> None:
        s = self.seg(sid)
        if s is None or text == s.notes:
            return
        self._push_undo()
        s.notes = text
        self._log(sid, "notes", "Annotator notes updated")

    def apply_snr_threshold(self, threshold: float) -> int:
        self.snr_threshold = float(threshold)
        self.backend.snr_threshold = self.snr_threshold
        changed = 0
        snap = self._snapshot()
        for s in self.segments:
            if s.snr_mean is None:
                continue
            new_auto = "clean" if s.snr_mean >= self.snr_threshold else "noisy"
            if s.quality == s.quality_auto:
                s.quality = new_auto
            if new_auto != s.quality_auto:
                changed += 1
            s.quality_auto = new_auto
        if changed:
            self._push_undo(snap)
            self._log(None, "snr", f"Clean/Noisy threshold set to {self.snr_threshold:g} dB ({changed} segments updated)")
        return changed

    def reprocess_segment(self, sid: int) -> Segment | None:
        s = self.seg(sid)
        if s is None or self.video is None:
            return None
        new = self.backend.reprocess_segment(self.video.id, copy.deepcopy(s))
        self._push_undo()
        for name in (
            "snr", "snr_energy", "snr_wada", "quality", "quality_auto",
            "asr_a", "asr_b", "final_text", "status", "decided_by", "confidence",
        ):
            setattr(s, name, getattr(new, name))
        self._log(sid, "reprocess", "SNR re-computed for edited segment")
        return s

    # ------------------------------------------------------------------
    # Cut & structural edits (move / split / merge / add / delete)
    # ------------------------------------------------------------------
    def _neighbors(self, s: Segment) -> tuple[Segment | None, Segment | None]:
        i = self.index_of(s.id)
        prev = self.segments[i - 1] if i > 0 else None
        nxt = self.segments[i + 1] if i + 1 < len(self.segments) else None
        return prev, nxt

    def _stale_check(self, s: Segment) -> None:
        if s.has_text and (s.asr_a or s.asr_b or s.final_text) and s.status == "accepted":
            s.status = "needs_review"
            s.decided_by = "human"

    def set_times(self, sid: int, start: float, end: float, sync_neighbors: bool = True) -> None:
        s = self.seg(sid)
        if s is None:
            return
        dur = self.video.duration if self.video else 1e9
        prev, nxt = self._neighbors(s)

        new_start = round(float(start), 3)
        new_end = round(float(end), 3)
        if abs(new_start - s.start) <= 1e-6 and abs(new_end - s.end) <= 1e-6:
            return

        snap = self._snapshot()
        shared_prev = prev if (sync_neighbors and prev and abs(prev.end - s.start) < SEAM_EPS) else None
        shared_nxt = nxt if (sync_neighbors and nxt and abs(nxt.start - s.end) < SEAM_EPS) else None

        lo = (shared_prev.start + MIN_LEN) if shared_prev else (prev.end if prev else 0.0)
        hi = (shared_nxt.end - MIN_LEN) if shared_nxt else (nxt.start if nxt else dur)
        hi = min(hi, dur)

        new_start = min(max(new_start, lo), hi - MIN_LEN)
        new_end = min(max(new_end, new_start + MIN_LEN), hi)
        new_start = round(new_start, 3)
        new_end = round(new_end, 3)

        if abs(new_start - s.start) <= 1e-6 and abs(new_end - s.end) <= 1e-6:
            return

        old = (s.start, s.end)
        s.start = new_start
        s.end = new_end
        if shared_prev is not None:
            shared_prev.end = new_start
            if abs(new_start - old[0]) > 1e-6:
                self._stale_check(shared_prev)
        if shared_nxt is not None:
            shared_nxt.start = new_end
            if abs(new_end - old[1]) > 1e-6:
                self._stale_check(shared_nxt)

        self._push_undo(snap)
        self._stale_check(s)
        self._log(
            sid,
            "time",
            f"Time changed from {fmt_time(old[0], True)}-{fmt_time(old[1], True)} "
            f"to {fmt_time(s.start, True)}-{fmt_time(s.end, True)}",
        )

    def split(self, sid: int, t: float | None = None) -> int | None:
        s = self.seg(sid)
        if s is None:
            return None
        if t is None or not (s.start + MIN_LEN < float(t) < s.end - MIN_LEN):
            t = (s.start + s.end) / 2.0
        t = round(float(t), 3)
        self._push_undo()
        new_id = max(x.id for x in self.segments) + 1
        right = copy.deepcopy(s)
        right.id = new_id
        right.start = t
        s.end = t
        for part in (s, right):
            if part.has_text:
                part.status = "needs_review"
                part.decided_by = "human"
        self.segments.insert(self.index_of(sid) + 1, right)
        self._log(sid, "split", f"Segment split at {fmt_time(t, True)}")
        self.selected_id = sid
        return new_id

    def merge(self, sid: int, direction: str = "next") -> bool:
        idx = self.index_of(sid)
        if idx < 0:
            return False
        if direction == "prev":
            if idx == 0:
                return False
            chosen = [self.segments[idx - 1], self.segments[idx]]
        else:
            if idx + 1 >= len(self.segments):
                return False
            chosen = [self.segments[idx], self.segments[idx + 1]]

        self._push_undo()
        first, last = chosen[0], chosen[-1]
        total = sum(s.duration for s in chosen)

        def wavg(attr: str) -> float | None:
            vals = [(getattr(s, attr), s.duration) for s in chosen if getattr(s, attr) is not None]
            if not vals:
                return None
            return round(sum(v * d for v, d in vals) / sum(d for _, d in vals), 1)

        longest = max(chosen, key=lambda s: s.duration)
        first.label = longest.label
        first.confidence = round(min(s.confidence for s in chosen), 2)
        first.snr_energy = wavg("snr_energy")
        first.snr_wada = wavg("snr_wada")
        first.snr = wavg("snr")
        first.quality = "noisy" if any(s.quality == "noisy" for s in chosen) else first.quality
        first.quality_auto = first.quality
        for attr in ("asr_a", "asr_b", "final_text"):
            setattr(first, attr, " ".join(getattr(s, attr) for s in chosen if getattr(s, attr)).strip())
        rank = {"accepted": 0, "needs_review": 1, "difficult": 2}
        first.status = max((s.status for s in chosen), key=lambda x: rank.get(x, 1))
        if first.has_text and not first.final_text:
            first.status = "needs_review" if first.status == "accepted" else first.status
        first.decided_by = "human"
        first.end = last.end
        drop_id = last.id
        self.segments = [s for s in self.segments if s.id != drop_id]
        note = f" ({total:.1f} s)" if total > 15 and first.has_text else ""
        self._log(first.id, "merge", f"Merged 2 segments{note}")
        self.selected_id = first.id
        return True

    def delete(self, sid: int) -> bool:
        idx = self.index_of(sid)
        if idx < 0:
            return False
        self._push_undo()
        self._log(sid, "delete", "Segment deleted")
        self.segments = [s for s in self.segments if s.id != sid]
        if self.segments:
            self.selected_id = self.segments[min(idx, len(self.segments) - 1)].id
        else:
            self.selected_id = None
        return True

    def add(self, start: float, end: float | None = None, label: str = "speech") -> int | None:
        if self.video is None or label not in LABELS:
            return None
        dur = self.video.duration or (max((s.end for s in self.segments), default=10.0) + 10.0)
        start = min(max(float(start), 0.0), max(0.0, dur - MIN_LEN))
        if end is None:
            end = start + 5.0
        end = min(float(end), dur)
        for s in self.segments:
            if s.start <= start < s.end:
                start = s.end
        nxt = [s.start for s in self.segments if s.start >= start]
        if nxt:
            end = min(end, min(nxt))
        if end - start < MIN_LEN:
            return None
        self._push_undo()
        new_id = max((x.id for x in self.segments), default=0) + 1
        seg = Segment(id=new_id, start=round(start, 3), end=round(end, 3), label=label, confidence=1.0)
        seg.status = "accepted" if label in ("music", "silence") else "needs_review"
        seg.decided_by = "human"
        self.segments.append(seg)
        self.segments.sort(key=lambda s: s.start)
        self._log(new_id, "add", f"Segment added at {fmt_time(start, True)}")
        self.selected_id = new_id
        return new_id

    # ------------------------------------------------------------------
    # Audio slicing & visual inspection helpers
    # ------------------------------------------------------------------
    def slice_audio(self, start: float, end: float) -> tuple[np.ndarray, int]:
        if self.video is None:
            return np.zeros(0, dtype=np.float32), 16000
        samples, sr = self.backend.load_audio(self.video.id)
        a = max(0, int(round(float(start) * sr)))
        b = min(len(samples), int(round(float(end) * sr)))
        if b <= a:
            return np.zeros(int(0.1 * sr), dtype=np.float32), sr
        return samples[a:b], sr

    def render_segment_plot(self, sid: int, out_png: str | Path, context_sec: float = 2.5) -> str | None:
        s = self.seg(sid)
        if s is None or self.video is None:
            return None
        import matplotlib.pyplot as plt
        from scipy import signal

        samples, sr = self.backend.load_audio(self.video.id)
        total_dur = len(samples) / sr
        t0 = max(0.0, s.start - context_sec)
        t1 = min(total_dur, s.end + context_sec)
        if t1 <= t0:
            return None
        i0, i1 = int(t0 * sr), int(t1 * sr)
        clip = samples[i0:i1]
        if len(clip) < 64:
            return None

        colors = {"speech": "#22c55e", "music": "#3b82f6", "mixed": "#f59e0b", "silence": "#94a3b8"}
        fig, (ax_w, ax_s) = plt.subplots(
            2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.4]}
        )
        t_ax = np.linspace(t0, t1, len(clip))
        ax_w.plot(t_ax, clip, color="#38bdf8", lw=0.75)

        for seg in self.segments:
            if seg.end < t0 or seg.start > t1:
                continue
            a, b = max(t0, seg.start), min(t1, seg.end)
            col = colors.get(seg.label, "#94a3b8")
            is_sel = seg.id == s.id
            ax_w.axvspan(a, b, color=col, alpha=0.28 if is_sel else 0.12)
            ax_w.axvline(seg.start, color="#ef4444" if is_sel else "#64748b", ls="-" if is_sel else "--", lw=1.6 if is_sel else 0.8)
            ax_w.axvline(seg.end, color="#ef4444" if is_sel else "#64748b", ls="-" if is_sel else "--", lw=1.6 if is_sel else 0.8)

        ax_w.set_ylabel("Amplitude")
        ax_w.set_title(
            f"Segment #{self.number_of(s.id)} [{s.label.upper()}]  {fmt_time(s.start, True)} - {fmt_time(s.end, True)} ({s.duration:.2f}s)",
            fontsize=11,
            fontweight="bold",
        )
        ax_w.grid(True, ls=":", alpha=0.35)

        nperseg = min(400, len(clip))
        noverlap = min(240, max(0, nperseg - 16))
        f, tt, S = signal.spectrogram(clip, fs=sr, nperseg=nperseg, noverlap=noverlap, nfft=512, mode="psd")
        S_db = 10.0 * np.log10(S + 1e-12)
        ax_s.imshow(
            S_db,
            origin="lower",
            aspect="auto",
            extent=[t0, t0 + len(tt) * max(1, (nperseg - noverlap)) / sr, 0, sr / 2000.0],
            vmin=S_db.max() - 75.0,
            vmax=S_db.max(),
            cmap="magma",
        )
        ax_s.axvline(s.start, color="#ef4444", lw=1.8)
        ax_s.axvline(s.end, color="#ef4444", lw=1.8)
        ax_s.set_xlim(t0, t1)
        ax_s.set_xlabel("Time (s)")
        ax_s.set_ylabel("Freq (kHz)")

        fig.tight_layout()
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        return str(out_png)
