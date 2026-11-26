"""Holds the project state shown by the GUI and applies every annotator edit.

All edits go through this class so that undo/redo, the change log and the
dirty flag stay consistent. Widgets only read state and call these methods.
"""
from __future__ import annotations

import copy
from datetime import datetime

from PySide6.QtCore import QObject, QTimer, Signal

from .backend.base import Backend
from .models import LABEL_TITLES, STATUS_TITLES, Segment, VideoInfo, fmt_time

MIN_LEN = 0.1
SEAM_EPS = 1e-3


class ProjectController(QObject):
    videosChanged = Signal()
    videoLoaded = Signal(str)
    structureChanged = Signal()  # rows added/removed/replaced
    segmentChanged = Signal(int)  # values of one segment changed
    selectionChanged = Signal()
    logAdded = Signal(dict)
    savedStateChanged = Signal()
    undoStateChanged = Signal()
    pipelineChanged = Signal()

    def __init__(self, backend: Backend, user: str = "Mohamed Ahmed"):
        super().__init__()
        self.backend = backend
        self.user = user
        self.snr_threshold = float(getattr(backend, "snr_threshold", 25.0))
        self.asr_names = tuple(getattr(backend, "asr_names", ("Whisper", "Audar")))
        self.videos: list[VideoInfo] = backend.list_videos()
        self.video: VideoInfo | None = None
        self.segments: list[Segment] = []
        self.log: list[dict] = []
        self.selected: list[int] = []
        self.dirty = False
        self.last_saved: str = ""
        self._undo: list[list[Segment]] = []
        self._redo: list[list[Segment]] = []
        self._drag_before: list[Segment] | None = None
        self._drag_info: tuple[int, str, float] | None = None

        self._autosave = QTimer(self)
        self._autosave.setInterval(30_000)
        self._autosave.timeout.connect(self._autosave_tick)
        self._autosave.start()

    # lookups
    def seg(self, sid: int) -> Segment | None:
        for s in self.segments:
            if s.id == sid:
                return s
        return None

    def index_of(self, sid: int) -> int:
        for i, s in enumerate(self.segments):
            if s.id == sid:
                return i
        return -1

    def number_of(self, sid: int) -> int:
        return self.index_of(sid) + 1

    @property
    def current(self) -> Segment | None:
        return self.seg(self.selected[0]) if self.selected else None

    def segment_at(self, t: float) -> Segment | None:
        for s in self.segments:
            if s.start <= t < s.end:
                return s
        return None

    def counts(self) -> dict[str, int]:
        c = {"accepted": 0, "needs_review": 0, "difficult": 0}
        for s in self.segments:
            c[s.status] = c.get(s.status, 0) + 1
        return c

    def conflict_ids(self) -> list[int]:
        return [s.id for s in self.segments if s.status in ("needs_review", "difficult")]

    # video handling
    def load_video(self, video_id: str) -> None:
        if self.dirty:
            self.save()
        self.video = next(v for v in self.videos if v.id == video_id)
        self.segments = sorted(self.backend.load_segments(video_id), key=lambda s: s.start)
        self.log = self.backend.load_log(video_id)
        self.selected = []
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        self.last_saved = ""
        self.videoLoaded.emit(video_id)
        self.structureChanged.emit()
        self.selectionChanged.emit()
        self.undoStateChanged.emit()
        self.savedStateChanged.emit()
        for entry in self.log[-30:]:
            self.logAdded.emit(entry)

    def switch_backend(self, backend: Backend) -> None:
        if self.dirty:
            self.save()
        self.backend = backend
        self.snr_threshold = float(getattr(backend, "snr_threshold", self.snr_threshold))
        self.asr_names = tuple(getattr(backend, "asr_names", self.asr_names))
        self.videos = backend.list_videos()
        self.video = None
        self.segments = []
        self.log = []
        self.selected = []
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        self.videosChanged.emit()
        self.structureChanged.emit()
        self.selectionChanged.emit()
        self.undoStateChanged.emit()
        self.savedStateChanged.emit()

    def refresh_videos(self) -> None:
        self.videos = self.backend.list_videos()
        self.videosChanged.emit()

    def replace_segments(self, segments: list[Segment]) -> None:
        self.segments = sorted(segments, key=lambda s: s.start)
        self.selected = []
        self._undo.clear()
        self._redo.clear()
        self.dirty = True
        self.structureChanged.emit()
        self.selectionChanged.emit()
        self.undoStateChanged.emit()
        self.savedStateChanged.emit()

    # saving
    def save(self) -> None:
        if not self.video:
            return
        self.backend.save_segments(self.video.id, self.segments, self.log)
        self.dirty = False
        self.last_saved = datetime.now().strftime("%H:%M:%S")
        self.refresh_videos()
        self.savedStateChanged.emit()

    def _autosave_tick(self) -> None:
        if self.dirty:
            self.save()

    # selection
    def set_selection(self, ids: list[int]) -> None:
        ids = [i for i in ids if self.seg(i)]
        if ids == self.selected:
            return
        self.selected = ids
        self.selectionChanged.emit()

    def select_relative(self, step: int) -> None:
        if not self.segments:
            return
        i = self.index_of(self.selected[0]) if self.selected else -1
        i = min(max(i + step, 0), len(self.segments) - 1)
        self.set_selection([self.segments[i].id])

    def select_next_conflict(self, step: int = 1) -> bool:
        ids = self.conflict_ids()
        if not ids:
            return False
        cur = self.index_of(self.selected[0]) if self.selected else (-1 if step > 0 else len(self.segments))
        order = [(self.index_of(i), i) for i in ids]
        order.sort()
        if step > 0:
            nxt = next((i for idx, i in order if idx > cur), order[0][1])
        else:
            nxt = next((i for idx, i in reversed(order) if idx < cur), order[-1][1])
        self.set_selection([nxt])
        return True

    # undo / redo
    def _snapshot(self) -> list[Segment]:
        return copy.deepcopy(self.segments)

    def _push_undo(self, snap: list[Segment] | None = None) -> None:
        self._undo.append(snap if snap is not None else self._snapshot())
        if len(self._undo) > 200:
            self._undo.pop(0)
        self._redo.clear()
        self.undoStateChanged.emit()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._snapshot())
        self.segments = self._undo.pop()
        self._after_restore("Undo")

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._snapshot())
        self.segments = self._redo.pop()
        self._after_restore("Redo")

    def _after_restore(self, what: str) -> None:
        self.selected = [i for i in self.selected if self.seg(i)]
        self._mark_dirty()
        self.structureChanged.emit()
        self.selectionChanged.emit()
        self.undoStateChanged.emit()
        self._log(None, what, f"{what} applied", kind="info", persist=False)

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.savedStateChanged.emit()

    # change log
    def _log(self, sid: int | None, action: str, detail: str, kind: str = "edit", persist: bool = True) -> None:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "user": self.user,
            "segment": self.number_of(sid) if sid else None,
            "segment_id": sid,
            "action": action,
            "detail": detail,
            "kind": kind,
        }
        if persist:
            self.log.append(entry)
            s = self.seg(sid) if sid else None
            if s is not None:
                s.changed_by = self.user
                s.changed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                s.change_log.append({"time": s.changed_at, "action": action, "detail": detail})
        self.logAdded.emit(entry)

    def log_info(self, message: str, kind: str = "ok") -> None:
        self._log(None, "info", message, kind=kind, persist=False)

    def _touch(self, sid: int) -> None:
        self._mark_dirty()
        self.segmentChanged.emit(sid)

    # field edits
    def set_label(self, ids: list[int], label: str) -> None:
        targets = [s for s in (self.seg(i) for i in ids) if s and s.label != label]
        if not targets:
            return
        self._push_undo()
        for s in targets:
            old = s.label
            s.label = label
            if label in ("music", "silence"):
                s.quality = None
                s.status = "accepted"
                s.decided_by = "human"
            elif old in ("music", "silence"):
                # needs transcripts and SNR: let the annotator re-run ASR
                s.status = "needs_review"
                s.decided_by = "human"
            self._log(
                s.id,
                "label",
                f"Label changed from {LABEL_TITLES[old]} to {LABEL_TITLES[label]}",
            )
            self._touch(s.id)

    def set_status(self, ids: list[int], status: str) -> None:
        targets = [s for s in (self.seg(i) for i in ids) if s and s.status != status]
        if not targets:
            return
        self._push_undo()
        for s in targets:
            old = s.status
            s.status = status
            self._log(s.id, "status", f"Status changed from {STATUS_TITLES[old]} to {STATUS_TITLES[status]}")
            self._touch(s.id)

    def set_quality(self, ids: list[int], quality: str | None) -> None:
        # quality None means "back to the automatic value"
        targets = [s for s in (self.seg(i) for i in ids) if s and s.has_text]
        if not targets:
            return
        self._push_undo()
        for s in targets:
            new = quality if quality else s.quality_auto
            if new == s.quality:
                continue
            old = s.quality
            s.quality = new
            self._log(s.id, "quality", f"Clean/Noisy changed from {old or '-'} to {new or '-'}")
            self._touch(s.id)

    def choose_transcript(self, sid: int, engine: str) -> None:
        s = self.seg(sid)
        if not s:
            return
        text = s.asr_a if engine == "a" else s.asr_b
        self._push_undo()
        s.final_text = text
        s.status = "accepted"
        s.decided_by = engine
        name = self.asr_names[0] if engine == "a" else self.asr_names[1]
        self._log(sid, "transcript", f"Chose {name} transcript")
        self._touch(sid)

    def set_final_text(self, sid: int, text: str) -> None:
        s = self.seg(sid)
        if not s or text == s.final_text:
            return
        self._push_undo()
        s.final_text = text
        s.status = "accepted" if text.strip() else s.status
        s.decided_by = "edited"
        self._log(sid, "transcript", "Final transcript edited by hand")
        self._touch(sid)

    def set_notes(self, sid: int, text: str) -> None:
        s = self.seg(sid)
        if not s or text == s.notes:
            return
        self._push_undo()
        s.notes = text
        self._log(sid, "notes", "Annotator notes updated")
        self._touch(sid)

    # time edits
    def _neighbors(self, s: Segment) -> tuple[Segment | None, Segment | None]:
        i = self.index_of(s.id)
        prev = self.segments[i - 1] if i > 0 else None
        nxt = self.segments[i + 1] if i + 1 < len(self.segments) else None
        return prev, nxt

    def _move_boundary(self, sid: int, which: str, t: float) -> None:
        s = self.seg(sid)
        if not s:
            return
        prev, nxt = self._neighbors(s)
        dur = self.video.duration if self.video else 1e9
        if which == "start":
            shared = prev if prev and abs(prev.end - s.start) < SEAM_EPS else None
            lo = (shared.start + MIN_LEN) if shared else (prev.end if prev else 0.0)
            hi = s.end - MIN_LEN
            t = min(max(t, lo), hi)
            s.start = round(t, 3)
            if shared:
                shared.end = s.start
                self.segmentChanged.emit(shared.id)
        else:
            shared = nxt if nxt and abs(nxt.start - s.end) < SEAM_EPS else None
            hi = (shared.end - MIN_LEN) if shared else (nxt.start if nxt else dur)
            lo = s.start + MIN_LEN
            t = min(max(t, lo), min(hi, dur))
            s.end = round(t, 3)
            if shared:
                shared.start = s.end
                self.segmentChanged.emit(shared.id)

    def _stale_check(self, s: Segment) -> None:
        # the transcripts were made for the old cut, so a human has to confirm them again
        if s.has_text and (s.asr_a or s.asr_b or s.final_text) and s.status == "accepted":
            s.status = "needs_review"
            s.decided_by = "human"

    def set_times(self, sid: int, start: float | None = None, end: float | None = None) -> None:
        s = self.seg(sid)
        if not s:
            return
        old = (s.start, s.end)
        snap = self._snapshot()
        if start is not None and abs(start - s.start) > 1e-6:
            self._move_boundary(sid, "start", start)
        if end is not None and abs(end - s.end) > 1e-6:
            self._move_boundary(sid, "end", end)
        if (s.start, s.end) == old:
            self.segmentChanged.emit(sid)  # lets spin boxes snap back
            return
        self._push_undo(snap)
        self._stale_check(s)
        self._log(
            sid,
            "time",
            f"Time changed from {fmt_time(old[0], True)}-{fmt_time(old[1], True)} "
            f"to {fmt_time(s.start, True)}-{fmt_time(s.end, True)}",
        )
        self._touch(sid)

    # dragging boundaries with the mouse (one undo step per drag)
    def begin_drag(self, sid: int, which: str) -> None:
        s = self.seg(sid)
        if not s:
            return
        self._drag_before = self._snapshot()
        self._drag_info = (sid, which, s.start if which == "start" else s.end)

    def drag_boundary(self, sid: int, which: str, t: float) -> None:
        self._move_boundary(sid, which, t)
        self._mark_dirty()
        self.segmentChanged.emit(sid)

    def end_drag(self) -> None:
        if self._drag_before is None or self._drag_info is None:
            return
        sid, which, old = self._drag_info
        s = self.seg(sid)
        new = (s.start if which == "start" else s.end) if s else old
        if s and abs(new - old) > 1e-6:
            self._push_undo(self._drag_before)
            self._stale_check(s)
            prev, nxt = self._neighbors(s)
            for nb in (prev, nxt):
                if nb is not None and (abs(nb.end - s.start) < SEAM_EPS or abs(nb.start - s.end) < SEAM_EPS):
                    self._stale_check(nb)
            self._log(
                sid,
                "time",
                f"Cut moved: {which} {fmt_time(old, True)} to {fmt_time(new, True)}",
            )
        self._drag_before = None
        self._drag_info = None

    # structure edits
    def split(self, sid: int, t: float | None = None) -> int | None:
        s = self.seg(sid)
        if not s:
            return None
        if t is None or not (s.start + MIN_LEN < t < s.end - MIN_LEN):
            t = (s.start + s.end) / 2
        self._push_undo()
        new_id = max(x.id for x in self.segments) + 1
        right = copy.deepcopy(s)
        right.id = new_id
        right.start = round(t, 3)
        s.end = round(t, 3)
        for part in (s, right):
            if part.has_text:
                part.status = "needs_review"
                part.decided_by = "human"
        self.segments.insert(self.index_of(sid) + 1, right)
        self._log(sid, "split", f"Segment split at {fmt_time(t, True)}")
        self._mark_dirty()
        self.structureChanged.emit()
        self.set_selection([sid])
        return new_id

    def merge(self, ids: list[int]) -> bool:
        chosen = sorted((s for s in (self.seg(i) for i in ids) if s), key=lambda s: s.start)
        if len(chosen) < 2:
            if len(chosen) == 1:
                i = self.index_of(chosen[0].id)
                if i + 1 < len(self.segments):
                    chosen.append(self.segments[i + 1])
            if len(chosen) < 2:
                return False
        self._push_undo()
        first = chosen[0]
        last = chosen[-1]
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
        first.quality = "noisy" if any(s.quality == "noisy" for s in chosen) else first.quality
        first.quality_auto = first.quality
        for attr in ("asr_a", "asr_b", "final_text"):
            setattr(first, attr, " ".join(getattr(s, attr) for s in chosen if getattr(s, attr)).strip())
        rank = {"accepted": 0, "needs_review": 1, "difficult": 2}
        first.status = max((s.status for s in chosen), key=lambda x: rank[x])
        if first.has_text and not first.final_text:
            first.status = "needs_review" if first.status == "accepted" else first.status
        first.decided_by = "human"
        first.end = last.end
        drop = {s.id for s in chosen[1:]}
        self.segments = [s for s in self.segments if s.id not in drop]
        note = f" ({total:.1f} s)" if total > 15 and first.has_text else ""
        self._log(first.id, "merge", f"Merged {len(chosen)} segments{note}")
        self._mark_dirty()
        self.structureChanged.emit()
        self.set_selection([first.id])
        return True

    def delete(self, ids: list[int]) -> None:
        drop = {i for i in ids if self.seg(i)}
        if not drop:
            return
        self._push_undo()
        first = min(self.index_of(i) for i in drop)
        for i in sorted(drop, key=self.index_of):
            self._log(i, "delete", "Segment deleted")
        self.segments = [s for s in self.segments if s.id not in drop]
        self.selected = []
        self._mark_dirty()
        self.structureChanged.emit()
        if self.segments:
            self.set_selection([self.segments[min(first, len(self.segments) - 1)].id])
        else:
            self.selectionChanged.emit()

    def add(self, start: float, end: float | None = None, label: str = "speech") -> int | None:
        if not self.video:
            return None
        dur = self.video.duration
        start = min(max(start, 0.0), dur - MIN_LEN)
        if end is None:
            end = start + 5.0
        end = min(end, dur)
        # keep clear of neighbours
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
        seg.status = "needs_review"
        seg.decided_by = "human"
        self.segments.append(seg)
        self.segments.sort(key=lambda s: s.start)
        self._log(new_id, "add", f"Segment added at {fmt_time(start, True)}")
        self._mark_dirty()
        self.structureChanged.emit()
        self.set_selection([new_id])
        return new_id

    def apply_snr_threshold(self, threshold: float) -> int:
        # recompute the automatic clean/noisy label; hand-set values are kept
        self.snr_threshold = threshold
        changed = 0
        snap = self._snapshot()
        for s in self.segments:
            if not s.has_text or s.snr_mean is None:
                continue
            new_auto = "clean" if s.snr_mean >= threshold else "noisy"
            if s.quality == s.quality_auto:
                s.quality = new_auto
            if new_auto != s.quality_auto:
                changed += 1
            s.quality_auto = new_auto
        if changed:
            self._push_undo(snap)
            self._log(None, "snr", f"Clean/Noisy threshold set to {threshold:g} dB ({changed} segments updated)", kind="info")
            self._mark_dirty()
            self.structureChanged.emit()
        return changed

    def apply_reprocessed(self, sid: int, new: Segment) -> None:
        # called after the backend re-ran SNR and ASR on a copy of the segment
        s = self.seg(sid)
        if not s:
            return
        self._push_undo()
        for name in ("snr", "snr_energy", "snr_wada", "quality", "quality_auto", "asr_a", "asr_b",
                     "final_text", "status", "decided_by", "confidence"):
            setattr(s, name, getattr(new, name))
        self._log(sid, "reprocess", "SNR and ASR re-run for this segment")
        self._touch(sid)
