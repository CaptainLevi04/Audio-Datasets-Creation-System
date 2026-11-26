"""Headless checks of the annotation logic and the main window wiring."""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from sautlab import export
from sautlab.backend.mock import MockBackend
from sautlab.main_window import MainWindow
from sautlab.models import Segment
from sautlab.textutil import compare_transcripts, normalize_ar
from sautlab.theme import STYLESHEET


def pump(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def main(workspace: str):
    # text utilities
    check(normalize_ar("أَحْمَدُ، المَدْرَسَةُ ـــ") == "احمد المدرسه", "arabic normalization")
    r = compare_transcripts("هذا مثال على التعليم", "هذا مثال على التعليم")
    check(r["agree"] and r["status"] == "accepted", "identical transcripts accepted")
    r = compare_transcripts("هذا مثال على التعليم الحديث اليوم", "هذا مثال على التعلم الحديثة اليوم")
    check(r["status"] == "needs_review" and r["diff_words"] == 2, "small difference needs review")
    r = compare_transcripts("واحد اثنان ثلاثة اربعة خمسة ستة", "سبعة ثمانية تسعة عشرة حداشر")
    check(r["status"] == "difficult", "very different is difficult")

    app = QApplication([])
    app.setStyleSheet(STYLESHEET)
    win = MainWindow(MockBackend(workspace, 3.0))
    win.resize(1600, 1000)
    win.show()
    t0 = time.time()
    while win.audio.audio is None and time.time() - t0 < 90:
        pump(200)
    c = win.c
    check(win.audio.audio is not None and len(c.segments) > 5, f"video loaded with {len(c.segments)} segments")

    # selection and inspector
    first_conflict = c.conflict_ids()[0]
    c.set_selection([first_conflict])
    pump(100)
    check(c.current.id == first_conflict, "selection works")
    check(win.inspector.start_spin.value() == c.current.start, "inspector shows start")

    # label change + undo
    sid = c.selected[0]
    old = c.seg(sid).label
    c.set_label([sid], "music")
    check(c.seg(sid).label == "music" and c.seg(sid).status == "accepted", "label -> music accepts segment")
    c.undo()
    check(c.seg(sid).label == old, "undo restores label")
    c.redo()
    check(c.seg(sid).label == "music", "redo reapplies label")
    c.undo()

    # boundary drag keeps neighbours contiguous
    idx = c.index_of(sid)
    a, b = c.segments[idx], c.segments[idx + 1]
    seam = a.end
    c.begin_drag(a.id, "end")
    c.drag_boundary(a.id, "end", seam - 0.5)
    c.end_drag()
    check(abs(a.end - (seam - 0.5)) < 1e-6 and abs(b.start - a.end) < 1e-6, "dragging a border moves both neighbours")
    c.undo()
    a, b = c.seg(a.id), c.seg(b.id)
    check(abs(a.end - seam) < 1e-6 and abs(b.start - seam) < 1e-6, "undo of a drag restores both")

    # clamp: cannot cross neighbour
    c.set_times(a.id, end=b.end + 100)
    check(a.end < b.end and b.end - b.start >= 0.099, "border cannot cross the next segment")
    c.undo()
    a, b = c.seg(a.id), c.seg(b.id)

    # times via spin boxes
    old_start = a.start
    c.set_times(a.id, start=old_start + 0.3)
    check(abs(c.seg(a.id).start - (old_start + 0.3)) < 1e-6, "start edit applied")
    c.undo()
    a = c.seg(a.id)

    # split / merge / delete / add
    n0 = len(c.segments)
    mid = (a.start + a.end) / 2
    new_id = c.split(a.id, mid)
    check(len(c.segments) == n0 + 1 and c.seg(new_id).start == round(mid, 3), "split adds a segment at the cut")
    check(c.merge([a.id, new_id]) and len(c.segments) == n0, "merge joins them again")
    c.delete([c.segments[-1].id])
    check(len(c.segments) == n0 - 1, "delete removes a segment")
    c.undo()
    check(len(c.segments) == n0, "undo restores deleted segment")
    gap_start = c.segments[-1].end
    c.delete([c.segments[-1].id])
    nid = c.add(c.segments[-1].end + 0.0, label="music")
    check(nid is not None and c.seg(nid).label == "music", "add creates a segment in the gap")
    c.undo()
    c.undo()

    # transcripts
    texty = next(s for s in c.segments if s.has_text and s.status == "needs_review")
    c.set_selection([texty.id])
    c.choose_transcript(texty.id, "b")
    check(texty.final_text == texty.asr_b and texty.status == "accepted" and texty.decided_by == "b", "choose model B accepts")
    c.set_final_text(texty.id, "نص معدل يدويا")
    check(texty.final_text == "نص معدل يدويا" and texty.decided_by == "edited", "manual transcript edit")
    c.set_quality([texty.id], "noisy")
    check(texty.quality == "noisy", "override quality")
    c.set_quality([texty.id], None)
    check(texty.quality == texty.quality_auto, "quality back to automatic")

    # conflict filter
    win.table.status_combo.setCurrentIndex(win.table.status_combo.findData("conflicts"))
    pump(50)
    shown = win.table.proxy.rowCount()
    check(shown == len(c.conflict_ids()) and shown < len(c.segments), f"conflicts-only filter shows {shown} rows")
    win.table.status_combo.setCurrentIndex(0)
    win.table.search.setText("silence")
    pump(50)
    check(0 < win.table.proxy.rowCount() < len(c.segments), "search filters rows")
    win.table.search.setText("")

    # change log and persistence
    check(any(e["action"] == "transcript" for e in c.log), "changes are logged with user")
    check(all(e["user"] for e in c.log), "every log entry has a user")
    c.save()
    reloaded = MockBackend(workspace, 3.0).load_segments(c.video.id)
    check(len(reloaded) == len(c.segments), "saved work reloads from disk")

    # export
    tmp = tempfile.mkdtemp()
    export.write_csv(Path(tmp) / "a.csv", c.segments)
    text = (Path(tmp) / "a.csv").read_text(encoding="utf-8-sig")
    check(text.splitlines()[0].startswith("start,end"), "csv header written")
    n = export.export_segment_audio(win.backend.audio_path(c.video.id), tmp, c.video, c.segments)
    check(n > 0 and len(list(Path(tmp).glob("*.wav"))) == n == len(list(Path(tmp).glob("*.txt"))), f"{n} wav+txt pairs exported")
    shutil.rmtree(tmp)

    # player + zoom + keyboard-free actions
    win.player.play_range(c.segments[1].start, c.segments[1].end)
    pump(300)
    check(win.player.playing and win.player.pos > c.segments[1].start, "player advances")
    win.player.pause()
    win.audio.zoom(0.5)
    pump(200)
    x0, x1 = win.audio.view_range()
    check(x1 - x0 < win.audio.duration, "zoom in narrows the view")
    check(win.audio.img_detail.isVisible(), "detailed spectrogram appears when zoomed")

    # reprocess in a worker
    tid = texty.id
    win._reprocess(tid)
    pump(1200)
    check(c.seg(tid).asr_a != "", "re-run ASR returns")

    print(f"\nfailures: {check.failed}")
    return check.failed


if __name__ == "__main__":
    ws = tempfile.mkdtemp()
    code = main(ws)
    shutil.rmtree(ws, ignore_errors=True)
    sys.exit(1 if code else 0)
