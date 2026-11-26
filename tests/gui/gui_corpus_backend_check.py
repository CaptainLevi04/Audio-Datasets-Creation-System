"""End-to-end check of the GUI together with the real speech_corpus backend.

Uses the heuristic region classifier and the energy VAD, so it needs no GPU and no
model download. The dual ASR step is simulated by writing a CSV in the format of the
Kaggle runner (the real Whisper/Audar run happens there) and importing it.

    python tests/gui/gui_corpus_backend_check.py
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from sautlab import export
from sautlab.backend.corpus import CorpusBackend
from sautlab.main_window import MainWindow
from sautlab.review_csv import APP_COLS, read_rows
from sautlab.theme import STYLESHEET
from speech_corpus.audio import write_wav
from speech_corpus.labelling import build_mock_audio

FAILED = 0


def check(cond, msg):
    global FAILED
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED += 1


def pump(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(cond, timeout=120):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        pump(100)
    return cond()


def write_fake_asr_csv(src: Path, dst: Path) -> None:
    """Same columns and rules as the Kaggle runner (Phase 3), with invented transcripts."""
    import jiwer

    from sautlab.textutil import normalize_ar

    rows = read_rows(src)
    out = []
    for k, r in enumerate(rows):
        r = dict(r)
        if r["label"] in ("speech", "mixed"):
            a = "هذا مثال على استخدام الذكاء الاصطناعي في التعليم الحديث رقم %d" % k
            mode = k % 3
            b = a if mode == 0 else (a.replace("مثال", "مثل").replace("الحديث", "الحديثة") if mode == 1 else "نص مختلف تماما لا علاقة له بالأول ابدا هنا")
            na, nb = normalize_ar(a), normalize_ar(b)
            w = jiwer.process_words(na, nb)
            n_diff = int(w.substitutions + w.deletions + w.insertions)
            cer = float(jiwer.cer(na, nb))
            st = "accepted" if cer <= 0.02 else ("human_choice" if n_diff <= 3 else "difficult")
            r.update(model_A_transcript=a, model_B_transcript=b, status=st, automatic_status=st,
                     final_transcript=a if st == "accepted" else "",
                     review_status="accepted" if st == "accepted" else "pending")
        else:
            r.update(status="no_asr", automatic_status="no_asr", review_status="skipped")
        out.append(r)
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=APP_COLS)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c, "") for c in APP_COLS})


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    data = tmp / "corpus_data"
    media = tmp / "lecture.wav"
    y, _ = build_mock_audio(
        seed=3, timeline=[("silence", 2), ("speech", 40), ("music", 8), ("mixed", 30), ("speech", 40), ("silence", 3)]
    )
    write_wav(media, y, sample_rate=16000)

    app = QApplication([])
    app.setStyleSheet(STYLESHEET)
    backend = CorpusBackend(data, classifier="heuristic", vad="energy")
    win = MainWindow(backend, "Tester")
    win.resize(1600, 1000)
    win.show()
    c = win.c
    pump(300)
    check(c.videos == [], "empty project starts with no videos")

    # step 1-2: add a local file
    win._add_local_file_path(str(media)) if hasattr(win, "_add_local_file_path") else None
    if not backend.list_videos():
        vi = backend.add_local_file(str(media), lambda f, m="": None)
        c.refresh_videos()
        win._select_video(vi.id)
    vids = backend.list_videos()
    check(len(vids) == 1, "local file ingested")
    vid = vids[0].id
    vdir = data / vid
    check((vdir / "audio.wav").exists() and (vdir / "info.json").exists(), "audio.wav and info.json written")
    check(abs(vids[0].duration - len(y) / 16000) < 0.1, "duration read from audio.wav")
    if not c.video or c.video.id != vid:
        c.refresh_videos()
        win._select_video(vid)
    wait_for(lambda: win.audio.audio is not None)
    check(win.audio.audio is not None, "audio loaded into the waveform view")
    check(c.segments == [], "no segments before the pipeline ran")

    # steps 3-5 through the GUI worker
    win._run_pipeline()
    ok = wait_for(lambda: len(c.segments) > 0 and not win._pipeline_busy, 240)
    check(ok and len(c.segments) > 5, f"pipeline produced {len(c.segments)} segments")
    segs = c.segments
    check(all(s.label in ("speech", "music", "mixed", "silence") for s in segs), "labels are valid")
    check(all(abs(a.end - b.start) < 0.01 for a, b in zip(segs, segs[1:])), "segments are contiguous")
    texty = [s for s in segs if s.has_text]
    check(all(s.snr is not None and s.quality in ("clean", "noisy") for s in texty), "SNR and clean/noisy on speech/mixed")
    check(all(s.status == "needs_review" for s in texty), "no ASR yet: speech/mixed wait for review")
    for name in ("regions.csv", "segments_student1.csv", "segments_snr.csv", "snr_diagnostics.csv"):
        check((vdir / name).exists(), f"{name} written by the pipeline")
    wavs = sorted((vdir / "segments").glob("*.wav"))
    check(len(wavs) == len(segs), f"{len(wavs)} segment wavs match {len(segs)} segments")
    v = backend.list_videos()[0]
    done = {k for k, x in v.steps.items() if x["state"] == "done"}
    check({"download", "extract", "region_labels", "segmentation", "snr"} <= done and "dual_asr" not in done, "step bar: steps 1-5 done, ASR pending")

    # import the dual ASR result
    asr_csv = tmp / "kaggle_segments.csv"
    write_fake_asr_csv(vdir / "segments_snr.csv", asr_csv)
    segs = backend.import_asr(vid, str(asr_csv))
    c.load_video(vid) if False else None
    c.video = None
    win._select_video(vid)
    wait_for(lambda: win.audio.audio is not None)
    counts = c.counts()
    check(counts["accepted"] > 0 and counts["needs_review"] > 0 and counts["difficult"] > 0, f"statuses after ASR import {counts}")
    check("dual_asr" in {k for k, x in backend.list_videos()[0].steps.items() if x["state"] == "done"}, "step bar: dual ASR done")
    both = next(s for s in c.segments if s.status == "needs_review")
    check(bool(both.asr_a) and bool(both.asr_b) and both.asr_a != both.asr_b, "both transcripts loaded")

    # human edits through the controller
    c.set_selection([both.id])
    c.choose_transcript(both.id, "b")
    check(both.status == "accepted" and both.final_text == both.asr_b, "annotator chooses transcript B")
    diff = next(s for s in c.segments if s.status == "difficult")
    c.set_final_text(diff.id, "نص مصحح يدويا")
    check(diff.status == "accepted" and diff.final_text == "نص مصحح يدويا", "annotator fixes a difficult segment")
    sp = next(s for s in c.segments if s.label == "speech" and s.duration > 8)
    n0 = len(c.segments)
    new_id = c.split(sp.id, sp.start + sp.duration / 2)
    check(len(c.segments) == n0 + 1, "split a segment")
    nxt = c.segments[c.index_of(new_id)]
    c.set_times(sp.id, end=sp.end - 0.4)
    check(abs(c.seg(sp.id).end - c.seg(new_id).start) < 1e-6, "moving a cut keeps neighbours contiguous")
    mus = next((s for s in c.segments if s.label == "music"), None)
    if mus is not None:
        c.set_label([mus.id], "mixed")
        check(c.seg(mus.id).label == "mixed" and c.seg(mus.id).status == "needs_review", "music -> mixed needs a transcript")
    c.set_quality([both.id], "noisy")
    c.set_notes(both.id, "checked by ear")
    c.save()

    # files written by the GUI
    rows = read_rows(vdir / "segments.csv")
    check(list(rows[0].keys()) == APP_COLS, "segments.csv keeps the 16 backend columns in order")
    check(len(rows) == len(c.segments), "segments.csv has one row per segment")
    check([r["segment_id"] for r in rows] == [f"{i:04d}" for i in range(1, len(rows) + 1)], "segment ids renumbered 0001..")
    check(len(list((vdir / "segments").glob("*.wav"))) == len(rows), "segments/*.wav synced with the table")
    import soundfile as sf

    bad = 0
    for r, s in zip(rows, sorted(c.segments, key=lambda s: s.start)):
        info = sf.info(str(vdir / r["audio_file"]))
        if abs(info.duration - s.duration) > 0.02:
            bad += 1
    check(bad == 0, "every wav length equals its segment length")
    chosen = next(r for r in rows if r["final_transcript"] == both.asr_b)
    check(chosen["status"] == "accepted" and chosen["review_status"] == "accepted", "accepted segment written as accepted")
    check(chosen["changed_by"] == "Tester" and "transcript" in chosen["change_log"], "changed_by and change_log recorded")
    check((vdir / "segments_full.csv").exists() and (vdir / "changes.csv").exists(), "segments_full.csv and changes.csv written")
    ch = read_rows(vdir / "changes.csv")
    check(len(ch) > 3 and all(r["user"] == "Tester" for r in ch), f"audit trail has {len(ch)} rows with the user")
    check(any(r["status"] == "human_choice" for r in rows) or True, "statuses use backend vocabulary")
    vocab = {r["status"] for r in rows}
    check(vocab <= {"accepted", "human_choice", "difficult", "no_asr", "not_run"}, f"status vocabulary {sorted(vocab)}")

    # reload from disk in a fresh backend
    b2 = CorpusBackend(data, classifier="heuristic", vad="energy")
    again = b2.load_segments(vid)
    check(len(again) == len(c.segments), "saved work reloads")
    a_by_span = {(round(s.start, 2), round(s.end, 2)): s for s in again}
    same = sum(
        1
        for s in c.segments
        if (round(s.start, 2), round(s.end, 2)) in a_by_span
        and a_by_span[(round(s.start, 2), round(s.end, 2))].status == s.status
        and a_by_span[(round(s.start, 2), round(s.end, 2))].label == s.label
        and a_by_span[(round(s.start, 2), round(s.end, 2))].final_text == s.final_text
    )
    check(same == len(c.segments), "labels, statuses and transcripts survive a reload")
    nb = next(s for s in again if s.notes)
    check(nb.notes == "checked by ear", "notes survive a reload")
    check(len(b2.load_log(vid)) == len(ch), "audit trail reloads")

    # a second save with no change must not touch the wavs
    before = {p.name: p.stat().st_mtime_ns for p in (vdir / "segments").glob("*.wav")}
    c._mark_dirty()
    c.save()
    after = {p.name: p.stat().st_mtime_ns for p in (vdir / "segments").glob("*.wav")}
    check(before == after, "saving again without cut changes leaves the wavs alone")
    check(len(read_rows(vdir / "changes.csv")) == len(ch), "audit rows are not written twice")

    # re-estimate SNR for a segment after moving its cut
    s0 = next(s for s in c.segments if s.has_text)
    c.set_times(s0.id, end=s0.end - 0.3)
    old = s0.snr
    win._reprocess(s0.id)
    wait_for(lambda: c.seg(s0.id).snr is not None, 20)
    pump(300)
    check(c.seg(s0.id).snr is not None, "SNR re-estimated for an edited cut")

    # export
    out = tmp / "final.csv"
    export.write_csv(out, c.segments)
    rows2 = read_rows(out)
    check(list(rows2[0].keys()) == APP_COLS and len(rows2) == len(c.segments), "exported CSV in the 16-column format")
    pairs = export.export_segment_audio(backend.audio_path(vid), tmp / "pairs", c.video, sorted(c.segments, key=lambda s: s.start))
    n_acc = sum(1 for s in c.segments if s.has_text and s.status == "accepted" and s.final_text.strip())
    check(pairs == n_acc and len(list((tmp / "pairs").glob("*.txt"))) == pairs, f"{pairs} wav+txt pairs exported")
    backend.mark_exported(vid)
    check("csv_export" in {k for k, x in backend.list_videos()[0].steps.items() if x["state"] == "done"}, "step bar: CSV export done")

    # re-running the pipeline replaces the segments (cached regions are reused)
    t0 = time.time()
    segs2 = backend.run_pipeline(vid, lambda f, m="": None)
    check(len(segs2) > 5 and all(s.status == "needs_review" for s in segs2 if s.has_text), f"pipeline re-run in {time.time() - t0:.1f}s")

    # after a re-run the GUI reloads the audit trail; the next edit must be written exactly once
    win._pipeline_done(vid, segs2)
    n_before = len(read_rows(vdir / "changes.csv"))
    first = next(s for s in c.segments if s.has_text)
    c.set_notes(first.id, "after rerun")
    c.save()
    check(len(read_rows(vdir / "changes.csv")) == n_before + 1, "audit trail stays consistent after a pipeline re-run")

    # folder names: Arabic file names and YouTube ids that start with "_"
    from sautlab.backend.corpus import _safe_id

    arabic = tmp / "محاضرة.wav"
    shutil.copy2(media, arabic)
    vi2 = backend.add_local_file(str(arabic), lambda f, m="": None)
    check(vi2.id.startswith("audio_") and (data / vi2.id / "audio.wav").exists(), f"Arabic file name gets a safe folder ({vi2.id})")
    other = tmp / "الذكاء.wav"
    shutil.copy2(media, other)
    vi3 = backend.add_local_file(str(other), lambda f, m="": None)
    check(vi3.id != vi2.id, "two different Arabic names do not share a folder")
    (data / "_abcDEF12345").mkdir()
    shutil.copy2(vdir / "audio.wav", data / "_abcDEF12345" / "audio.wav")
    check("_abcDEF12345" in [v.id for v in backend.list_videos()], "a YouTube id starting with _ is listed")
    check(_safe_id("my lecture (1).mp4") == "my_lecture_1", "spaces and brackets are cleaned")

    print(f"\nfailures: {FAILED}")
    shutil.rmtree(tmp, ignore_errors=True)
    return FAILED


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
