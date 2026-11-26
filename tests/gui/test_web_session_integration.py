"""Headless 3-phase integration test: Student 1 -> Student 2 -> Student 3 (CorpusBackend + AnnotationSession).

Runs without PySide6, Gradio, or GPU models by exercising the exact modules
used by `runner.ipynb` and `sautlab.web_app`.
"""
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sautlab import export
from sautlab.backend.corpus import CorpusBackend
from sautlab.review_csv import APP_COLS, read_rows
from sautlab.session import AnnotationSession
from sautlab.textutil import compare_transcripts, highlight_html, normalize_ar
from speech_corpus.audio import DEFAULT_SR, write_wav
from speech_corpus.cut_engine_v2 import cut_speech_and_mixed_segments
from speech_corpus.labelling import build_mock_audio
from speech_corpus.quality_control import split_corpus, validate_segments_integrity
from speech_corpus.snr import compute_snr_for_segments, estimate_segment_snr


class TestThreePhaseIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="three_phase_test_"))
        self.data_root = self.tmp / "corpus_data"
        self.data_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_phase1_to_3_for_video(self, vid: str, seed: int):
        vdir = self.data_root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        timeline = [
            ("silence", 6.0),
            ("speech", 16.0),
            ("music", 10.0),
            ("mixed", 16.0),
            ("speech", 14.0),
        ]
        y, gt_df = build_mock_audio(seed=seed, timeline=timeline, sr=DEFAULT_SR)
        write_wav(vdir / "audio.wav", y, sample_rate=DEFAULT_SR)
        info = {
            "video_id": vid,
            "title": f"Test {vid}",
            "channel": "UnitTest",
            "duration_s": round(len(y) / DEFAULT_SR, 2),
            "license": "CC-BY",
            "url": f"https://example.com/{vid}",
        }
        (vdir / "info.json").write_text(json.dumps(info), encoding="utf-8")

        regions_df = gt_df.copy()
        regions_df["region_id"] = [f"R{i+1:04d}" for i in range(len(regions_df))]
        regions_df["confidence"] = 0.95
        regions_df.to_csv(vdir / "regions.csv", index=False)

        # Phase 1: Student 1 segmentation + QC
        seg_df = cut_speech_and_mixed_segments(
            audio=y, regions_df=regions_df, out_dir=vdir, sr=DEFAULT_SR,
            use_word_verification=False, run_ablation=False,
        )
        qc = validate_segments_integrity(seg_df, video_dir=vdir, total_dur=len(y) / DEFAULT_SR, sr=DEFAULT_SR)
        self.assertTrue(qc["passed"], f"QC failed: {qc['errors']}")
        seg_df.to_csv(vdir / "segments_student1.csv", index=False)

        # Phase 2: Student 2 Step 5 SNR on ALL segments
        s1_df = pd.read_csv(vdir / "segments_student1.csv", dtype={"segment_id": str})
        snr_df, _ = compute_snr_for_segments(s1_df, base_dir=vdir, threshold_db=25.0, save_csv=True, verbose=False)
        snr_df.to_csv(vdir / "segments_snr.csv", index=False, encoding="utf-8-sig")

        diag_rows = []
        for _, row in s1_df.iterrows():
            clip, sr_w = sf.read(str(vdir / row["wav_file"]), dtype="float32")
            est = estimate_segment_snr(clip, sr=sr_w, threshold_db=25.0)
            diag_rows.append({
                "segment_id": row["segment_id"],
                "label": row["label"],
                "snr_energy_db": est["snr_energy_db"],
                "snr_wada_db": est["snr_wada_db"],
                "snr": est["snr"],
                "quality": est["quality"],
                "snr_disagree": est["snr_disagree"],
            })
        pd.DataFrame(diag_rows).to_csv(vdir / "snr_diagnostics.csv", index=False, encoding="utf-8-sig")

        # Phase 3: Student 2 Step 6 Dual ASR on speech & mixed ONLY
        snr_df["_row_idx"] = np.arange(len(snr_df))
        mask = snr_df["label"].astype(str).str.lower().isin({"speech", "mixed"})
        df_asr = snr_df[mask].copy().reset_index(drop=True)
        df_non = snr_df[~mask].copy().reset_index(drop=True)

        self.assertGreater(len(df_asr), 0)
        self.assertGreater(len(df_non), 0)

        statuses, a_txts, b_txts, finals = [], [], [], []
        for k in range(len(df_asr)):
            a = "السلام عليكم ورحمة الله وبركاته هذا اختبار"
            if k == 0:
                b = a
                st = "accepted"
                fin = a
            elif k == 1:
                b = "السلام عليكم ورحمه الله وبركاته هذا تجربة"
                st = "human_choice"
                fin = ""
            else:
                b = "نص مختلف تماما عن الجملة الأولى في كل الكلمات تقريبا"
                st = "difficult"
                fin = ""
            a_txts.append(a)
            b_txts.append(b)
            statuses.append(st)
            finals.append(fin)

        df_asr["model_A_transcript"] = a_txts
        df_asr["model_B_transcript"] = b_txts
        df_asr["final_transcript"] = finals
        df_asr["status"] = statuses
        df_asr["automatic_status"] = statuses
        df_asr["review_status"] = ["accepted" if s == "accepted" else "pending" for s in statuses]

        df_non["model_A_transcript"] = ""
        df_non["model_B_transcript"] = ""
        df_non["final_transcript"] = ""
        df_non["status"] = "no_asr"
        df_non["automatic_status"] = "no_asr"
        df_non["review_status"] = "skipped"

        merged = pd.concat([df_asr, df_non], ignore_index=True).sort_values("_row_idx").reset_index(drop=True)
        app_df = merged[APP_COLS].copy()
        app_df.to_csv(vdir / "segments.csv", index=False, encoding="utf-8-sig")
        app_df.to_csv(vdir / "segments_asr.csv", index=False, encoding="utf-8-sig")

        full_df = merged.drop(columns=["_row_idx"], errors="ignore").copy()
        for col in [c for c in s1_df.columns if c not in full_df.columns]:
            full_df[col] = s1_df[col].values
        full_df.to_csv(vdir / "segments_full.csv", index=False, encoding="utf-8-sig")
        return vdir

    def test_end_to_end_3_phases_with_gui_overrides_and_packaging(self):
        vdirs = [
            self._run_phase1_to_3_for_video("vid_a", seed=11),
            self._run_phase1_to_3_for_video("vid_b", seed=22),
            self._run_phase1_to_3_for_video("vid_c", seed=33),
        ]

        # Phase 4: Student 3 CorpusBackend + AnnotationSession (same engine as Kaggle Web GUI)
        backend = CorpusBackend(self.data_root, classifier="heuristic", vad="energy", snr_threshold=25.0)
        session = AnnotationSession(backend, user="Amr")
        self.assertEqual([v.id for v in session.videos], ["vid_a", "vid_b", "vid_c"])

        session.load_video("vid_a", auto_save=False)
        self.assertGreater(len(session.segments), 4)
        self.assertIn("dual_asr", {k for k, v in session.video.steps.items() if v["state"] == "done"})

        # Check diff highlighting + audio slicing + spectrogram plot rendering
        conflict = next(s for s in session.segments if s.status == "needs_review")
        cmp = compare_transcripts(conflict.asr_a, conflict.asr_b)
        self.assertEqual(cmp["status"], "needs_review")
        html_a = highlight_html(conflict.asr_a, (cmp["tokens_a"], cmp["diff_a"]), "#fbbf24")
        self.assertIn("<span", html_a)

        clip, sr = session.slice_audio(conflict.start - 1.5, conflict.start + 1.5)
        self.assertEqual(sr, 16000)
        self.assertGreater(len(clip), 16000)

        plot_file = session.render_segment_plot(conflict.id, self.tmp / "seg_plot.png")
        self.assertTrue(Path(plot_file).exists())

        # Human overrides: choose B, edit difficult, override quality, move cut + recompute SNR, split, undo
        session.choose_transcript(conflict.id, "b")
        self.assertEqual(conflict.status, "accepted")
        self.assertEqual(conflict.final_text, conflict.asr_b)

        diff_seg = next(s for s in session.segments if s.status == "difficult")
        session.set_final_text(diff_seg.id, "تصحيح يدوي للنص الصعب")
        self.assertEqual(diff_seg.status, "accepted")

        session.set_quality(conflict.id, "noisy")
        self.assertEqual(conflict.quality, "noisy")

        old_end = conflict.end
        session.set_times(conflict.id, conflict.start, old_end - 0.5, sync_neighbors=True)
        self.assertAlmostEqual(conflict.end, old_end - 0.5, places=3)
        session.reprocess_segment(conflict.id)
        self.assertIsNotNone(conflict.snr)

        n0 = len(session.segments)
        sp_seg = next(s for s in session.segments if s.duration > 8.0)
        new_id = session.split(sp_seg.id, (sp_seg.start + sp_seg.end) / 2.0)
        self.assertIsNotNone(new_id)
        self.assertEqual(len(session.segments), n0 + 1)
        self.assertTrue(session.undo())
        self.assertEqual(len(session.segments), n0)

        # Save back to disk and verify persistence + audit trail + .wav/.txt export
        session.save()
        vdir_a = vdirs[0]
        self.assertTrue((vdir_a / "changes.csv").exists())
        self.assertTrue((vdir_a / "segments_full.csv").exists())
        self.assertTrue((vdir_a / "segments_asr.csv").exists())
        rows = read_rows(vdir_a / "segments.csv")
        self.assertEqual(list(rows[0].keys()), APP_COLS)

        # Verify CorpusBackend reload sees saved changes and human_review step
        reloaded = CorpusBackend(self.data_root).load_segments("vid_a")
        self.assertEqual(len(reloaded), len(session.segments))
        self.assertTrue(any(r.final_text == "تصحيح يدوي للنص الصعب" for r in reloaded))

        n_exp = export.export_segment_audio(backend.audio_path("vid_a"), self.tmp / "exp_a", session.video, session.segments)
        self.assertGreater(n_exp, 0)

        # Phase 5: Global corpus manifest + zero-leakage split + zip including audio.wav
        full_parts = []
        for vd in vdirs:
            df_f = pd.read_csv(vd / "segments_full.csv", dtype={"segment_id": str})
            full_parts.append(df_f)
        manifest_full = pd.concat(full_parts, ignore_index=True)
        splits = split_corpus(manifest_full, out_dir=self.data_root / "splits")
        self.assertEqual(splits["status"], "SUCCESS")

        zip_path = self.tmp / "corpus_final_output.zip"
        skip_names = {"audio_conditioned.flac", "video.mp4"}
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(self.data_root.rglob("*")):
                if f.is_file() and f.name not in skip_names and not f.name.endswith(".zip"):
                    zf.write(f, arcname=f.relative_to(self.tmp))

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            self.assertIn("corpus_data/vid_a/audio.wav", names)
            self.assertIn("corpus_data/vid_a/segments.csv", names)
            self.assertIn("corpus_data/vid_a/changes.csv", names)


if __name__ == "__main__":
    unittest.main()
