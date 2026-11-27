"""Tests for local Dual ASR (NVIDIA Conformer + Quantized Audar GGUF), CorpusBackend, and GUI."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sautlab.backend.corpus import CorpusBackend
from sautlab.local_asr import (
    _wait_llama_server_ready,
    clean_audar_text,
    free_cuda_memory,
    run_local_dual_asr,
    score_transcript_pair,
    transcribe_audar_quantized,
    transcribe_conformer,
)
from speech_corpus import ast_model
from speech_corpus.audio import DEFAULT_SR, write_wav
from speech_corpus.labelling import build_mock_audio


class _FakeLlamaHandler(BaseHTTPRequestHandler):
    requests_log: list[str] = []

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        self.requests_log.append(self.path)
        idx = len(self.requests_log)
        if idx == 1:
            txt = "language Arabic<asr_text>مرحبا بكم في اختبار النظام الصوتي"
        elif idx == 2:
            txt = "مرحبا بكم في اختبار النظام الصوت"
        else:
            txt = "language None<asr_text>"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"text": txt}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass


class LocalDualAsrTests(unittest.TestCase):
    def test_clean_audar_text_and_scoring(self) -> None:
        self.assertEqual(clean_audar_text("language None<asr_text>"), "")
        self.assertEqual(clean_audar_text("language Arabic<asr_text>السلام عليكم"), "السلام عليكم")
        self.assertEqual(clean_audar_text("نص عادي"), "نص عادي")

        acc = score_transcript_pair("أهلاً بكم", "اهلا بكم")
        self.assertEqual(acc["csv_status"], "accepted")
        hc = score_transcript_pair("أهلاً بكم في الدرس", "اهلا بكم في الكتاب")
        self.assertEqual(hc["csv_status"], "human_choice")
        diff = score_transcript_pair("نص", "")
        self.assertEqual(diff["csv_status"], "difficult")

    def test_free_cuda_memory_clears_ast_cache(self) -> None:
        ast_model._AST_CACHE["engine"] = object()
        free_cuda_memory()
        self.assertEqual(ast_model._AST_CACHE, {})

    def test_conformer_and_audar_server_and_corpus_pipeline_integration(self) -> None:
        _FakeLlamaHandler.requests_log = []
        srv = HTTPServer(("127.0.0.1", 0), _FakeLlamaHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tp = Path(tmp)
                data_dir = tp / "corpus_data"

                # Test `transcribe_audar_quantized` against the local HTTP server
                w1 = tp / "a.wav"
                w2 = tp / "b.wav"
                w3 = tp / "c.wav"
                tone = np.full(16000, 0.05, dtype=np.float32)
                for p in (w1, w2, w3):
                    write_wav(p, tone, sample_rate=DEFAULT_SR)

                audar_out = transcribe_audar_quantized([w1, w2, w3], server_url=url)
                self.assertEqual(audar_out[0], "مرحبا بكم في اختبار النظام الصوتي")
                self.assertEqual(audar_out[1], "مرحبا بكم في اختبار النظام الصوت")
                self.assertEqual(audar_out[2], "")

                # Test `transcribe_conformer` with mocked NeMo model & verify GPU cache cleared
                fake_hyp = [MagicMock(text="مرحبا بكم في اختبار النظام الصوتي"), MagicMock(text="تجربة ثانية")]
                fake_model = MagicMock()
                fake_model.transcribe.return_value = fake_hyp
                fake_nemo_asr = MagicMock()
                fake_nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained.return_value = fake_model
                fake_nemo_collections = MagicMock()
                fake_nemo_collections.asr = fake_nemo_asr
                fake_nemo_pkg = MagicMock()
                fake_nemo_pkg.collections = fake_nemo_collections

                with (
                    patch.dict(
                        sys.modules,
                        {
                            "nemo": fake_nemo_pkg,
                            "nemo.collections": fake_nemo_collections,
                            "nemo.collections.asr": fake_nemo_asr,
                        },
                    ),
                    patch("sautlab.local_asr.free_cuda_memory") as free_mock,
                ):
                    conf_out = transcribe_conformer([w1, w2], rms_values=[0.1, 0.0001], device="cpu")
                self.assertEqual(conf_out, ["مرحبا بكم في اختبار النظام الصوتي", ""])
                fake_model.change_decoding_strategy.assert_called_with(decoder_type="ctc")
                self.assertGreaterEqual(free_mock.call_count, 2)

                # Full CorpusBackend local pipeline + MainWindow integration (with mocked ASR backends)
                media = tp / "sample_ar.wav"
                y, _ = build_mock_audio(
                    seed=7, timeline=[("silence", 2.0), ("speech", 18.0), ("music", 6.0), ("mixed", 16.0)]
                )
                write_wav(media, y, sample_rate=DEFAULT_SR)

                backend = CorpusBackend(
                    data_dir, classifier="heuristic", vad="energy", word_verification=False, run_asr=True
                )
                self.assertEqual(backend.asr_names, ("Conformer", "Audar"))
                vi = backend.add_local_file(str(media), lambda _f, _m="": None)
                vdir = data_dir / vi.id

                def _fake_conf(wavs, **_kw):
                    return [
                        "مرحبا بكم في اختبار النظام" if i == 0 else "كلام مختلف تماما"
                        for i in range(len(wavs))
                    ]

                def _fake_aud(wavs, **_kw):
                    return [
                        "مرحبا بكم في اختبار النظام" if i == 0 else "نص اخر لا يطابق الاول ابدا هنا الان"
                        for i in range(len(wavs))
                    ]

                with (
                    patch("sautlab.local_asr.transcribe_conformer", side_effect=_fake_conf),
                    patch("sautlab.local_asr.transcribe_audar_quantized", side_effect=_fake_aud),
                ):
                    segs = backend.run_pipeline(vi.id, lambda _f, _m="": None)

                text_segs = [s for s in segs if s.has_text]
                non_text = [s for s in segs if not s.has_text]
                self.assertTrue(len(text_segs) >= 2)
                self.assertTrue(any(s.status == "accepted" for s in text_segs))
                self.assertTrue(any(s.status == "difficult" for s in text_segs))
                self.assertTrue(all(not s.asr_a and not s.asr_b for s in non_text))
                self.assertTrue((vdir / "segments.csv").is_file())
                self.assertTrue((vdir / "segments_full.csv").is_file())
                self.assertTrue((vdir / "asr_conformer_ckpt.csv").is_file())
                self.assertTrue((vdir / "asr_audar_ckpt.csv").is_file())

                v_after = backend.list_videos()[0]
                self.assertEqual(v_after.steps.get("dual_asr", {}).get("state"), "done")

                # Re-process a single segment updates both ASR transcripts too
                with (
                    patch("sautlab.local_asr.transcribe_conformer", return_value=["نص محدث"]),
                    patch("sautlab.local_asr.transcribe_audar_quantized", return_value=["نص محدث"]),
                ):
                    re_seg = backend.reprocess_segment(vi.id, text_segs[0])
                self.assertEqual(re_seg.asr_a, "نص محدث")
                self.assertEqual(re_seg.asr_b, "نص محدث")
                self.assertEqual(re_seg.status, "accepted")

                # Restored MainWindow opens cleanly with ("Conformer", "Audar")
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
                from PySide6.QtWidgets import QApplication
                from sautlab.main_window import MainWindow

                app = QApplication.instance() or QApplication([])
                win = MainWindow(backend, "Tester")
                app.processEvents()
                self.assertEqual(win.c.asr_names, ("Conformer", "Audar"))
                self.assertFalse(hasattr(win.source, "dataset"))
                from PySide6.QtGui import QAction

                settings_actions = [a.text() for a in win.findChildren(QAction)]
                self.assertIn("Clear processing for current video...", settings_actions)
                win.close()

                # Clearing processing keeps downloaded audio/info.json and resets Steps 3-6 to pending
                cleared = backend.clear_processing(vi.id)
                self.assertTrue((vdir / "audio.wav").is_file())
                self.assertTrue((vdir / "info.json").is_file())
                self.assertFalse((vdir / "regions.csv").exists())
                self.assertFalse((vdir / "segments.csv").exists())
                self.assertFalse((vdir / "segments").exists())
                self.assertFalse((vdir / "transcripts").exists())
                self.assertEqual(cleared.steps["download"]["state"], "done")
                self.assertEqual(cleared.steps["extract"]["state"], "done")
                for st in ("region_labels", "segmentation", "snr", "dual_asr"):
                    self.assertEqual(cleared.steps[st]["state"], "pending")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_audar_subprocess_log_file_prevents_stderr_pipe_deadlock(self) -> None:
        import subprocess

        # 1. Early exit error tail is captured from the temp log file on Windows.
        with tempfile.NamedTemporaryFile(mode="w+b", prefix="sautlab_test_", suffix=".log", delete=False) as log_fh:
            log_path = Path(log_fh.name)
            proc = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stderr.write('simulated crash reason'); sys.exit(2)"],
                stdout=subprocess.DEVNULL,
                stderr=log_fh,
            )
            try:
                proc.wait(timeout=10)
                with self.assertRaises(RuntimeError) as ctx:
                    _wait_llama_server_ready(proc, "http://127.0.0.1:1", timeout_s=5.0, log_path=log_path)
                self.assertIn("simulated crash reason", str(ctx.exception))
            finally:
                log_fh.close()
                log_path.unlink(missing_ok=True)

        # 2. >150 KB of stderr across 85 requests must never deadlock and leaves no leftover temp logs.
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            fake_exe = tp / "llama-server.exe"
            fake_exe.write_bytes(b"")
            fake_model = tp / "m.gguf"
            fake_mmproj = tp / "mm.gguf"
            wav = tp / "clip.wav"
            write_wav(wav, np.full(8000, 0.05, dtype=np.float32), sample_rate=DEFAULT_SR)

            child_script = (
                "import sys, json\n"
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "args = sys.argv[1:]\n"
                "port = int(args[args.index('--port') + 1])\n"
                "assert args[args.index('-np') + 1] == '1'\n"
                "class H(BaseHTTPRequestHandler):\n"
                "  def do_GET(self):\n"
                "    self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
                "  def do_POST(self):\n"
                "    n = int(self.headers.get('Content-Length', '0')); self.rfile.read(n)\n"
                "    sys.stderr.write('LOG ' + ('x' * 2048) + '\\n'); sys.stderr.flush()\n"
                "    self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()\n"
                "    self.wfile.write(json.dumps({'text': 'مرحبا'}).encode('utf-8'))\n"
                "  def log_message(self, *a): pass\n"
                "HTTPServer(('127.0.0.1', port), H).serve_forever()\n"
            )

            real_popen = subprocess.Popen
            captured_cmds: list[list[str]] = []

            def _spawn_py_server(cmd, **kw):
                captured_cmds.append(list(cmd))
                self.assertNotEqual(kw.get("stderr"), subprocess.PIPE)
                return real_popen([sys.executable, "-c", child_script, *cmd[1:]], **kw)

            before_logs = set(Path(tempfile.gettempdir()).glob("sautlab_llama_*.log"))
            with (
                patch("sautlab.local_asr.ensure_llama_cpp_server", return_value=fake_exe),
                patch("sautlab.local_asr.ensure_audar_gguf", return_value=(fake_model, fake_mmproj)),
                patch("sautlab.local_asr.subprocess.Popen", side_effect=_spawn_py_server),
            ):
                out = transcribe_audar_quantized([wav] * 85)
            after_logs = set(Path(tempfile.gettempdir()).glob("sautlab_llama_*.log"))

            self.assertEqual(len(out), 85)
            self.assertTrue(all(x == "مرحبا" for x in out))
            self.assertIn("-np", captured_cmds[0])
            self.assertEqual(before_logs, after_logs)

    def test_two_detector_energy_silero_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            data_dir = tp / "corpus_data"
            default_backend = CorpusBackend(data_dir, run_asr=False)
            self.assertFalse(default_backend.word_verification)

            media = tp / "speech_ar.wav"
            y, _ = build_mock_audio(seed=11, timeline=[("silence", 1.5), ("speech", 18.0), ("silence", 1.5)])
            write_wav(media, y, sample_rate=DEFAULT_SR)

            backend = CorpusBackend(data_dir, classifier="heuristic", vad="energy", run_asr=False)
            vi = backend.add_local_file(str(media), lambda _f, _m="": None)
            vdir = data_dir / vi.id
            # Even if a stale word_timestamps.json sits on disk, 2-detector default ignores Whisper completely.
            (vdir / "word_timestamps.json").write_text('[{"start": 2.0, "end": 4.0, "word": "تجربة"}]', encoding="utf-8")

            with patch("speech_corpus.cut_engine_v2._load_word_model") as load_mock:
                segs = backend.run_pipeline(vi.id, lambda _f, _m="": None)
            self.assertTrue(len(segs) >= 1)
            load_mock.assert_not_called()

            grid_df = pd.read_csv(vdir / "voting_grid.csv")
            self.assertTrue((grid_df["num_detectors"] == 2).all())
            self.assertTrue((grid_df["whisper_status"] == "UNAVAILABLE").all())
            self.assertEqual(int(grid_df["whisper_vote"].sum()), 0)

    def test_ast_recitation_mapping_and_silence_guard(self) -> None:
        self.assertIn("chant", ast_model.SPEECH_NAMES)
        self.assertIn("mantra", ast_model.SPEECH_NAMES)

        # Unaccompanied melodic recitation at -19 dB must NOT be misclassified as silence
        # even when a high continuous noise floor produces sil_thr around -18.3 dB.
        d_recite = ast_model.decide_window(p_sp=0.61, p_mu=0.14, p_si=0.0, rms_db=-18.95, floor_db=-22.27)
        self.assertEqual(d_recite["pred_content"], "speech")

        # True quiet or digital silence still maps cleanly to silence.
        d_quiet = ast_model.decide_window(p_sp=0.05, p_mu=0.03, p_si=0.40, rms_db=-35.0, floor_db=-37.0)
        self.assertEqual(d_quiet["pred_content"], "silence")
        d_hard = ast_model.decide_window(p_sp=0.10, p_mu=0.05, p_si=0.80, rms_db=-55.0, floor_db=-60.0)
        self.assertEqual(d_hard["pred_content"], "silence")

        # Vocal+music outro at -33 dB with strong music/speech probs must not become silence.
        d_outro = ast_model.decide_window(p_sp=0.58, p_mu=0.61, p_si=0.018, rms_db=-33.2, floor_db=-32.9)
        self.assertEqual(d_outro["pred_content"], "mixed")

    def test_dp_preserves_real_pauses_with_rank4_valley_fallback_and_word_guard(self) -> None:
        from speech_corpus.cut_engine_v2 import PauseCandidate, plan_unified_dp_cuts, verify_candidates_with_words

        c_strong1 = PauseCandidate(
            t=8.0, p_start=7.8, p_end=8.2, duration=0.4,
            vote_class="STRONG", vote_count=2.0, energy_vote=1, silero_vote=1, whisper_vote=0,
            energy_margin_db=8.0, silero_margin=0.4, is_accepted=True, needs_review=0, review_reason="none",
        )
        c_strong2 = PauseCandidate(
            t=17.5, p_start=17.3, p_end=17.7, duration=0.4,
            vote_class="STRONG", vote_count=2.0, energy_vote=1, silero_vote=1, whisper_vote=0,
            energy_margin_db=8.0, silero_margin=0.4, is_accepted=True, needs_review=0, review_reason="none",
        )
        # 21s gap after 17.5 bridged by one FALLBACK_SPLIT candidate at 28.0 -> total 38.5s.
        c_fb = PauseCandidate(
            t=28.0, p_start=27.9, p_end=28.1, duration=0.2,
            vote_class="FALLBACK_SPLIT", vote_count=0.0, energy_vote=0, silero_vote=0, whisper_vote=0,
            energy_margin_db=-1.0, silero_margin=-0.2, is_accepted=False, needs_review=1,
            review_reason="fallback_split", weak=True,
        )
        segs, tag = plan_unified_dp_cuts(38.5, [c_strong1, c_strong2, c_fb], transitions=[])
        self.assertNotEqual(tag, "blind_cut_emergency")
        self.assertEqual([s["end"] for s in segs], [8.0, 17.5, 28.0, 38.5])

        # Stretched Whisper word timestamp must not mark a genuine STRONG acoustic pause as word_boundary_risk.
        c_weak = PauseCandidate(
            t=3.0, p_start=2.95, p_end=3.05, duration=0.1,
            vote_class="WEAK", vote_count=1.0, energy_vote=1, silero_vote=0, whisper_vote=0,
            energy_margin_db=1.0, silero_margin=0.0, is_accepted=False, needs_review=1,
            review_reason="weak_pause", weak=True,
        )
        words = [
            {"start": 2.8, "end": 3.25, "word": "كلمة"},       # short word <0.60s overlapping c_weak
            {"start": 7.0, "end": 8.6, "word": "ممتدة"},        # stretched >=0.60s across STRONG pause at 8.0s
        ]
        out_cands, risks, _ = verify_candidates_with_words([c_weak, c_strong1], words)
        self.assertEqual(risks, 1)
        self.assertTrue(out_cands[0].word_risk)
        self.assertFalse(out_cands[1].word_risk)

    def test_ensure_llama_cpp_server_handles_cpu_and_missing_cudart(self) -> None:
        from sautlab.local_asr import (
            LLAMA_CPP_CPU_URL,
            LLAMA_CPP_CUDART_URL,
            LLAMA_CPP_RELEASE_URL,
            ensure_llama_cpp_server,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            urls: list[str] = []

            def _fake_dl(url: str, root: Path, zip_name: str) -> None:
                urls.append(url)
                root.mkdir(parents=True, exist_ok=True)
                if url in (LLAMA_CPP_RELEASE_URL, LLAMA_CPP_CPU_URL):
                    (root / "llama-server.exe").write_bytes(b"")
                elif url == LLAMA_CPP_CUDART_URL:
                    (root / "cudart64_12.dll").write_bytes(b"")
                    (root / "cublas64_12.dll").write_bytes(b"")

            with (
                patch("sautlab.local_asr.shutil.which", return_value=None),
                patch("sautlab.local_asr._has_cuda", return_value=False),
                patch("sautlab.local_asr._download_and_extract_zip", side_effect=_fake_dl),
            ):
                exe_cpu = ensure_llama_cpp_server(cache_dir=tp)
            self.assertEqual(urls, [LLAMA_CPP_CPU_URL])
            self.assertEqual(exe_cpu.name, "llama-server.exe")

            urls.clear()
            with (
                patch("sautlab.local_asr.shutil.which", return_value=None),
                patch("sautlab.local_asr._has_cuda", return_value=True),
                patch("sautlab.local_asr._has_cuda_12_runtime", return_value=False),
                patch("sautlab.local_asr._download_and_extract_zip", side_effect=_fake_dl),
            ):
                exe_cuda = ensure_llama_cpp_server(cache_dir=tp)
            self.assertEqual(urls, [LLAMA_CPP_RELEASE_URL, LLAMA_CPP_CUDART_URL])
            self.assertEqual(exe_cuda.name, "llama-server.exe")


if __name__ == "__main__":
    unittest.main()
