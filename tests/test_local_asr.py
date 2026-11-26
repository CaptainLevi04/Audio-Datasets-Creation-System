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

                backend = CorpusBackend(data_dir, classifier="heuristic", vad="energy", run_asr=True)
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
                win.close()
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
