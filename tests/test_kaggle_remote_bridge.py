"""Tests for Kaggle <-> Local GUI compact bridge and remote GPU worker."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sautlab.backend.corpus import CorpusBackend
from sautlab.backend.remote_kaggle import RemoteKaggleBackend
from sautlab.kaggle_bridge import (
    SR,
    auto_import_latest_zip,
    build_corpus_archive,
    ensure_conditioned_flac,
    find_latest_kaggle_zip,
    hydrate_video_dir,
    should_auto_import,
    unpack_corpus_archive,
)
from sautlab.remote_server import (
    RemoteJobManager,
    create_remote_http_server,
    start_remote_server,
)
from speech_corpus.audio import read_wav, write_wav


def _make_tone(dur_s: float = 12.0, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0.0, dur_s, int(dur_s * SR), endpoint=False, dtype=np.float32)
    # Mix tones + modulation so FLAC compresses cleanly and realistically vs PCM WAV
    wave = 0.25 * np.sin(2.0 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2.0 * np.pi * 1.5 * t))
    return wave.astype(np.float32)


def _populate_sample_video(vdir: Path, vid: str = "vid_demo_001", dur_s: float = 12.0) -> None:
    vdir.mkdir(parents=True, exist_ok=True)
    audio = _make_tone(dur_s=dur_s)
    write_wav(vdir / "audio.wav", audio, sample_rate=SR)
    (vdir / "info.json").write_text(
        json.dumps(
            {
                "video_id": vid,
                "title": "Sample Arabic Lecture",
                "channel": "Demo",
                "duration_s": dur_s,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "license": "Creative Commons Attribution",
                "is_creative_commons": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"start": 0.0, "end": 6.0, "label": "speech"},
            {"start": 6.0, "end": 12.0, "label": "music"},
        ]
    ).to_csv(vdir / "regions.csv", index=False)

    seg_dir = vdir / "segments"
    seg_dir.mkdir(exist_ok=True)
    write_wav(seg_dir / "segment_0001.wav", audio[: 6 * SR], sample_rate=SR)
    write_wav(seg_dir / "segment_0002.wav", audio[6 * SR :], sample_rate=SR)

    rows = [
        {
            "start": 0.0,
            "end": 6.0,
            "label": "speech",
            "snr": 29.5,
            "quality": "clean",
            "model_A_transcript": "مرحبا بكم في المحاضرة",
            "model_B_transcript": "مرحبا بكم في المحاضره",
            "final_transcript": "مرحبا بكم في المحاضرة",
            "status": "accepted",
            "review_status": "accepted",
            "automatic_status": "accepted",
            "segment_id": "0001",
            "audio_file": "segments/segment_0001.wav",
            "changed_by": "",
            "changed_at": "",
            "change_log": "[]",
        },
        {
            "start": 6.0,
            "end": 12.0,
            "label": "music",
            "snr": 12.0,
            "quality": "noisy",
            "model_A_transcript": "",
            "model_B_transcript": "",
            "final_transcript": "",
            "status": "no_asr",
            "review_status": "skipped",
            "automatic_status": "no_asr",
            "segment_id": "0002",
            "audio_file": "segments/segment_0002.wav",
            "changed_by": "",
            "changed_at": "",
            "change_log": "[]",
        },
    ]
    pd.DataFrame(rows).to_csv(vdir / "segments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(vdir / "segments_full.csv", index=False, encoding="utf-8-sig")
    (vdir / "transcripts").mkdir(exist_ok=True)
    (vdir / "transcripts" / "segment_0001.txt").write_text("مرحبا بكم في المحاضرة", encoding="utf-8")


class KaggleBridgeAndRemoteTests(unittest.TestCase):
    def test_compact_archive_roundtrip_and_corpus_backend_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            kaggle_data = tmp_p / "kaggle_working" / "corpus_data"
            vid = "dQw4w9WgXcQ"
            vdir = kaggle_data / vid
            _populate_sample_video(vdir, vid=vid)
            # Ensure sensitive/unneeded files are excluded from archive
            (vdir / ".cookies.txt").write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            (vdir / "video.mp4").write_bytes(b"fake_mp4_bytes" * 200)

            compact_zip = tmp_p / "compact.zip"
            full_zip = tmp_p / "full.zip"
            meta_zip = tmp_p / "meta.zip"
            build_corpus_archive(kaggle_data, compact_zip, mode="compact")
            build_corpus_archive(kaggle_data, full_zip, mode="full")
            build_corpus_archive(kaggle_data, meta_zip, mode="metadata")

            with zipfile.ZipFile(compact_zip, "r") as zf:
                names = set(zf.namelist())
                self.assertIn(f"corpus_data/{vid}/audio_conditioned.flac", names)
                self.assertIn(f"corpus_data/{vid}/segments.csv", names)
                self.assertNotIn(f"corpus_data/{vid}/audio.wav", names)
                self.assertNotIn(f"corpus_data/{vid}/segments/segment_0001.wav", names)
                self.assertNotIn(f"corpus_data/{vid}/video.mp4", names)
                self.assertNotIn(f"corpus_data/{vid}/.cookies.txt", names)

            with zipfile.ZipFile(meta_zip, "r") as zf:
                mnames = set(zf.namelist())
                self.assertFalse(any(n.endswith(".flac") or n.endswith(".wav") for n in mnames))
                self.assertIn(f"corpus_data/{vid}/segments.csv", mnames)

            self.assertLess(compact_zip.stat().st_size, full_zip.stat().st_size)

            # Import via CorpusBackend on local PC & check auto-hydration + Step 8 data
            local_data = tmp_p / "local_pc" / "corpus_data"
            backend = CorpusBackend(local_data, classifier="heuristic", vad="energy")
            imported = backend.import_kaggle_zip(compact_zip)
            self.assertEqual(imported, [vid])

            local_vdir = local_data / vid
            self.assertTrue((local_vdir / "audio.wav").exists())
            self.assertTrue((local_vdir / "segments" / "segment_0001.wav").exists())
            self.assertTrue((local_vdir / "segments" / "segment_0002.wav").exists())

            reconstructed, sr = read_wav(local_vdir / "audio.wav", target_sr=SR)
            self.assertEqual(sr, SR)
            self.assertEqual(len(reconstructed), 12 * SR)

            vids = backend.list_videos()
            self.assertEqual([v.id for v in vids], [vid])
            self.assertTrue(all(vids[0].steps.values()))
            segs = backend.load_segments(vid)
            self.assertEqual(len(segs), 2)
            self.assertEqual(segs[0].asr_a, "مرحبا بكم في المحاضرة")
            self.assertEqual(segs[0].status, "accepted")

    def test_remote_kaggle_server_cookies_and_hybrid_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            kaggle_data = tmp_p / "kaggle_corpus"
            work_root = tmp_p / "kaggle_work"
            work_root.mkdir()
            # Simulate read-only cookies in /kaggle/input
            ro_cookies = tmp_p / "kaggle_input_cookies.txt"
            ro_cookies.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tSECRET_COOKIE\n", encoding="utf-8")

            seen_cookies: list[tuple[Path | None, str | None, bool]] = []

            def fake_download(url: str, data_dir: Path, cookie_path: Path | None, progress):
                progress(0.4, "Fake yt-dlp downloading...")
                vid = "abcDEF12345"
                vdir = data_dir / vid
                vdir.mkdir(parents=True, exist_ok=True)
                write_wav(vdir / "audio.wav", _make_tone(8.0), sample_rate=SR)
                info = {
                    "video_id": vid,
                    "title": "Remote Downloaded Lecture",
                    "channel": "Kaggle",
                    "duration_s": 8.0,
                    "url": url,
                    "license": "Creative Commons Attribution",
                }
                (vdir / "info.json").write_text(json.dumps(info), encoding="utf-8")
                txt = cookie_path.read_text(encoding="utf-8") if cookie_path else None
                # Must NOT point at the original read-only input path!
                is_writable_copy = cookie_path is not None and cookie_path.resolve() != ro_cookies.resolve()
                seen_cookies.append((cookie_path, txt, is_writable_copy))
                return vid, info

            def fake_pipeline(vdir: Path, params: dict, progress):
                progress(0.5, "Fake AST + Cut Engine + Dual ASR...")
                _populate_sample_video(vdir, vid=vdir.name, dur_s=8.0)
                return {"video_id": vdir.name, "segments": 2}

            mgr = RemoteJobManager(
                data_dir=kaggle_data,
                work_root=work_root,
                cookies_source=ro_cookies,
                download_fn=fake_download,
                pipeline_fn=fake_pipeline,
            )
            server, _ = create_remote_http_server(
                data_dir=kaggle_data,
                host="127.0.0.1",
                port=0,
                cookies_source=ro_cookies,
                work_root=work_root,
                manager=mgr,
            )
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()

            try:
                local_data = tmp_p / "pc_corpus"
                client = RemoteKaggleBackend(
                    data_dir=local_data,
                    remote_url=f"http://127.0.0.1:{port}",
                    poll_interval=0.05,
                    classifier="heuristic",
                    vad="energy",
                )

                health = client.check_health()
                self.assertEqual(health["status"], "ok")
                self.assertTrue(health["has_cookies"])

                # 1) Remote download using Kaggle cookies
                info = client.download("https://www.youtube.com/watch?v=abcDEF12345", lambda _f, _m="": None)
                self.assertEqual(info.id, "abcDEF12345")
                self.assertEqual(len(seen_cookies), 1)
                cpath, ctxt, is_copy = seen_cookies[0]
                self.assertTrue(is_copy)
                self.assertIn("SECRET_COOKIE", ctxt or "")
                self.assertFalse(cpath.exists(), "Temporary writable cookies file should be cleaned up")

                # 2) Extract audio fetches ONLY FLAC & decodes audio.wav locally
                wav_path = Path(client.extract_audio("abcDEF12345", 16000, 1, lambda _f, _m="": None))
                self.assertTrue(wav_path.exists())
                self.assertTrue((local_data / "abcDEF12345" / "audio_conditioned.flac").exists())

                # 3) Run pipeline executes remotely, transfers metadata.zip ONLY, slices WAVs locally
                segs = client.run_pipeline("abcDEF12345", lambda _f, _m="": None)
                self.assertEqual(len(segs), 2)
                self.assertTrue((local_data / "abcDEF12345" / "segments" / "segment_0001.wav").exists())
                self.assertTrue((local_data / "abcDEF12345" / "segments" / "segment_0002.wav").exists())

                # 4) Pull pre-computed Kaggle-only video
                kaggle_only = kaggle_data / "precomputed_vid"
                _populate_sample_video(kaggle_only, vid="precomputed_vid", dur_s=8.0)
                ensure_conditioned_flac(kaggle_only)
                pulled = client.pull_remote_videos(video_ids=["precomputed_vid"])
                self.assertEqual(pulled, ["precomputed_vid"])
                self.assertTrue((local_data / "precomputed_vid" / "audio.wav").exists())
                self.assertEqual(len(client.load_segments("precomputed_vid")), 2)

                # 5) Local file upload fallback to Kaggle GPU
                local_media = tmp_p / "local_clip.wav"
                write_wav(local_media, _make_tone(6.0), sample_rate=SR)
                loc_info = client.add_local_file(local_media, lambda _f, _m="": None)
                loc_segs = client.run_pipeline(loc_info.id, lambda _f, _m="": None)
                self.assertEqual(len(loc_segs), 2)
                self.assertTrue((kaggle_data / loc_info.id / "audio_conditioned.flac").exists())

                # 6) Path traversal guard returns HTTP 400/404
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/api/videos/..secret/flac")
                self.assertIn(ctx.exception.code, (400, 404))

                # Ensure local client cookies override works when passed explicitly
                local_cookie = tmp_p / "client_cookies.txt"
                local_cookie.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tCLIENT_OVERRIDE\n", encoding="utf-8")
                client.cookies_file = str(local_cookie)
                client.download("https://www.youtube.com/watch?v=abcDEF12345", lambda _f, _m="": None)
                self.assertIn("CLIENT_OVERRIDE", seen_cookies[-1][1] or "")
            finally:
                server.shutdown()
                server.server_close()
                mgr.shutdown()

    def test_auto_downloads_discovery_watcher_and_tunnel_fallback(self) -> None:
        import os
        import time
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            dl_dir = tmp_p / "Downloads"
            dl_dir.mkdir()
            kaggle_data = tmp_p / "kaggle_src"
            vid = "vid_auto_001"
            _populate_sample_video(kaggle_data / vid, vid=vid)

            # Unrelated or incomplete files must be ignored
            partial = dl_dir / "corpus_final_output.zip.crdownload"
            partial.write_bytes(b"partial")
            with zipfile.ZipFile(dl_dir / "corpus_unrelated.zip", "w") as zf:
                zf.writestr("readme.txt", "ignore me")
            self.assertIsNone(find_latest_kaggle_zip(search_dirs=[dl_dir]))

            z1 = dl_dir / "corpus_final_output.zip"
            build_corpus_archive(kaggle_data, z1, mode="compact")
            # Active Chrome .crdownload companion must defer pickup
            companion = dl_dir / "corpus_final_output.zip.crdownload"
            companion.write_bytes(b"still downloading")
            self.assertIsNone(find_latest_kaggle_zip(search_dirs=[dl_dir]))
            companion.unlink()

            local_data = tmp_p / "local_gui_data"
            picked, vids = auto_import_latest_zip(local_data, search_dirs=[dl_dir])
            self.assertEqual(picked, z1.resolve())
            self.assertEqual(vids, [vid])
            self.assertTrue((local_data / vid / "audio.wav").exists())
            self.assertFalse(should_auto_import(z1, local_data))

            # Second auto-import call skips already-imported archive
            picked2, vids2 = auto_import_latest_zip(local_data, search_dirs=[dl_dir])
            self.assertIsNone(picked2)
            self.assertEqual(vids2, [])

            # Newer archive `corpus_final_output (1).zip` should be picked up by MainWindow live watcher
            vid2 = "vid_auto_002"
            _populate_sample_video(kaggle_data / vid2, vid=vid2)
            time.sleep(0.05)
            z2 = dl_dir / "corpus_final_output (1).zip"
            build_corpus_archive(kaggle_data, z2, mode="compact", video_ids=[vid2])

            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            from PySide6.QtCore import QEventLoop, QTimer
            from PySide6.QtWidgets import QApplication
            from sautlab.main_window import MainWindow

            app = QApplication.instance() or QApplication([])
            backend = RemoteKaggleBackend(local_data, classifier="heuristic", vad="energy")
            win = MainWindow(backend, "Tester", watch_downloads=[dl_dir])
            win._poll_downloads()

            deadline = time.time() + 10.0
            while not (local_data / vid2 / "segments" / "segment_0001.wav").exists() and time.time() < deadline:
                loop = QEventLoop()
                QTimer.singleShot(50, loop.quit)
                loop.exec()
            win._stop_workers()
            win.close()

            self.assertTrue((local_data / vid2 / "segments" / "segment_0001.wav").exists())
            self.assertFalse(should_auto_import(z2, local_data))

        # Tunnel auto-fallback chain (localhost.run -> pinggy -> cloudflared) + beacon publish
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        with (
            patch("sautlab.remote_server.start_localhost_run_tunnel", side_effect=RuntimeError("lhr down")),
            patch("sautlab.remote_server.start_pinggy_tunnel", return_value=(fake_proc, "https://xyz.a.free.pinggy.link")) as png_mock,
            patch("sautlab.remote_server.publish_kaggle_endpoint", return_value="amrgamal007/sautlab-endpoint") as pub_mock,
        ):
            handle = start_remote_server(data_dir=Path(tempfile.mkdtemp()), port=0, tunnel="auto", block=False)
            try:
                self.assertEqual(handle["url"], "https://xyz.a.free.pinggy.link")
                png_mock.assert_called_once()
                pub_mock.assert_called_once()
            finally:
                handle["stop"]()

        # Auto-beacon lazy connection: GUI starts local, Kaggle comes online later, next action auto-binds
        with tempfile.TemporaryDirectory() as tmp2:
            tp = Path(tmp2)

            def _dl(url, ddir, ck, prog):
                vd = ddir / "autoVid1234"
                vd.mkdir(parents=True, exist_ok=True)
                write_wav(vd / "audio.wav", _make_tone(4.0), sample_rate=SR)
                info = {"video_id": "autoVid1234", "title": "Auto", "duration_s": 4.0}
                (vd / "info.json").write_text(json.dumps(info), encoding="utf-8")
                return "autoVid1234", info

            mgr = RemoteJobManager(data_dir=tp / "kdata", download_fn=_dl)
            srv, _ = create_remote_http_server(data_dir=tp / "kdata", host="127.0.0.1", port=0, manager=mgr)
            port = srv.server_address[1]
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                lazy_client = RemoteKaggleBackend(tp / "pcdata", remote_url=None, auto_beacon=True, poll_interval=0.05)
                self.assertFalse(lazy_client.is_remote)
                with patch("sautlab.kaggle_bridge.fetch_kaggle_endpoint", return_value={"url": f"http://127.0.0.1:{port}"}):
                    vi = lazy_client.download("https://www.youtube.com/watch?v=autoVid1234", lambda _f, _m="": None)
                self.assertTrue(lazy_client.is_remote)
                self.assertEqual(lazy_client.remote_url, f"http://127.0.0.1:{port}")
                self.assertEqual(vi.id, "autoVid1234")
            finally:
                srv.shutdown()
                srv.server_close()
                mgr.shutdown()


if __name__ == "__main__":
    unittest.main()
