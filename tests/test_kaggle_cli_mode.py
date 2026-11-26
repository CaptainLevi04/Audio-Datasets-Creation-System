"""Tests for pure Kaggle CLI backend, persistent worker loop, dataset naming, and GUI integration."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sautlab.backend.kaggle_cli import KaggleCliBackend
from sautlab.kaggle_bridge import SR, sanitize_dataset_slug
from sautlab.kaggle_cli_worker import ensure_kaggle_deps, run_worker_session
from speech_corpus.audio import write_wav


def _write_dummy_corpus_video(vdir: Path, vid: str) -> None:
    vdir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 4.0, 4 * SR, endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    write_wav(vdir / "audio.wav", audio, sample_rate=SR)
    (vdir / "info.json").write_text(
        json.dumps({"video_id": vid, "title": f"Video {vid}", "channel": "Demo", "duration_s": 4.0}),
        encoding="utf-8",
    )
    pd.DataFrame([{"start": 0.0, "end": 4.0, "label": "speech"}]).to_csv(vdir / "regions.csv", index=False)
    seg_dir = vdir / "segments"
    seg_dir.mkdir(exist_ok=True)
    write_wav(seg_dir / "segment_0001.wav", audio, sample_rate=SR)
    rows = [
        {
            "start": 0.0,
            "end": 4.0,
            "label": "speech",
            "snr": 28.0,
            "quality": "clean",
            "model_A_transcript": f"نص تجريبي {vid}",
            "model_B_transcript": f"نص تجريبي {vid}",
            "final_transcript": f"نص تجريبي {vid}",
            "status": "accepted",
            "review_status": "accepted",
            "automatic_status": "accepted",
            "segment_id": "0001",
            "audio_file": "segments/segment_0001.wav",
            "changed_by": "",
            "changed_at": "",
            "change_log": "[]",
        }
    ]
    pd.DataFrame(rows).to_csv(vdir / "segments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(vdir / "segments_full.csv", index=False, encoding="utf-8-sig")


class _FakeKaggleStore:
    """In-memory fake Kaggle API backing datasets and kernel push/output."""

    def __init__(self, username: str = "amrgamal007"):
        self.username = username
        self.datasets: dict[str, dict[str, bytes]] = {}
        self.kernel_pushes: list[tuple[str, dict]] = []
        self.kernel_outputs: dict[str, Path] = {}
        self.kernel_statuses: list[str] = []

    def upload(self, slug: str, files: dict[str, Path | str | bytes], **_kw) -> str:
        clean = sanitize_dataset_slug(slug)
        ref = f"{self.username}/{clean}"
        bucket = self.datasets.setdefault(ref, {})
        for name, val in files.items():
            if isinstance(val, bytes):
                bucket[name] = val
            elif isinstance(val, str) and not Path(val).exists():
                bucket[name] = val.encode("utf-8")
            else:
                bucket[name] = Path(val).read_bytes()
        return ref

    def download(self, dataset_id: str, file_name: str, dest_dir: str | Path) -> Path | None:
        ref = dataset_id if "/" in dataset_id else f"{self.username}/{sanitize_dataset_slug(dataset_id)}"
        raw = self.datasets.get(ref, {}).get(file_name)
        if raw is None:
            return None
        dst = Path(dest_dir)
        dst.mkdir(parents=True, exist_ok=True)
        out = dst / file_name
        out.write_bytes(raw)
        return out


class KaggleCliModeTests(unittest.TestCase):
    def test_sanitize_dataset_slug(self) -> None:
        self.assertEqual(sanitize_dataset_slug("My Arabic Dataset 1"), "my-arabic-dataset-1")
        self.assertEqual(sanitize_dataset_slug("abc"), "sautlab-abc")
        self.assertEqual(sanitize_dataset_slug("amrgamal007/News Corpus!"), "news-corpus")
        self.assertEqual(sanitize_dataset_slug("   "), "sautlab-corpus")

    def test_ensure_deps_skips_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("shutil.which", return_value="/usr/bin/tool"),
                patch("subprocess.run") as sp_mock,
            ):
                installed = ensure_kaggle_deps(tmp)
            self.assertFalse(installed)
            sp_mock.assert_not_called()
            self.assertTrue((Path(tmp) / ".deps_ready").exists())

    def test_persistent_worker_reuses_session_and_skips_deps_on_second_link(self) -> None:
        store = _FakeKaggleStore("amrgamal007")
        seen_jobs: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            ddir = tp / "kaggle_data"
            wroot = tp / "kaggle_work"

            def fake_dl(url, data_dir, ck, prog):
                vid = url.split("v=")[-1]
                _write_dummy_corpus_video(data_dir / vid, vid)
                return vid, {"video_id": vid, "title": f"Vid {vid}"}

            def fake_steps(vdir, params, prog):
                seen_jobs.append(params.get("job_id"))
                return {"video_id": vdir.name, "segments": 1}

            job1 = {"job_id": "j001", "action": "process", "url": "https://www.youtube.com/watch?v=vidFirst0001", "dataset_slug": "my-custom-corpus"}
            # Enqueue job2 + shutdown after job1 finishes
            poll_seq = iter([
                None,
                None,
            ])

            def fake_download(ds_id, fname, dst):
                if len(seen_jobs) == 1:
                    store.upload(
                        "sautlab-queue",
                        {"job.json": json.dumps({"job_id": "j002", "action": "process", "url": "https://www.youtube.com/watch?v=vidSecond02", "dataset_slug": "my-custom-corpus"})},
                    )
                elif len(seen_jobs) >= 2:
                    store.upload(
                        "sautlab-queue",
                        {"job.json": json.dumps({"job_id": "stop99", "action": "shutdown"})},
                    )
                return store.download(ds_id, fname, dst)

            with (
                patch("sautlab.kaggle_cli_worker.ensure_kaggle_deps", return_value=False) as deps_mock,
                patch("sautlab.kaggle_cli_worker._get_authenticated_kaggle_api", return_value=(MagicMock(), "amrgamal007")),
                patch("sautlab.kaggle_cli_worker._default_download_video", side_effect=fake_dl),
                patch("sautlab.kaggle_cli_worker._default_run_steps3_to_6", side_effect=fake_steps),
                patch("sautlab.kaggle_cli_worker.upload_files_to_kaggle_dataset", side_effect=store.upload),
                patch("sautlab.kaggle_cli_worker.download_file_from_kaggle_dataset", side_effect=fake_download),
            ):
                run_worker_session(
                    initial_job=job1,
                    data_dir=ddir,
                    work_root=wroot,
                    poll_interval_s=0.01,
                    use_subprocess=False,
                )

            self.assertEqual(seen_jobs, ["j001", "j002"])
            deps_mock.assert_called_once()
            self.assertIn("corpus_final_output.zip", store.datasets["amrgamal007/my-custom-corpus"])
            st = json.loads(store.datasets["amrgamal007/my-custom-corpus"]["job_status.json"].decode("utf-8"))
            self.assertEqual(st["worker_mode"], "persistent")
            self.assertEqual(st["video_id"], "vidSecond02")

    def test_backend_pushes_runs_auto_downloads_and_reuses_persistent_session(self) -> None:
        store = _FakeKaggleStore("amrgamal007")

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            pc_data = tp / "pc_corpus"
            dl_dir = tp / "Downloads"
            dl_dir.mkdir()
            kaggle_out_dir = tp / "fake_kout"
            kaggle_src = tp / "kaggle_src"

            api_mock = MagicMock()
            api_mock.dataset_status.return_value = "ready"
            api_mock.dataset_list_files.return_value = []
            api_mock.kernels_pull.side_effect = Exception("use defaults")

            push_count = {"n": 0}
            active_vid = {"vid": ""}

            def fake_push(folder):
                push_count["n"] += 1
                job_in = json.loads((Path(folder) / "main-runner.ipynb").read_text(encoding="utf-8"))
                store.kernel_pushes.append((folder, job_in))

            def fake_status(kernel_ref):
                # Trigger worker output publication when backend starts polling
                qraw = json.loads(store.datasets["amrgamal007/sautlab-queue"]["job.json"].decode("utf-8"))
                jid = qraw["job_id"]
                vid = qraw["video_id"]
                active_vid["vid"] = vid
                ds_slug = qraw["dataset_slug"]
                if ds_slug not in store.datasets or json.loads(store.datasets[ds_slug].get("job_status.json", b"{}").decode() or "{}").get("job_id") != jid:
                    _write_dummy_corpus_video(kaggle_src / vid, vid)
                    from sautlab.kaggle_bridge import build_corpus_archive

                    zpath = tp / f"{vid}.zip"
                    build_corpus_archive(kaggle_src, zpath, mode="compact", video_ids=[vid])
                    store.upload(
                        ds_slug,
                        {
                            "corpus_final_output.zip": zpath,
                            "job_status.json": json.dumps({"job_id": jid, "video_id": vid, "status": "done", "worker_mode": "persistent"}),
                        },
                    )
                obj = MagicMock()
                obj.status = "RUNNING"
                obj.failure_message = None
                return obj

            api_mock.kernels_push.side_effect = fake_push
            api_mock.kernels_status.side_effect = fake_status

            backend = KaggleCliBackend(
                data_dir=pc_data,
                dataset_name="My Test Corpus",
                poll_interval=0.01,
                classifier="heuristic",
                vad="energy",
            )
            self.assertEqual(backend.dataset_name, "my-test-corpus")

            with (
                patch("sautlab.backend.kaggle_cli._get_authenticated_kaggle_api", return_value=(api_mock, "amrgamal007")),
                patch("sautlab.backend.kaggle_cli.upload_files_to_kaggle_dataset", side_effect=store.upload),
                patch("sautlab.backend.kaggle_cli.download_file_from_kaggle_dataset", side_effect=store.download),
                patch("sautlab.backend.kaggle_cli.default_downloads_dirs", return_value=[dl_dir]),
            ):
                vi1 = backend.download("https://www.youtube.com/watch?v=Xr2iW-qZ_II", lambda _f, _m="": None)
                self.assertEqual(vi1.id, "Xr2iW-qZ_II")
                self.assertEqual(push_count["n"], 1)
                self.assertTrue((pc_data / "Xr2iW-qZ_II" / "audio.wav").exists())
                self.assertTrue((pc_data / "Xr2iW-qZ_II" / "segments" / "segment_0001.wav").exists())
                self.assertTrue((dl_dir / "corpus_final_output_my-test-corpus_Xr2iW-qZ_II.zip").exists())

                segs1 = backend.load_segments("Xr2iW-qZ_II")
                self.assertEqual(len(segs1), 1)
                self.assertIn("Xr2iW-qZ_II", segs1[0].asr_a)

                # Second URL reuses warm persistent Kaggle session without a second kernel push!
                backend.set_dataset_name("custom_ds_2")
                vi2 = backend.download("https://www.youtube.com/watch?v=vidSecond02", lambda _f, _m="": None)
                self.assertEqual(vi2.id, "vidSecond02")
                self.assertEqual(push_count["n"], 1)
                self.assertIn("amrgamal007/custom-ds-2", store.datasets)

                # Closing GUI sends shutdown to queue dataset
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
                from PySide6.QtWidgets import QApplication
                from sautlab.main_window import MainWindow

                app = QApplication.instance() or QApplication([])
                win = MainWindow(backend, "Tester")
                app.processEvents()
                self.assertFalse(win.source.dataset.isHidden())
                self.assertEqual(win.source.dataset.text(), "custom-ds-2")
                win._stop_workers()
                win.close()

                last_q = json.loads(store.datasets["amrgamal007/sautlab-queue"]["job.json"].decode("utf-8"))
                self.assertEqual(last_q["action"], "shutdown")

            # One-shot COMPLETE fallback: Kaggle lacks KAGGLE_API_TOKEN inside container,
            # so PC downloads via `kernels_output` and mirrors into the user's named dataset!
            with tempfile.TemporaryDirectory() as tmp2:
                tp2 = Path(tmp2)
                os_store = _FakeKaggleStore("amrgamal007")
                os_api = MagicMock()
                os_api.dataset_status.return_value = "ready"
                os_api.dataset_list_files.return_value = []
                os_api.kernels_pull.side_effect = Exception("use defaults")

                def oneshot_status(kernel_ref):
                    obj = MagicMock()
                    obj.status = "COMPLETE"
                    obj.failure_message = None
                    return obj

                def oneshot_output(kernel_ref, path, force=False, quiet=True):
                    qraw = json.loads(os_store.datasets["amrgamal007/sautlab-queue"]["job.json"].decode("utf-8"))
                    vid = qraw["video_id"]
                    jid = qraw["job_id"]
                    src_dir = tp2 / "k_src"
                    _write_dummy_corpus_video(src_dir / vid, vid)
                    from sautlab.kaggle_bridge import build_corpus_archive

                    out_p = Path(path)
                    out_p.mkdir(parents=True, exist_ok=True)
                    build_corpus_archive(src_dir, out_p / "corpus_final_output.zip", mode="compact", video_ids=[vid])
                    (out_p / "job_status.json").write_text(
                        json.dumps({"job_id": jid, "video_id": vid, "status": "done", "worker_mode": "oneshot"}),
                        encoding="utf-8",
                    )
                    return [], None

                os_api.kernels_status.side_effect = oneshot_status
                os_api.kernels_output.side_effect = oneshot_output

                os_backend = KaggleCliBackend(
                    data_dir=tp2 / "pc_out",
                    dataset_name="oneshot_named_ds",
                    poll_interval=0.01,
                    classifier="heuristic",
                    vad="energy",
                )
                with (
                    patch("sautlab.backend.kaggle_cli._get_authenticated_kaggle_api", return_value=(os_api, "amrgamal007")),
                    patch("sautlab.backend.kaggle_cli.upload_files_to_kaggle_dataset", side_effect=os_store.upload),
                    patch("sautlab.backend.kaggle_cli.download_file_from_kaggle_dataset", side_effect=os_store.download),
                    patch("sautlab.backend.kaggle_cli.default_downloads_dirs", return_value=[tp2 / "dl"]),
                ):
                    (tp2 / "dl").mkdir()
                    vi_os = os_backend.download("https://www.youtube.com/watch?v=oneShotVid1", lambda _f, _m="": None)
                self.assertEqual(vi_os.id, "oneShotVid1")
                self.assertIn("amrgamal007/oneshot-named-ds", os_store.datasets)
                self.assertTrue((tp2 / "pc_out" / "oneShotVid1" / "segments" / "segment_0001.wav").exists())


if __name__ == "__main__":
    unittest.main()
