"""Pure Kaggle CLI backend (`kaggle kernels push` / `datasets` / `kernels output`).

Runs heavy processing (YouTube download with Kaggle `/kaggle/input` cookies +
`deno`, audio extraction, AST, Cut Engine v2, SNR, Whisper large-v3, and
Audar-ASR-V1-Turbo) 100% on Kaggle GPU via the official Kaggle API without any
external tunnel (`localhost.run`, `cloudflared`, `pinggy`), automatically creates /
updates the user's named Kaggle dataset, downloads the compact archive once
ready, and hydrates WAVs & segments locally for the SautLab PySide6 GUI.
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ..kaggle_bridge import (
    _get_authenticated_kaggle_api,
    default_downloads_dirs,
    download_file_from_kaggle_dataset,
    ensure_conditioned_flac,
    record_auto_import,
    sanitize_dataset_slug,
    unpack_corpus_archive,
    upload_files_to_kaggle_dataset,
)
from ..models import Segment, VideoInfo
from .base import Progress
from .corpus import CorpusBackend

REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_error_from_kaggle_log(log_path: Path) -> str:
    if not log_path.is_file():
        return "Kaggle kernel failed (no log file returned)."
    try:
        entries = json.loads(log_path.read_text(encoding="utf-8", errors="replace"))
        lines = [str(e.get("data", "")) for e in entries if isinstance(e, dict)]
    except Exception:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    text = "".join(lines).strip()
    if not text:
        return "Kaggle kernel failed with an empty log."
    tail = "\n".join(text.splitlines()[-25:])
    return f"Kaggle kernel error:\n{tail}"


def _status_text(resp: Any) -> tuple[str, str | None]:
    st = getattr(resp, "status", None)
    if st is None and isinstance(resp, dict):
        st = resp.get("status")
    s_raw = str(st or resp or "").upper()
    for known in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "RUNNING", "QUEUED"):
        if known in s_raw:
            s_raw = known
            break
    fail_msg = getattr(resp, "failure_message", None) or getattr(resp, "failureMessage", None)
    if isinstance(resp, dict):
        fail_msg = fail_msg or resp.get("failureMessage")
    return s_raw, fail_msg


class KaggleCliBackend(CorpusBackend):
    """Automates Kaggle GPU runs cleanly using `kaggle` CLI / `KaggleApi`."""

    def __init__(
        self,
        data_dir: str | Path = "corpus_data",
        dataset_name: str = "sautlab-corpus",
        kernel_slug: str = "main-runner",
        queue_slug: str = "sautlab-queue",
        cookies_dataset: str = "yt-cookies",
        poll_interval: float = 5.0,
        notebook_template: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.dataset_name = sanitize_dataset_slug(dataset_name)
        self.kernel_slug = kernel_slug
        self.queue_slug = queue_slug
        self.cookies_dataset = cookies_dataset
        self.poll_interval = float(poll_interval)
        self.notebook_template = Path(notebook_template) if notebook_template else (REPO_ROOT / "main-runner.ipynb")
        self._active_kernel_ref: str | None = None
        self._persistent_worker: bool = False
        self._shutdown_sent: bool = False

    def set_dataset_name(self, name: str) -> str:
        self.dataset_name = sanitize_dataset_slug(name)
        return self.dataset_name

    def _auth(self):
        api, username = _get_authenticated_kaggle_api()
        if api is None or not username:
            raise RuntimeError(
                "Kaggle API credentials not found. Set ~/.kaggle/access_token or KAGGLE_API_TOKEN."
            )
        return api, username

    def _resolve_cookies_dataset(self, api, username: str) -> str | None:
        if "/" in self.cookies_dataset:
            return self.cookies_dataset
        cand = f"{username}/{self.cookies_dataset}"
        try:
            api.dataset_list_files(cand)
            return cand
        except Exception:
            try:
                matches = api.dataset_list(user=username, search="cookies")
                for m in matches:
                    ref = getattr(m, "ref", "")
                    if "cookie" in ref.lower():
                        return ref
            except Exception:
                pass
        return None

    def _prepare_kernel_bundle(
        self,
        api,
        username: str,
        job: dict[str, Any],
        bundle_dir: Path,
    ) -> str:
        kernel_ref = f"{username}/{self.kernel_slug}"
        meta: dict[str, Any] = {
            "id": kernel_ref,
            "title": "Main_Runner",
            "code_file": "main-runner.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        with tempfile.TemporaryDirectory(prefix="sautlab_kmeta_") as meta_tmp:
            try:
                api.kernels_pull(kernel_ref, path=meta_tmp, metadata=True, quiet=True)
                pulled_meta = Path(meta_tmp) / "kernel-metadata.json"
                if pulled_meta.is_file():
                    existing = json.loads(pulled_meta.read_text(encoding="utf-8"))
                    meta.update(existing)
                    meta["code_file"] = "main-runner.ipynb"
                    meta["enable_gpu"] = True
                    meta["enable_internet"] = True
            except Exception:
                pass

        ds_sources = list(meta.get("dataset_sources") or [])
        cookies_ref = self._resolve_cookies_dataset(api, username)
        if cookies_ref and cookies_ref not in ds_sources:
            ds_sources.append(cookies_ref)
        queue_ref = f"{username}/{self.queue_slug}"
        if queue_ref not in ds_sources:
            ds_sources.append(queue_ref)
        meta["dataset_sources"] = ds_sources

        nb = json.loads(self.notebook_template.read_text(encoding="utf-8"))
        # Configure notebook Cell 6 for pure Kaggle CLI worker mode with CURRENT_JOB embedded
        c6 = "".join(nb["cells"][6]["source"])
        c6 = c6.replace("START_REMOTE_SERVER = True", "START_REMOTE_SERVER = False")
        job_lit = json.dumps(job, ensure_ascii=False)
        cli_snippet = (
            f"\nKAGGLE_CLI_MODE = True\n"
            f"CURRENT_JOB = {job_lit}\n"
        )
        c6 = c6 + cli_snippet
        lines = c6.splitlines(keepends=True)
        if lines and lines[-1].endswith("\n"):
            lines[-1] = lines[-1][:-1]
        nb["cells"][6]["source"] = lines

        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "main-runner.ipynb").write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (bundle_dir / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return kernel_ref

    def _wait_dataset_ready(self, api, dataset_id: str, timeout_s: float = 30.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                st = str(api.dataset_status(dataset_id) or "").lower()
                if "ready" in st:
                    return
            except Exception:
                pass
            time.sleep(1.5)

    def _save_zip_to_downloads_and_unpack(self, zip_src: Path, ds_slug: str, vid: str) -> list[str]:
        dl_dirs = default_downloads_dirs()
        saved_zip = zip_src
        if dl_dirs:
            dst_zip = dl_dirs[0] / f"corpus_final_output_{ds_slug}_{vid}.zip"
            try:
                shutil.copy2(zip_src, dst_zip)
                saved_zip = dst_zip
            except Exception:
                saved_zip = zip_src

        vids = unpack_corpus_archive(saved_zip, self.root, hydrate=True)
        record_auto_import(saved_zip, self.root, vids)
        self._flags_cache.clear()
        return vids

    def _run_kaggle_job(
        self,
        job: dict[str, Any],
        extra_queue_files: dict[str, Path] | None,
        progress: Progress,
    ) -> str:
        api, username = self._auth()
        ds_slug = sanitize_dataset_slug(job.get("dataset_slug") or self.dataset_name)
        job["dataset_slug"] = ds_slug
        job["queue_slug"] = self.queue_slug
        dataset_id = f"{username}/{ds_slug}"
        queue_id = f"{username}/{self.queue_slug}"
        kernel_ref = f"{username}/{self.kernel_slug}"
        self._active_kernel_ref = kernel_ref

        progress(0.03, f"Sending job to Kaggle queue ({queue_id})...")
        q_files: dict[str, Path | str | bytes] = {"job.json": json.dumps(job, indent=2)}
        if extra_queue_files:
            q_files.update(extra_queue_files)
        upload_files_to_kaggle_dataset(
            self.queue_slug,
            q_files,
            title="SautLab Queue",
            version_notes=f"Job {job['job_id']}",
            public=False,
        )
        self._wait_dataset_ready(api, queue_id, timeout_s=25.0)

        need_push = True
        if self._persistent_worker:
            try:
                st, _ = _status_text(api.kernels_status(kernel_ref))
                if st == "RUNNING":
                    need_push = False
                    progress(0.12, f"Warm Kaggle session active (restarting fresh worker without pip)...")
            except Exception:
                need_push = True

        if need_push:
            self._persistent_worker = False
            progress(0.08, f"Starting Kaggle GPU session ({kernel_ref})...")
            with tempfile.TemporaryDirectory(prefix="sautlab_kpush_") as push_dir:
                self._prepare_kernel_bundle(api, username, job, Path(push_dir))
                api.kernels_push(push_dir)

        t0 = time.time()
        target_jid = job["job_id"]
        with tempfile.TemporaryDirectory(prefix="sautlab_kout_") as out_tmp:
            out_dir = Path(out_tmp)
            while True:
                elapsed = int(time.time() - t0)
                # 1) Check if persistent worker already uploaded job_status.json to named dataset
                ds_status_file = download_file_from_kaggle_dataset(dataset_id, "job_status.json", out_dir / "ds")
                if ds_status_file is not None and ds_status_file.is_file():
                    try:
                        st_data = json.loads(ds_status_file.read_text(encoding="utf-8"))
                    except Exception:
                        st_data = {}
                    if st_data.get("job_id") == target_jid:
                        if st_data.get("status") == "failed":
                            raise RuntimeError(st_data.get("error") or "Kaggle GPU worker job failed.")
                        if st_data.get("status") == "done":
                            self._persistent_worker = st_data.get("worker_mode") == "persistent"
                            progress(0.86, f"Downloading dataset {dataset_id} automatically...")
                            zip_dl = download_file_from_kaggle_dataset(
                                dataset_id, "corpus_final_output.zip", out_dir / "ds_zip"
                            )
                            if zip_dl is None:
                                raise RuntimeError(f"Dataset {dataset_id} marked done, but corpus_final_output.zip missing.")
                            vid = str(st_data.get("video_id") or job.get("video_id") or "")
                            progress(0.95, "Extracting & hydrating local audio + segments...")
                            vids = self._save_zip_to_downloads_and_unpack(zip_dl, ds_slug, vid)
                            if not vid and vids:
                                vid = vids[0]
                            progress(1.0, f"Ready in GUI (Kaggle dataset: {dataset_id})")
                            return vid

                # 2) Poll Kaggle kernel worker status
                k_status, k_fail = _status_text(api.kernels_status(kernel_ref))
                if k_status == "QUEUED":
                    progress(0.15, f"Waiting in Kaggle GPU queue... ({elapsed}s)")
                elif k_status == "RUNNING":
                    progress(0.45, f"Running Steps 1-6 on Kaggle GPU (dataset: {ds_slug})... ({elapsed}s)")
                elif k_status == "COMPLETE":
                    progress(0.80, "Kaggle run complete — downloading output archive...")
                    k_out = out_dir / "kernel"
                    k_out.mkdir(parents=True, exist_ok=True)
                    api.kernels_output(kernel_ref, path=str(k_out), force=True, quiet=True)
                    st_file = k_out / "job_status.json"
                    if st_file.is_file():
                        st_data = json.loads(st_file.read_text(encoding="utf-8"))
                        if st_data.get("status") == "failed":
                            raise RuntimeError(st_data.get("error") or "Kaggle job failed.")
                        vid = str(st_data.get("video_id") or job.get("video_id") or "")
                    else:
                        st_data = {}
                        vid = str(job.get("video_id") or "")
                    zip_out = k_out / "corpus_final_output.zip"
                    if not zip_out.is_file():
                        raise RuntimeError(_extract_error_from_kaggle_log(k_out / f"{self.kernel_slug}.log"))

                    progress(0.88, f"Publishing Kaggle dataset '{dataset_id}'...")
                    st_payload = st_data or {"job_id": target_jid, "video_id": vid, "status": "done", "worker_mode": "oneshot"}
                    st_file.write_text(json.dumps(st_payload, indent=2), encoding="utf-8")
                    upload_files_to_kaggle_dataset(
                        ds_slug,
                        {"corpus_final_output.zip": zip_out, "job_status.json": st_file},
                        title=ds_slug.replace("-", " ").title(),
                        version_notes=f"Auto-published for {vid}",
                        public=False,
                    )
                    self._persistent_worker = False
                    progress(0.95, "Extracting & hydrating local audio + segments...")
                    vids = self._save_zip_to_downloads_and_unpack(zip_out, ds_slug, vid)
                    if not vid and vids:
                        vid = vids[0]
                    progress(1.0, f"Ready in GUI (Kaggle dataset: {dataset_id})")
                    return vid
                elif k_status in {"ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}:
                    k_out = out_dir / "err"
                    k_out.mkdir(parents=True, exist_ok=True)
                    try:
                        api.kernels_output(kernel_ref, path=str(k_out), force=True, quiet=True)
                    except Exception:
                        pass
                    detail = _extract_error_from_kaggle_log(k_out / f"{self.kernel_slug}.log")
                    if k_fail:
                        detail = f"{k_fail}\n{detail}"
                    raise RuntimeError(detail)

                time.sleep(self.poll_interval)

    def download(self, url: str, progress: Progress) -> VideoInfo:
        from speech_corpus.downloader import extract_video_id

        vid_guess = extract_video_id(url)
        job = {
            "job_id": uuid.uuid4().hex[:12],
            "action": "process",
            "url": url.strip(),
            "video_id": vid_guess,
            "dataset_slug": self.dataset_name,
            "snr_threshold": self.snr_threshold,
            "snr_combine": self.snr_combine,
            "word_verification": self.word_verification,
            "zip_mode": "compact",
            "created_at": time.time(),
        }
        vid = self._run_kaggle_job(job, extra_queue_files=None, progress=progress)
        return self._refresh(self._dir(vid))

    def extract_audio(self, video_id: str, sample_rate: int, channels: int, progress: Progress) -> str:
        vdir = self._dir(video_id)
        wav = vdir / "audio.wav"
        if wav.exists():
            progress(1.0, "16 kHz audio ready locally")
            return str(wav)
        if (vdir / "audio_conditioned.flac").exists():
            from ..kaggle_bridge import hydrate_video_dir

            hydrate_video_dir(vdir, slice_segments=False)
            self._refresh(vdir)
            progress(1.0, "16 kHz audio hydrated")
            return str(wav)
        return super().extract_audio(video_id, sample_rate, channels, progress)

    def run_pipeline(self, video_id: str, progress: Progress) -> list[Segment]:
        vdir = self._dir(video_id)
        info_path = vdir / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        url = info.get("url") if info.get("source") == "youtube" else None

        extra_files = None
        if not url:
            flac = ensure_conditioned_flac(vdir)
            extra_files = {"audio_conditioned.flac": flac}

        job = {
            "job_id": uuid.uuid4().hex[:12],
            "action": "process",
            "url": url or "",
            "video_id": video_id,
            "dataset_slug": self.dataset_name,
            "snr_threshold": self.snr_threshold,
            "snr_combine": self.snr_combine,
            "word_verification": self.word_verification,
            "zip_mode": "compact",
            "created_at": time.time(),
        }
        self._run_kaggle_job(job, extra_queue_files=extra_files, progress=progress)
        self._flags_cache.clear()
        (vdir / "gui_state.json").unlink(missing_ok=True)
        segs = self.load_segments(video_id)
        self._update_state(vdir, wav_spans=[[s.start, s.end] for s in segs])
        self._refresh(vdir)
        return segs

    def shutdown_session(self) -> None:
        """Signal any persistent Kaggle worker session to exit cleanly when GUI closes."""
        if self._shutdown_sent or not self._active_kernel_ref:
            return
        self._shutdown_sent = True
        try:
            upload_files_to_kaggle_dataset(
                self.queue_slug,
                {
                    "job.json": json.dumps(
                        {
                            "job_id": f"stop_{uuid.uuid4().hex[:8]}",
                            "action": "shutdown",
                            "created_at": time.time(),
                        }
                    )
                },
                title="SautLab Queue",
                version_notes="Shutdown signal from closed GUI",
                public=False,
            )
        except Exception:
            pass
