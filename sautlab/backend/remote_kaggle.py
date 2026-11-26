"""Remote Kaggle GPU adapter extending `CorpusBackend`.

Runs heavy GPU steps (download, 16 kHz extraction, AST region labelling,
Cut Engine v2, SNR, Whisper large-v3, and Audar-ASR-V1-Turbo) on a Kaggle
Remote Server while keeping the PySide6 `SautLab` GUI and all Step 8 human
review / override / export operations 100% local with minimal data transfer
(lossless FLAC + metadata zips only).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..kaggle_bridge import (
    ensure_conditioned_flac,
    hydrate_video_dir,
    unpack_corpus_archive,
)
from ..models import Segment, VideoInfo
from .base import Progress
from .corpus import CorpusBackend, _write_json


class RemoteKaggleBackend(CorpusBackend):
    """Hybrid backend: Kaggle executes Steps 1-6; local GUI handles Step 8."""

    def __init__(
        self,
        data_dir: str | Path = "corpus_data",
        remote_url: str | None = None,
        poll_interval: float = 1.5,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.remote_url = (remote_url or "").strip().rstrip("/") or None
        self.poll_interval = float(poll_interval)

    @property
    def is_remote(self) -> bool:
        return bool(self.remote_url)

    def _url(self, route: str) -> str:
        if not self.remote_url:
            raise RuntimeError("Kaggle Remote Server URL is not configured.")
        return f"{self.remote_url}{route}"

    def _api_json(self, method: str, route: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(self._url(route), data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except Exception:
                detail = body
            raise RuntimeError(f"Kaggle server error ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Kaggle server at {self.remote_url}: {e.reason}") from e

    def _upload_file(self, route: str, file_path: Path, timeout: float = 300.0) -> dict:
        raw = file_path.read_bytes()
        req = urllib.request.Request(
            self._url(route),
            data=raw,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Upload failed ({e.code}): {body}") from e

    def _download_file(self, route: str, dst: Path, progress: Progress, label: str, timeout: float = 300.0) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        req = urllib.request.Request(self._url(route), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if total > 0:
                            progress(min(0.95, got / total), f"{label}: {got / 1e6:.1f} / {total / 1e6:.1f} MB")
                        else:
                            progress(0.5, f"{label}: {got / 1e6:.1f} MB")
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Download failed ({e.code}): {body}") from e
        tmp.replace(dst)
        return dst

    def _poll_job(self, job_id: str, progress: Progress, scale: tuple[float, float] = (0.0, 1.0)) -> dict:
        lo, hi = scale
        while True:
            st = self._api_json("GET", f"/api/jobs/{job_id}")
            frac = lo + (hi - lo) * float(st.get("progress", 0.0))
            msg = st.get("message", "")
            progress(min(max(frac, 0.0), 1.0), msg)
            state = st.get("status")
            if state == "done":
                return st.get("result") or {}
            if state == "failed":
                raise RuntimeError(st.get("error") or "Remote Kaggle job failed.")
            time.sleep(self.poll_interval)

    def check_health(self) -> dict:
        return self._api_json("GET", "/api/health", timeout=15.0)

    def list_remote_videos(self) -> list[dict]:
        return self._api_json("GET", "/api/videos", timeout=30.0).get("videos", [])

    def pull_remote_videos(self, video_ids: list[str] | None = None, progress: Progress | None = None) -> list[str]:
        prog = progress or (lambda _f, _m="": None)
        remote_vids = self.list_remote_videos()
        targets = [v["video_id"] for v in remote_vids] if video_ids is None else list(video_ids)
        pulled = []
        for idx, vid in enumerate(targets, 1):
            tmp_zip = self.root / f".{vid}_compact.zip"
            prog((idx - 1) / max(1, len(targets)), f"Downloading {vid} (compact FLAC + CSVs)...")
            self._download_file(f"/api/videos/{vid}/compact.zip", tmp_zip, prog, f"Pulling {vid}")
            try:
                unpack_corpus_archive(tmp_zip, self.root, hydrate=True)
            finally:
                tmp_zip.unlink(missing_ok=True)
            pulled.append(vid)
        self._flags_cache.clear()
        self.list_videos()
        prog(1.0, f"Pulled {len(pulled)} video(s) from Kaggle")
        return pulled

    # ------------------------------------------------------------------
    # Overridden Backend API methods (route heavy steps to Kaggle GPU)
    # ------------------------------------------------------------------
    def download(self, url: str, progress: Progress) -> VideoInfo:
        if not self.is_remote:
            return super().download(url, progress)

        cookies_txt = None
        if self.cookies_file and Path(self.cookies_file).exists():
            cookies_txt = Path(self.cookies_file).read_text(encoding="utf-8", errors="replace")

        progress(0.02, "Sending URL to Kaggle (using Kaggle cookies if available)...")
        job = self._api_json("POST", "/api/jobs/download", {"url": url, "cookies_txt": cookies_txt})
        res = self._poll_job(job["job_id"], progress, scale=(0.05, 0.95))
        vid = res["video_id"]
        vdir = self.root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        _write_json(vdir / "info.json", res["info"])
        progress(1.0, "Downloaded on Kaggle")
        return self._refresh(vdir)

    def extract_audio(self, video_id: str, sample_rate: int, channels: int, progress: Progress) -> str:
        vdir = self._dir(video_id)
        wav = vdir / "audio.wav"
        if not self.is_remote or wav.exists():
            return super().extract_audio(video_id, sample_rate, channels, progress)

        if sample_rate != 16000 or channels != 1:
            raise ValueError("The corpus pipeline needs 16 kHz mono 16-bit WAV.")

        flac_path = vdir / "audio_conditioned.flac"
        if not flac_path.exists():
            progress(0.05, "Downloading lossless FLAC audio from Kaggle...")
            self._download_file(f"/api/videos/{video_id}/flac", flac_path, progress, "Downloading FLAC")

        progress(0.92, "Decoding FLAC to 16 kHz WAV locally...")
        hydrate_video_dir(vdir, slice_segments=False)
        progress(1.0, "Audio ready")
        self._refresh(vdir)
        return str(wav)

    def run_pipeline(self, video_id: str, progress: Progress) -> list[Segment]:
        if not self.is_remote:
            return super().run_pipeline(video_id, progress)

        vdir = self._dir(video_id)
        # If video was added locally (e.g. add_local_file), upload compact FLAC once if Kaggle lacks it
        remote_ids = {v["video_id"] for v in self.list_remote_videos()}
        if video_id not in remote_ids:
            progress(0.03, "Uploading compressed FLAC audio to Kaggle...")
            flac_path = ensure_conditioned_flac(vdir)
            self._upload_file(f"/api/videos/{video_id}/upload_flac", flac_path)
            info_file = vdir / "info.json"
            if info_file.exists():
                info = json.loads(info_file.read_text(encoding="utf-8"))
                self._api_json("POST", f"/api/videos/{video_id}/info", info)

        progress(0.08, "Starting Steps 3-6 (AST + Cut Engine + SNR + Dual ASR) on Kaggle GPU...")
        job = self._api_json(
            "POST",
            "/api/jobs/pipeline",
            {
                "video_id": video_id,
                "snr_threshold": self.snr_threshold,
                "snr_combine": self.snr_combine,
                "word_verification": self.word_verification,
            },
        )
        self._poll_job(job["job_id"], progress, scale=(0.10, 0.90))

        # Fetch ONLY tiny metadata bundle (< 500 KB) and slice WAVs locally from audio.wav
        meta_zip = vdir / ".metadata_bundle.zip"
        progress(0.92, "Downloading transcripts & metadata from Kaggle...")
        self._download_file(f"/api/videos/{video_id}/metadata.zip", meta_zip, progress, "Metadata")
        try:
            unpack_corpus_archive(meta_zip, self.root, hydrate=False)
        finally:
            meta_zip.unlink(missing_ok=True)

        progress(0.97, "Slicing local segment WAVs from audio.wav...")
        hydrate_video_dir(vdir, slice_segments=True)

        self._flags_cache.clear()
        (vdir / "gui_state.json").unlink(missing_ok=True)
        segs = self.load_segments(video_id)
        self._update_state(vdir, wav_spans=[[s.start, s.end] for s in segs])
        self._refresh(vdir)
        progress(1.0, "Pipeline (Steps 3-6) finished via Kaggle GPU")
        return segs
