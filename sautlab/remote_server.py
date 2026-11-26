"""Remote Kaggle GPU worker server for SautLab.

Exposes a zero-dependency multi-threaded HTTP API (`ThreadingHTTPServer`) that runs
on Kaggle GPU sessions (optionally behind a Cloudflare quick tunnel) so a user can run
the native desktop PySide6 GUI (`run_gui.py --kaggle-url <url>`) on their own PC
with minimal data transfer (lossless FLAC audio + <1 MB metadata/transcript zips).
"""
from __future__ import annotations

import gc
import glob
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .kaggle_bridge import (
    SR,
    build_corpus_archive,
    ensure_conditioned_flac,
    hydrate_video_dir,
    run_dual_asr_on_video_dir,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_CF_URL_RE = re.compile(r"https://[A-Za-z0-9-]+\.trycloudflare\.com")
_LHR_URL_RE = re.compile(r"https://[A-Za-z0-9-]+\.(?:lhr\.life|localhost\.run)")
_CF_BIN_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

ProgressCb = Callable[[float, str], None]


def _validate_id(val: str, kind: str = "id") -> str:
    v = str(val or "").strip()
    if not v or v in {".", ".."} or not _SAFE_ID.match(v):
        raise ValueError(f"Invalid {kind}: {val!r}")
    return v


def discover_kaggle_cookies(explicit: str | Path | None = None) -> Path | None:
    """Locate a YouTube cookies file from explicit path or Kaggle `/kaggle/input`."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.is_dir():
        matches = sorted(glob.glob(str(kaggle_in / "**" / "*cookies*.txt"), recursive=True))
        for m in matches:
            mp = Path(m)
            if mp.is_file():
                return mp
    return None


def prepare_writable_cookies(
    explicit: str | Path | None = None,
    cookies_txt: str | None = None,
    work_root: str | Path | None = None,
) -> tuple[Path | None, bool]:
    """Create a writable cookies copy outside `data_dir` (Kaggle `/kaggle/input` is read-only).

    Returns `(cookie_path, should_unlink)`.
    """
    tmp_dir = Path(work_root) if work_root else Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f".cookies_remote_{uuid.uuid4().hex[:8]}.txt"

    if cookies_txt and cookies_txt.strip():
        target.write_text(cookies_txt, encoding="utf-8")
        return target, True

    src = discover_kaggle_cookies(explicit)
    if src is not None:
        shutil.copyfile(src, target)
        return target, True

    return None, False


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _download_with_ytdlp_cli(url: str, vdir: Path, cookies_file: Path | None) -> tuple[Path, dict]:
    vdir.mkdir(parents=True, exist_ok=True)
    wav_path = vdir / "audio.wav"
    info_path = vdir / "info.json"

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--write-info-json",
        "-o",
        str(vdir / "temp_audio.%(ext)s"),
        url,
    ]
    if cookies_file is not None:
        cmd[1:1] = ["--cookies", str(cookies_file)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{res.stderr.strip() or res.stdout.strip()}")

    raw_info = vdir / "temp_audio.info.json"
    meta = json.loads(raw_info.read_text(encoding="utf-8")) if raw_info.exists() else {}
    media_files = [f for f in vdir.glob("temp_audio.*") if not f.name.endswith(".json")]
    if not media_files:
        raise RuntimeError("yt-dlp finished but no downloaded media file was found.")

    from speech_corpus.audio import get_ffmpeg_bin

    ffmpeg = get_ffmpeg_bin()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(media_files[0]), "-ar", str(SR), "-ac", "1", str(wav_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    for f in vdir.glob("temp_audio.*"):
        f.unlink(missing_ok=True)

    lic = str(meta.get("license") or "").lower()
    info = {
        "video_id": vdir.name,
        "source": "youtube",
        "url": url,
        "webpage_url": meta.get("webpage_url") or url,
        "title": meta.get("title") or "Unknown Title",
        "channel": meta.get("channel") or meta.get("uploader") or "Unknown Channel",
        "duration_s": meta.get("duration"),
        "upload_date": meta.get("upload_date"),
        "license": meta.get("license") or "Standard YouTube License",
        "is_creative_commons": ("creative commons" in lic) or ("cc-by" in lic) or ("public domain" in lic),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return wav_path, info


def _default_download_video(
    url: str,
    data_dir: Path,
    cookies_file: Path | None,
    progress: ProgressCb,
) -> tuple[str, dict]:
    from speech_corpus.downloader import download_video, extract_video_id, find_cached_video

    vid = extract_video_id(url)
    vdir = data_dir / vid
    vdir.mkdir(parents=True, exist_ok=True)

    info_path = vdir / "info.json"
    if (vdir / "audio.wav").exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        progress(0.85, "Using cached audio on Kaggle")
    else:
        cached = find_cached_video(url, search_dirs=[data_dir])
        if cached is not None:
            wav_path, info = cached
            vdir = wav_path.parent
            progress(0.85, "Using cached video on Kaggle")
        elif shutil.which("yt-dlp"):
            progress(0.15, "Downloading audio on Kaggle (yt-dlp CLI + deno)...")
            _, info = _download_with_ytdlp_cli(url, vdir, cookies_file=cookies_file)
        else:
            progress(0.15, "Downloading video on Kaggle...")
            _, info = download_video(url, vdir, cookies_file=cookies_file)

    progress(0.92, "Compressing lossless FLAC on Kaggle...")
    ensure_conditioned_flac(vdir)
    return vdir.name, info


def _default_run_steps3_to_6(
    vdir: Path,
    params: dict[str, Any],
    progress: ProgressCb,
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from speech_corpus.ast_model import classify_audio, save_regions, validate_cached_regions
    from speech_corpus.audio import read_wav
    from speech_corpus.cut_engine_v2 import cut_speech_and_mixed_segments
    from speech_corpus.quality_control import generate_qc_markdown_report, validate_segments_integrity, video_summary
    from speech_corpus.segmentation import detect_speech_activity_overlay
    from speech_corpus.snr import compute_snr_for_segments, estimate_segment_snr, plot_snr_histogram
    from speech_corpus.visualization import plot_segments_overview, plot_spectrogram_regions

    snr_threshold = float(params.get("snr_threshold", 25.0))
    snr_combine = str(params.get("snr_combine", "mean"))
    word_verif = bool(params.get("word_verification", False))
    accept_cer = float(params.get("accept_cer", 0.02))
    max_word_diff = int(params.get("max_word_diff", 3))
    asr_limit = params.get("asr_limit")

    wav_path = vdir / "audio.wav"
    if not wav_path.exists() and (vdir / "audio_conditioned.flac").exists():
        hydrate_video_dir(vdir, slice_segments=False)
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing audio.wav in {vdir}")

    progress(0.05, "Step 2: Loading 16 kHz audio on Kaggle...")
    audio, _ = read_wav(wav_path, target_sr=SR)
    dur_s = len(audio) / SR
    info_path = vdir / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"video_id": vdir.name}

    progress(0.14, "Step 3/4: Detecting speech pauses (VAD)...")
    _, pauses_df = detect_speech_activity_overlay(
        audio=audio,
        sr=SR,
        silence_threshold_s=0.25,
        min_pause_ms=120,
        vad_backend="auto",
    )

    regions_csv = vdir / "regions.csv"
    windows_csv = vdir / "windows_ast.csv"
    if validate_cached_regions(regions_csv, expected_source_id=vdir.name, expected_duration=dur_s, expected_version="2.2"):
        regions_df = pd.read_csv(regions_csv)
    else:
        progress(0.24, "Step 3: Classifying macro-regions with AST on Kaggle GPU...")
        regions_df, windows_df = classify_audio(y=audio, sr=SR, win_s=10.0, hop_s=5.0, batch_size=32, pauses_df=pauses_df)
        save_regions(regions_df, regions_csv, source_id=vdir.name, duration=dur_s, version="2.2")
        if windows_df is not None and not windows_df.empty:
            windows_df.to_csv(windows_csv, index=False)
    _free_gpu()

    spec_path = vdir / "full_spectrogram.png"
    fig = plot_spectrogram_regions(
        audio,
        regions_df,
        sr=SR,
        title=f"Full Audio Spectrogram - {str(info.get('title', vdir.name))[:40]}",
        save_path=spec_path,
    )
    plt.close(fig)

    progress(0.38, "Step 4: Cutting speech/mixed segments with Cut Engine v2...")
    segments_df = cut_speech_and_mixed_segments(
        audio=audio,
        regions_df=regions_df,
        out_dir=vdir,
        sr=SR,
        pauses=pauses_df,
        use_word_verification=word_verif,
        word_model_size="small",
        run_ablation=True,
    )
    segments_dir = vdir / "segments"
    staging_dir = vdir / "segments_staging"
    if not segments_dir.exists() and staging_dir.exists():
        staging_dir.rename(segments_dir)
        for col in ("audio_file", "wav_file"):
            if col in segments_df.columns:
                segments_df[col] = segments_df[col].map(
                    lambda x: x.replace("segments_staging/", "segments/") if isinstance(x, str) else x
                )
    if not segments_dir.exists():
        raise RuntimeError(f"Cut Engine v2 produced no segments for {vdir.name}")
    segments_df.to_csv(vdir / "segments_student1.csv", index=False)
    _free_gpu()

    fig = plot_segments_overview(audio, segments_df, sr=SR, max_segments=20, save_path=vdir / "segments_overview.png")
    plt.close(fig)
    qc = validate_segments_integrity(segments_df, video_dir=vdir, total_dur=dur_s, sr=SR)
    generate_qc_markdown_report(segments_df, report_path=vdir / "qc_report.md", video_dir=vdir, qc=qc)
    video_summary(regions_df=regions_df, segments_df=segments_df, info=info, video_dir=vdir)
    del audio
    gc.collect()

    progress(0.50, "Step 5: Estimating WADA + Energy SNR...")
    seg_src = segments_df.copy()
    if "wav_file" not in seg_src.columns and "audio_file" in seg_src.columns:
        seg_src["wav_file"] = seg_src["audio_file"].map(
            lambda x: f"segments/{Path(str(x)).name}" if pd.notnull(x) and not (vdir / str(x)).exists() else x
        )
    snr_df, _ = compute_snr_for_segments(
        seg_src,
        base_dir=vdir,
        threshold_db=snr_threshold,
        combine=snr_combine,
        save_csv=True,
        verbose=False,
    )
    snr_df.to_csv(vdir / "segments_snr.csv", index=False, encoding="utf-8-sig")

    diag_rows = []
    for _, row in seg_src.iterrows():
        rel = row["wav_file"] if "wav_file" in seg_src.columns else row["audio_file"]
        wp = vdir / Path(str(rel))
        if not wp.exists():
            wp = vdir / "segments" / Path(str(rel)).name
        samples, sr_w = read_wav(wp, target_sr=SR)
        est = estimate_segment_snr(samples, sr=sr_w, threshold_db=snr_threshold, combine=snr_combine)
        diag_rows.append(
            {
                "segment_id": row["segment_id"],
                "label": row["label"],
                "snr_energy_db": est["snr_energy_db"],
                "snr_wada_db": est["snr_wada_db"],
                "snr": est["snr"],
                "quality": est["quality"],
                "snr_disagree": est["snr_disagree"],
            }
        )
    pd.DataFrame(diag_rows).to_csv(vdir / "snr_diagnostics.csv", index=False, encoding="utf-8-sig")
    fig = plot_snr_histogram(snr_df, threshold_db=snr_threshold, save_path=vdir / "snr_histogram.png")
    plt.close(fig)

    out_df = run_dual_asr_on_video_dir(
        vdir,
        accept_cer=accept_cer,
        max_word_diff=max_word_diff,
        limit=asr_limit,
        force=False,
        progress=lambda f, m="": progress(0.58 + 0.38 * f, m),
    )
    ensure_conditioned_flac(vdir)
    _free_gpu()
    return {"video_id": vdir.name, "segments": len(out_df)}


class RemoteJobManager:
    """Thread-safe job queue & corpus filesystem store for the remote Kaggle worker."""

    def __init__(
        self,
        data_dir: str | Path = "corpus_data",
        work_root: str | Path | None = None,
        cookies_source: str | Path | None = None,
        download_fn: Callable[[str, Path, Path | None, ProgressCb], tuple[str, dict]] | None = None,
        pipeline_fn: Callable[[Path, dict[str, Any], ProgressCb], dict[str, Any]] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.work_root = Path(work_root) if work_root else self.data_dir.parent
        self.cookies_source = Path(cookies_source) if cookies_source else discover_kaggle_cookies()
        self.download_fn = download_fn or _default_download_video
        self.pipeline_fn = pipeline_fn or _default_run_steps3_to_6
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kaggle-gpu")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def health_info(self) -> dict[str, Any]:
        cuda_avail = False
        dev = "cpu"
        try:
            import torch

            cuda_avail = bool(torch.cuda.is_available())
            dev = torch.cuda.get_device_name(0) if cuda_avail else "cpu"
        except Exception:
            pass
        return {
            "status": "ok",
            "cuda": cuda_avail,
            "device": dev,
            "has_cookies": discover_kaggle_cookies(self.cookies_source) is not None,
            "videos_count": len(self.list_videos()),
        }

    def list_videos(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.data_dir.iterdir()):
            if not p.is_dir() or p.name == "splits":
                continue
            info_path = p / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
            out.append(
                {
                    "video_id": p.name,
                    "title": info.get("title") or p.name,
                    "duration_s": info.get("duration_s"),
                    "has_flac": (p / "audio_conditioned.flac").exists(),
                    "has_wav": (p / "audio.wav").exists(),
                    "has_asr": (p / "segments.csv").exists() or (p / "segments_asr.csv").exists(),
                }
            )
        return out

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def submit_download(self, url: str, cookies_txt: str | None = None) -> str:
        url = str(url or "").strip()
        if not url:
            raise ValueError("URL must not be empty.")
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": "download",
                "status": "queued",
                "progress": 0.0,
                "message": "Queued on Kaggle...",
                "result": None,
                "error": None,
            }

        def _worker() -> None:
            cookie_path = None
            should_unlink = False
            try:
                self._update_job(job_id, status="running", progress=0.05, message="Preparing cookies & download...")
                cookie_path, should_unlink = prepare_writable_cookies(
                    explicit=self.cookies_source,
                    cookies_txt=cookies_txt,
                    work_root=self.work_root,
                )
                vid, info = self.download_fn(
                    url,
                    self.data_dir,
                    cookie_path,
                    lambda f, m="": self._update_job(job_id, progress=float(f), message=str(m)),
                )
                vid = _validate_id(vid, "video_id")
                ensure_conditioned_flac(self.data_dir / vid)
                self._update_job(
                    job_id,
                    status="done",
                    progress=1.0,
                    message="Downloaded on Kaggle",
                    result={"video_id": vid, "info": info},
                )
            except Exception as exc:
                self._update_job(job_id, status="failed", error=str(exc), message=f"Download failed: {exc}")
            finally:
                if should_unlink and cookie_path is not None:
                    cookie_path.unlink(missing_ok=True)

        self._executor.submit(_worker)
        return job_id

    def submit_pipeline(self, video_id: str, params: dict[str, Any]) -> str:
        vid = _validate_id(video_id, "video_id")
        vdir = self.data_dir / vid
        if not vdir.is_dir():
            raise FileNotFoundError(f"Unknown video_id on Kaggle: {vid}")

        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": "pipeline",
                "status": "queued",
                "progress": 0.0,
                "message": f"Queued {vid} for Kaggle GPU...",
                "result": None,
                "error": None,
            }

        def _worker() -> None:
            try:
                self._update_job(job_id, status="running", progress=0.02, message=f"Running Steps 3-6 for {vid}...")
                res = self.pipeline_fn(
                    vdir,
                    dict(params),
                    lambda f, m="": self._update_job(job_id, progress=float(f), message=str(m)),
                )
                self._update_job(
                    job_id,
                    status="done",
                    progress=1.0,
                    message="Pipeline (Steps 3-6) finished on Kaggle GPU",
                    result=res,
                )
            except Exception as exc:
                self._update_job(job_id, status="failed", error=str(exc), message=f"Pipeline failed: {exc}")

        self._executor.submit(_worker)
        return job_id


def _make_handler(manager: RemoteJobManager):
    class KaggleRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"File not found: {path.name}"})
                return
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, length=256 * 1024)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length > 0 else b""

        def do_GET(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            parts = [p for p in path.split("/") if p]
            try:
                if parts == ["api", "health"]:
                    self._send_json(HTTPStatus.OK, manager.health_info())
                    return

                if parts == ["api", "videos"]:
                    self._send_json(HTTPStatus.OK, {"videos": manager.list_videos()})
                    return

                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    job_id = _validate_id(parts[2], "job_id")
                    job = manager.get_job(job_id)
                    if job is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown job {job_id}"})
                        return
                    self._send_json(HTTPStatus.OK, job)
                    return

                if len(parts) == 4 and parts[:2] == ["api", "videos"]:
                    vid = _validate_id(parts[2], "video_id")
                    action = parts[3]
                    vdir = manager.data_dir / vid
                    if not vdir.is_dir():
                        self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown video {vid}"})
                        return

                    if action == "flac":
                        flac = ensure_conditioned_flac(vdir)
                        self._send_file(flac, "audio/flac")
                        return

                    if action in {"metadata.zip", "compact.zip"}:
                        mode = "metadata" if action == "metadata.zip" else "compact"
                        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{vid}_{mode}_", suffix=".zip")
                        os.close(tmp_fd)
                        tmp_zip = Path(tmp_name)
                        try:
                            build_corpus_archive(manager.data_dir, tmp_zip, mode=mode, video_ids=[vid])
                            self._send_file(tmp_zip, "application/zip")
                        finally:
                            tmp_zip.unlink(missing_ok=True)
                        return

                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown GET route: {path}"})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(exc)})

        def do_POST(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            parts = [p for p in path.split("/") if p]
            try:
                if parts == ["api", "jobs", "download"]:
                    payload = json.loads(self._read_body().decode("utf-8") or "{}")
                    job_id = manager.submit_download(
                        url=payload.get("url", ""),
                        cookies_txt=payload.get("cookies_txt"),
                    )
                    self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
                    return

                if parts == ["api", "jobs", "pipeline"]:
                    payload = json.loads(self._read_body().decode("utf-8") or "{}")
                    job_id = manager.submit_pipeline(payload.get("video_id", ""), payload)
                    self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
                    return

                if len(parts) == 4 and parts[:2] == ["api", "videos"]:
                    vid = _validate_id(parts[2], "video_id")
                    action = parts[3]
                    vdir = manager.data_dir / vid
                    vdir.mkdir(parents=True, exist_ok=True)

                    if action == "upload_flac":
                        raw = self._read_body()
                        flac = vdir / "audio_conditioned.flac"
                        flac.write_bytes(raw)
                        hydrate_video_dir(vdir, slice_segments=False)
                        self._send_json(HTTPStatus.OK, {"status": "ok", "video_id": vid, "bytes": len(raw)})
                        return

                    if action == "info":
                        payload = json.loads(self._read_body().decode("utf-8") or "{}")
                        (vdir / "info.json").write_text(
                            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        self._send_json(HTTPStatus.OK, {"status": "ok", "video_id": vid})
                        return

                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown POST route: {path}"})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(exc)})

    return KaggleRequestHandler


def create_remote_http_server(
    data_dir: str | Path = "corpus_data",
    host: str = "127.0.0.1",
    port: int = 8765,
    cookies_source: str | Path | None = None,
    work_root: str | Path | None = None,
    manager: RemoteJobManager | None = None,
) -> tuple[ThreadingHTTPServer, RemoteJobManager]:
    """Create the HTTP server and backing `RemoteJobManager` (pass `port=0` for an ephemeral port)."""
    mgr = manager or RemoteJobManager(
        data_dir=data_dir,
        work_root=work_root,
        cookies_source=cookies_source,
    )
    server = ThreadingHTTPServer((host, int(port)), _make_handler(mgr))
    server.daemon_threads = True
    return server, mgr


def _ensure_cloudflared() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    local_bin = Path(tempfile.gettempdir()) / "cloudflared"
    if not local_bin.exists():
        print("⬇️  Downloading cloudflared binary for HTTPS tunnel...")
        urllib.request.urlretrieve(_CF_BIN_URL, str(local_bin))
        local_bin.chmod(local_bin.stat().st_mode | stat.S_IEXEC)
    return str(local_bin)


def start_cloudflare_tunnel(port: int, timeout_s: float = 20.0) -> tuple[subprocess.Popen, str]:
    """Launch a Cloudflare quick tunnel pointing at `http://127.0.0.1:{port}`."""
    bin_path = _ensure_cloudflared()
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    url: str | None = None
    rate_limited = False
    lines: list[str] = []
    ev = threading.Event()

    def _reader() -> None:
        nonlocal url, rate_limited
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            m = _CF_URL_RE.search(line)
            if m and url is None:
                url = m.group(0)
                ev.set()
            elif "status 429" in line or "Tunnelstore request failed" in line:
                rate_limited = True
                ev.set()

    threading.Thread(target=_reader, daemon=True).start()
    ev.wait(timeout=timeout_s)
    if not url:
        proc.terminate()
        tail = "".join(lines[-20:])
        reason = "Cloudflare quick tunnel returned 429 rate-limit on this IP" if rate_limited else f"cloudflared tunnel did not publish a URL within {timeout_s}s"
        raise RuntimeError(f"{reason}.\n{tail}")
    return proc, url


def start_localhost_run_tunnel(port: int, timeout_s: float = 25.0) -> tuple[subprocess.Popen, str]:
    """Launch an SSH HTTPS tunnel via localhost.run (no API account or token needed)."""
    ssh_bin = shutil.which("ssh")
    if not ssh_bin:
        raise RuntimeError("`ssh` binary is not available for localhost.run fallback.")
    proc = subprocess.Popen(
        [
            ssh_bin,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=30",
            "-R", f"80:127.0.0.1:{port}",
            "nokey@localhost.run",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    url: str | None = None
    lines: list[str] = []
    ev = threading.Event()

    def _reader() -> None:
        nonlocal url
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            m = _LHR_URL_RE.search(line)
            if m and url is None:
                url = m.group(0)
                ev.set()

    threading.Thread(target=_reader, daemon=True).start()
    if not ev.wait(timeout=timeout_s) or not url:
        proc.terminate()
        tail = "".join(lines[-20:])
        raise RuntimeError(f"localhost.run tunnel did not publish a URL within {timeout_s}s.\n{tail}")
    return proc, url


def start_remote_server(
    data_dir: str | Path = "corpus_data",
    port: int = 8765,
    tunnel: str | None = "auto",
    cookies_source: str | Path | None = None,
    work_root: str | Path | None = None,
    block: bool = True,
) -> dict[str, Any]:
    """Start the Kaggle remote GPU worker server and print local GUI connection commands."""
    server, mgr = create_remote_http_server(
        data_dir=data_dir,
        host="127.0.0.1",
        port=port,
        cookies_source=cookies_source,
        work_root=work_root,
    )
    actual_port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    tunnel_proc = None
    public_url = f"http://127.0.0.1:{actual_port}"
    try:
        if tunnel in {"auto", "cloudflared"}:
            try:
                tunnel_proc, public_url = start_cloudflare_tunnel(actual_port)
            except Exception as cf_err:
                if tunnel == "auto" or "429" in str(cf_err):
                    print(f"⚠️  Cloudflare tunnel unavailable ({str(cf_err).splitlines()[0]}); trying localhost.run SSH tunnel...")
                    tunnel_proc, public_url = start_localhost_run_tunnel(actual_port)
                else:
                    raise
        elif tunnel == "localhost.run":
            tunnel_proc, public_url = start_localhost_run_tunnel(actual_port)
    except Exception:
        _stop_server(server, mgr, tunnel_proc)
        raise

    health = mgr.health_info()
    print("=" * 72)
    print("🚀 SautLab Kaggle Remote GPU Server is LIVE")
    print("=" * 72)
    print(f"   Public URL     : {public_url}")
    print(f"   Device         : {health['device']} (CUDA={health['cuda']})")
    print(f"   Kaggle Cookies : {'DETECTED' if health['has_cookies'] else 'Not found'}")
    print(f"   Videos cached  : {health['videos_count']}")
    print("")
    print("💻 Run on your PC:")
    print(f'   python run_gui.py --kaggle-url "{public_url}"')
    print("")
    print("⏹️  Interrupt the Kaggle cell (Stop button) when you finish in the local GUI.")
    print("=" * 72)

    handle = {
        "url": public_url,
        "local_url": f"http://127.0.0.1:{actual_port}",
        "server": server,
        "manager": mgr,
        "tunnel_proc": tunnel_proc,
        "stop": lambda: _stop_server(server, mgr, tunnel_proc),
    }
    if not block:
        return handle

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Kaggle Remote GPU Server...")
    finally:
        _stop_server(server, mgr, tunnel_proc)
    return handle


def _stop_server(server: ThreadingHTTPServer, mgr: RemoteJobManager, tunnel_proc: subprocess.Popen | None) -> None:
    try:
        server.shutdown()
        server.server_close()
    except Exception:
        pass
    mgr.shutdown()
    if tunnel_proc is not None and tunnel_proc.poll() is None:
        tunnel_proc.terminate()
