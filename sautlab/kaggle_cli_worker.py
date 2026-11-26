"""Kaggle-side CLI job worker (runs Steps 1-6 inside Kaggle GPU kernels).

Features:
- Installs dependencies once per container (`ensure_kaggle_deps`) and skips pip on
  warm restarts / subsequent links.
- Executes each video job inside a fresh Python subprocess so PyTorch/CUDA/RAM state
  restarts cleanly between links without losing installed dependencies or GPU slot.
- Automatically discovers read-only `/kaggle/input/**/*cookies*.txt` and copies a
  writable temporary cookie jar per YouTube download.
- Publishes compact archives directly into the user's named Kaggle dataset when
  `KAGGLE_API_TOKEN` is available in Kaggle Secrets, or falls back seamlessly to
  saving `/kaggle/working/corpus_final_output.zip` for single-shot `kaggle kernels output`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .kaggle_bridge import (
    _get_authenticated_kaggle_api,
    build_corpus_archive,
    download_file_from_kaggle_dataset,
    ensure_conditioned_flac,
    upload_files_to_kaggle_dataset,
)
from .remote_server import (
    _default_download_video,
    _default_run_steps3_to_6,
    prepare_writable_cookies,
)

_REQUIRED_IMPORTS = ("qwen_asr", "faster_whisper", "jiwer", "soundfile", "librosa", "kaggle")


def ensure_kaggle_deps(work_root: str | Path = "/kaggle/working") -> bool:
    """Install pipeline deps once; skip download/install if already available."""
    work = Path(work_root)
    marker = work / ".deps_ready"

    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.is_dir() and str(deno_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{deno_bin}:{os.environ['PATH']}"

    has_imports = all(importlib.util.find_spec(mod) is not None for mod in _REQUIRED_IMPORTS)
    has_ytdlp = shutil.which("yt-dlp") is not None
    has_deno = shutil.which("deno") is not None

    if has_imports and has_ytdlp and has_deno:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")
        print("⚡ Dependencies already installed — skipping pip/deno download.")
        return False

    print("📦 Installing Kaggle pipeline dependencies (first run in this container)...")
    if not has_ytdlp:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp[default]"], check=False)
    missing_pkgs = [
        pkg for mod, pkg in (
            ("qwen_asr", "qwen-asr"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
            ("jiwer", "jiwer"),
            ("faster_whisper", "faster-whisper"),
            ("kaggle", "kaggle"),
        )
        if importlib.util.find_spec(mod) is None
    ]
    if missing_pkgs:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing_pkgs], check=True)
    if not has_deno:
        subprocess.run(
            "curl -fsSL https://deno.land/install.sh | sh",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if deno_bin.is_dir() and str(deno_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{deno_bin}:{os.environ['PATH']}"

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok", encoding="utf-8")
    return True


def _load_mounted_queue_job(queue_slug: str = "sautlab-queue") -> dict[str, Any] | None:
    for root in (Path(f"/kaggle/input/{queue_slug}"), Path("/kaggle/input")):
        if not root.exists():
            continue
        direct = root / "job.json"
        if direct.is_file():
            return json.loads(direct.read_text(encoding="utf-8"))
        for match in sorted(root.rglob("job.json")):
            return json.loads(match.read_text(encoding="utf-8"))
    return None


def process_single_job(
    job: dict[str, Any],
    data_dir: str | Path,
    work_root: str | Path,
    out_zip: str | Path,
) -> dict[str, Any]:
    """Execute download + Steps 3-6 for one job dict and build a compact zip archive."""
    ddir = Path(data_dir)
    wroot = Path(work_root)
    ddir.mkdir(parents=True, exist_ok=True)
    wroot.mkdir(parents=True, exist_ok=True)
    zip_out = Path(out_zip)

    url = str(job.get("url") or "").strip()
    video_id = str(job.get("video_id") or "").strip()
    job_id = str(job.get("job_id") or "job")
    zip_mode = str(job.get("zip_mode") or "compact")

    def _log(_f: float, m: str = "") -> None:
        if m:
            print(f"[{job_id}] {m}", flush=True)

    cookie_path = None
    should_unlink = False
    try:
        cookie_path, should_unlink = prepare_writable_cookies(
            explicit=job.get("cookies_file"),
            cookies_txt=job.get("cookies_txt"),
            work_root=wroot,
        )
        if url:
            vid, _info = _default_download_video(url, ddir, cookie_path, _log)
        elif video_id:
            vdir_target = ddir / video_id
            vdir_target.mkdir(parents=True, exist_ok=True)
            if not (vdir_target / "audio.wav").exists():
                for cand in (
                    wroot / ".queue_poll" / "audio_conditioned.flac",
                    Path(f"/kaggle/input/{job.get('queue_slug', 'sautlab-queue')}/audio_conditioned.flac"),
                ):
                    if cand.is_file():
                        shutil.copy2(cand, vdir_target / "audio_conditioned.flac")
                        from .kaggle_bridge import hydrate_video_dir
                        hydrate_video_dir(vdir_target, slice_segments=False)
                        break
            if not (vdir_target / "audio.wav").exists():
                raise ValueError(f"Missing audio.wav or queued audio_conditioned.flac for video_id={video_id}")
            vid = video_id
        else:
            raise ValueError(f"Invalid Kaggle worker job: missing url or cached video_id ({job})")
    finally:
        if should_unlink and cookie_path is not None:
            cookie_path.unlink(missing_ok=True)

    vdir = ddir / vid
    summary = _default_run_steps3_to_6(vdir, dict(job), _log)
    ensure_conditioned_flac(vdir)
    build_corpus_archive(ddir, zip_out, mode=zip_mode, video_ids=[vid])

    result = {
        "job_id": job_id,
        "video_id": vid,
        "status": "done",
        "segments": summary.get("segments", 0),
        "zip_name": zip_out.name,
        "dataset_slug": job.get("dataset_slug"),
        "finished_at": time.time(),
    }
    return result


def _run_job_in_fresh_subprocess(
    job: dict[str, Any],
    data_dir: Path,
    work_root: Path,
    out_zip: Path,
) -> dict[str, Any]:
    """Run one video inside a fresh Python process so RAM/VRAM reset cleanly per video."""
    job_file = work_root / f".job_{job.get('job_id', 'cur')}.json"
    res_file = work_root / f".result_{job.get('job_id', 'cur')}.json"
    job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
    res_file.unlink(missing_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sautlab.kaggle_cli_worker",
            "--run-job-file",
            str(job_file),
            "--data-dir",
            str(data_dir),
            "--work-root",
            str(work_root),
            "--out-zip",
            str(out_zip),
            "--result-file",
            str(res_file),
        ],
        env=env,
        check=False,
    )
    try:
        if proc.returncode != 0:
            if res_file.exists():
                err = json.loads(res_file.read_text(encoding="utf-8")).get("error")
                raise RuntimeError(err or f"Worker subprocess failed with exit code {proc.returncode}")
            raise RuntimeError(f"Worker subprocess failed with exit code {proc.returncode}")
        return json.loads(res_file.read_text(encoding="utf-8"))
    finally:
        job_file.unlink(missing_ok=True)
        res_file.unlink(missing_ok=True)


def run_worker_session(
    initial_job: dict[str, Any] | None = None,
    data_dir: str | Path = "/kaggle/working/corpus_data",
    work_root: str | Path = "/kaggle/working",
    queue_slug: str = "sautlab-queue",
    poll_interval_s: float = 5.0,
    max_idle_s: float = 1800.0,
    use_subprocess: bool = True,
) -> dict[str, Any] | None:
    """Boot Kaggle GPU worker: keep alive & poll for new jobs when authenticated, else run once."""
    wroot = Path(work_root)
    ddir = Path(data_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    ensure_kaggle_deps(wroot)

    out_zip = wroot / "corpus_final_output.zip"
    status_path = wroot / "job_status.json"

    cur_job = dict(initial_job) if initial_job else _load_mounted_queue_job(queue_slug=queue_slug)
    api, username = _get_authenticated_kaggle_api()
    persistent = api is not None and bool(username)

    def _exec(j: dict[str, Any]) -> dict[str, Any]:
        if use_subprocess:
            return _run_job_in_fresh_subprocess(j, ddir, wroot, out_zip)
        return process_single_job(j, ddir, wroot, out_zip)

    last_job_id: str | None = None
    last_result: dict[str, Any] | None = None

    if not persistent:
        if cur_job is None:
            raise RuntimeError("No job provided and no /kaggle/input/sautlab-queue/job.json mounted.")
        try:
            res = _exec(cur_job)
            res["worker_mode"] = "oneshot"
            status_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            return res
        except Exception as exc:
            err_payload = {"job_id": cur_job.get("job_id"), "status": "failed", "error": str(exc), "worker_mode": "oneshot"}
            status_path.write_text(json.dumps(err_payload, indent=2), encoding="utf-8")
            raise

    # Persistent session loop: keeps container & pip deps alive until GUI sends shutdown!
    last_activity = time.time()
    print(f"🔁 Persistent Kaggle CLI session LIVE for user={username} (poll={poll_interval_s}s)")
    while True:
        if cur_job is not None:
            action = str(cur_job.get("action") or "process").lower()
            jid = str(cur_job.get("job_id") or "")
            ds_slug = str(cur_job.get("dataset_slug") or "sautlab-corpus")

            if action == "shutdown":
                print("🛑 Shutdown requested by local GUI — closing Kaggle session cleanly.")
                break

            if action == "process" and jid and jid != last_job_id:
                last_job_id = jid
                last_activity = time.time()
                print(f"\n▶ Running job {jid} -> dataset={ds_slug} (fresh subprocess, deps cached)")
                try:
                    res = _exec(cur_job)
                    res["worker_mode"] = "persistent"
                    status_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
                    upload_files_to_kaggle_dataset(
                        ds_slug,
                        {"corpus_final_output.zip": out_zip, "job_status.json": status_path},
                        version_notes=f"Completed {res.get('video_id', '')} ({jid})",
                        public=False,
                    )
                    last_result = res
                    print(f"✅ Published {res.get('video_id')} to Kaggle dataset {username}/{ds_slug}")
                except Exception as exc:
                    err_payload = {
                        "job_id": jid,
                        "status": "failed",
                        "error": str(exc),
                        "worker_mode": "persistent",
                    }
                    status_path.write_text(json.dumps(err_payload, indent=2), encoding="utf-8")
                    try:
                        upload_files_to_kaggle_dataset(
                            ds_slug,
                            {"job_status.json": status_path},
                            version_notes=f"Failed {jid}",
                            public=False,
                        )
                    except Exception:
                        pass
                cur_job = None
                continue

        if (time.time() - last_activity) > max_idle_s:
            print(f"⏱️  Idle timeout reached ({max_idle_s}s) — shutting down Kaggle session.")
            break

        time.sleep(poll_interval_s)
        tmp_dl = wroot / ".queue_poll"
        qfile = download_file_from_kaggle_dataset(f"{username}/{queue_slug}", "job.json", tmp_dl)
        if qfile is not None and qfile.is_file():
            try:
                cur_job = json.loads(qfile.read_text(encoding="utf-8"))
                if cur_job and cur_job.get("action") == "process" and not cur_job.get("url"):
                    download_file_from_kaggle_dataset(f"{username}/{queue_slug}", "audio_conditioned.flac", tmp_dl)
            except Exception:
                cur_job = None

    return last_result


def _cli() -> int:
    ap = argparse.ArgumentParser(description="SautLab Kaggle CLI job runner")
    ap.add_argument("--run-job-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--out-zip", required=True)
    ap.add_argument("--result-file", required=True)
    args = ap.parse_args()

    job = json.loads(Path(args.run_job_file).read_text(encoding="utf-8"))
    try:
        res = process_single_job(job, args.data_dir, args.work_root, args.out_zip)
        Path(args.result_file).write_text(json.dumps(res, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        err = {"job_id": job.get("job_id"), "status": "failed", "error": str(exc)}
        Path(args.result_file).write_text(json.dumps(err, indent=2), encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
