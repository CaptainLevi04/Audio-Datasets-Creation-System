"""Local GPU/CPU Dual ASR engine designed for 4 GB VRAM (e.g. RTX 3050 Laptop).

Model A: NVIDIA Arabic Conformer / FastConformer-Hybrid Large (NeMo, ~0.94 GB VRAM in CTC mode).
Model B: Audar-ASR-V1-Turbo Quantized GGUF (`Q4_K_M` + BF16 `mmproj`, ~2.0 GB VRAM via `llama.cpp`).

Models are loaded strictly one at a time and GPU memory is cleared between phases so peak
VRAM stays well below 4 GB.
"""
from __future__ import annotations

import base64
import csv
import gc
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .review_csv import APP_COLS
from .textutil import compare_transcripts

ProgressCb = Callable[[float, str], None]

SR = 16000
DEFAULT_CONFORMER_MODEL = "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"
DEFAULT_CONFORMER_DECODER = "ctc"
DEFAULT_AUDAR_REPO = "audarai/Audar-ASR-V1-Turbo"
DEFAULT_AUDAR_GGUF = "Audar-ASR-V1-Turbo-Q4_K_M.gguf"
DEFAULT_AUDAR_MMPROJ = "mmproj-Audar-ASR-V1-Turbo.gguf"
AUDAR_SYSTEM_PROMPT = "فرّغ الكلام العربي التالي."

LLAMA_CPP_RELEASE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b12780/"
    "llama-b12780-bin-win-cuda-12.4-x64.zip"
)

_AUDAR_PREFIX_RE = re.compile(r"^\s*language\s+[^\s<]+<asr_text>\s*", re.IGNORECASE)


def _noop_progress(_f: float, _msg: str = "") -> None:
    pass


def free_cuda_memory() -> None:
    """Release cached AST/NeMo/PyTorch models and empty CUDA allocator cache."""
    try:
        from speech_corpus import ast_model, segmentation

        ast_model._AST_CACHE.clear()
        segmentation._SILERO.clear()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def clean_audar_text(raw: str | None) -> str:
    """Normalize raw Audar/Qwen3-ASR output (handles `language None<asr_text>` empty verdict)."""
    text = (raw or "").strip()
    if not text:
        return ""
    if re.match(r"^\s*language\s+None<asr_text>", text, re.IGNORECASE):
        return ""
    text = _AUDAR_PREFIX_RE.sub("", text).strip()
    return text.replace("<asr_text>", "").strip()


def _install_nemo_windows_signal_shim() -> None:
    # NeMo exp_manager expects Unix signal.SIGKILL; Windows only provides SIGTERM.
    if not hasattr(signal, "SIGKILL"):
        signal.SIGKILL = signal.SIGTERM  # type: ignore[attr-defined]


def _subprocess_cuda_env(bin_dir: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    extra: list[str] = []
    if bin_dir is not None:
        extra.append(str(bin_dir))
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.is_dir():
        extra.append(str(torch_lib))
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def ensure_llama_cpp_server(cache_dir: Path | None = None) -> Path:
    """Locate `llama-server` executable or download the prebuilt Windows CUDA release once."""
    on_path = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if on_path:
        return Path(on_path)

    root = cache_dir or (Path.home() / ".cache" / "sautlab" / "llama_cpp")
    root.mkdir(parents=True, exist_ok=True)
    exe_name = "llama-server.exe" if sys.platform.startswith("win") else "llama-server"
    found = next(root.rglob(exe_name), None)
    if found is not None and found.is_file():
        return found

    if not sys.platform.startswith("win"):
        raise FileNotFoundError(
            "llama-server not found on PATH. Install llama.cpp with multimodal audio (mtmd) support."
        )

    import httpx

    zp = root / "llama-cuda.zip"
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for attempt in range(8):
            pos = zp.stat().st_size if zp.exists() else 0
            headers = {"Range": f"bytes={pos}-"} if pos > 0 else {}
            try:
                with client.stream("GET", LLAMA_CPP_RELEASE_URL, headers=headers) as r:
                    if r.status_code == 416:
                        break
                    r.raise_for_status()
                    mode = "ab" if (pos > 0 and r.status_code == 206) else "wb"
                    total = int(r.headers.get("content-length", 0)) + (pos if mode == "ab" else 0)
                    with open(zp, mode) as fh:
                        for chunk in r.iter_bytes(chunk_size=1024 * 512):
                            fh.write(chunk)
                            pos += len(chunk)
                    if not total or pos >= total:
                        break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2.0)

    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(root)
    zp.unlink(missing_ok=True)

    found = next(root.rglob(exe_name), None)
    if found is None or not found.is_file():
        raise RuntimeError(f"Downloaded archive to {root}, but {exe_name} was not found.")
    return found


def ensure_audar_gguf(
    repo_id: str = DEFAULT_AUDAR_REPO,
    gguf_file: str = DEFAULT_AUDAR_GGUF,
    mmproj_file: str = DEFAULT_AUDAR_MMPROJ,
) -> tuple[Path, Path]:
    """Download/resolve quantized Audar GGUF decoder + BF16 audio mmproj via Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    model_path = Path(gguf_file)
    if not model_path.is_file():
        model_path = Path(hf_hub_download(repo_id=repo_id, filename=gguf_file))

    mm_path = Path(mmproj_file)
    if not mm_path.is_file():
        mm_path = Path(hf_hub_download(repo_id=repo_id, filename=mmproj_file))

    return model_path, mm_path


def _resolve_conformer_source(model_id: str) -> tuple[str, bool]:
    """Return `(resolved_ref, is_local_nemo_file)` checking explicit paths and `models/*.nemo`."""
    cand = Path(model_id)
    if cand.is_file():
        return str(cand), True
    repo_models = Path(__file__).resolve().parents[1] / "models"
    if repo_models.is_dir():
        for name in ("speechtotext_ar_ar_conformer.nemo", "conformer_ar.nemo", f"{model_id}.nemo"):
            p = repo_models / name
            if p.is_file():
                return str(p), True
    return model_id, False


def transcribe_conformer(
    wav_paths: list[Path],
    rms_values: list[float] | None = None,
    model_id: str = DEFAULT_CONFORMER_MODEL,
    decoder_type: str = DEFAULT_CONFORMER_DECODER,
    device: str | None = None,
    batch_size: int = 8,
    progress: ProgressCb = _noop_progress,
) -> list[str]:
    """Transcribe WAV clips using NVIDIA Conformer / FastConformer in ~1 GB VRAM, then free GPU."""
    if not wav_paths:
        return []

    free_cuda_memory()
    _install_nemo_windows_signal_shim()

    import torch
    import nemo.collections.asr as nemo_asr

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src, is_local = _resolve_conformer_source(model_id)
    progress(0.05, f"Loading NVIDIA Conformer ({Path(src).name}) on {dev.upper()}...")

    model = None
    try:
        if is_local:
            model = nemo_asr.models.ASRModel.restore_from(src, map_location=torch.device(dev))
        else:
            model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                model_name=src, map_location=torch.device(dev)
            )
        model.eval()
        if decoder_type and hasattr(model, "change_decoding_strategy"):
            try:
                model.change_decoding_strategy(decoder_type=decoder_type)
            except Exception:
                pass

        out: list[str] = []
        total = len(wav_paths)
        for i in range(0, total, batch_size):
            batch_files = [str(p) for p in wav_paths[i : i + batch_size]]
            preds = model.transcribe(
                batch_files, batch_size=len(batch_files), num_workers=0, verbose=False
            )
            # Hybrid NeMo models may return tuple(hypotheses, ...).
            if isinstance(preds, tuple):
                preds = preds[0]
            for k, item in enumerate(preds):
                rms = rms_values[i + k] if rms_values is not None else 1.0
                if rms < 1e-3:
                    out.append("")
                    continue
                txt = item.text if hasattr(item, "text") else str(item)
                out.append(str(txt or "").strip())
            done_n = min(i + len(batch_files), total)
            progress(0.10 + 0.85 * (done_n / total), f"Conformer: {done_n}/{total}")
        return out
    finally:
        del model
        free_cuda_memory()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_llama_server_ready(proc: subprocess.Popen, base_url: str, timeout_s: float = 120.0) -> None:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}):\n{out[-1200:]}")
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.4)
    raise TimeoutError(f"llama-server did not become ready within {timeout_s:.0f}s.")


def _query_llama_server_audio(client, base_url: str, wav_path: Path) -> str:
    # Prefer native OpenAI-compatible `/v1/audio/transcriptions`; fallback to `/v1/chat/completions`.
    with open(wav_path, "rb") as fh:
        files = {"file": (wav_path.name, fh, "audio/wav")}
        data = {
            "language": "Arabic",
            "prompt": AUDAR_SYSTEM_PROMPT,
            "temperature": "0",
            "response_format": "json",
            "max_tokens": "256",
        }
        r = client.post(f"{base_url}/v1/audio/transcriptions", data=data, files=files, timeout=120.0)
    if r.status_code == 200:
        return clean_audar_text(r.json().get("text", ""))

    b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    payload = {
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": AUDAR_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]},
        ],
    }
    r2 = client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120.0)
    r2.raise_for_status()
    msg = r2.json()["choices"][0]["message"]["content"]
    return clean_audar_text(msg)


def transcribe_audar_quantized(
    wav_paths: list[Path],
    rms_values: list[float] | None = None,
    repo_id: str = DEFAULT_AUDAR_REPO,
    gguf_file: str = DEFAULT_AUDAR_GGUF,
    mmproj_file: str = DEFAULT_AUDAR_MMPROJ,
    gpu_layers: int = 99,
    ctx_size: int = 2048,
    server_url: str | None = None,
    progress: ProgressCb = _noop_progress,
) -> list[str]:
    """Transcribe WAV clips using quantized Audar-ASR-V1-Turbo (GGUF Q4_K_M + BF16 mmproj)."""
    if not wav_paths:
        return []

    free_cuda_memory()
    import httpx

    proc: subprocess.Popen | None = None
    try:
        if server_url:
            base_url = server_url.rstrip("/")
        else:
            progress(0.05, "Preparing Audar-ASR-V1-Turbo quantized (Q4_K_M GGUF)...")
            srv_exe = ensure_llama_cpp_server()
            model_path, mmproj_path = ensure_audar_gguf(repo_id, gguf_file, mmproj_file)
            port = _free_local_port()
            base_url = f"http://127.0.0.1:{port}"
            cmd = [
                str(srv_exe),
                "-m",
                str(model_path),
                "--mmproj",
                str(mmproj_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-ngl",
                str(gpu_layers),
                "-c",
                str(ctx_size),
                "--temp",
                "0",
            ]
            progress(0.10, "Starting quantized Audar GPU server...")
            proc = subprocess.Popen(
                cmd,
                env=_subprocess_cuda_env(srv_exe.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            _wait_llama_server_ready(proc, base_url, timeout_s=120.0)

        out: list[str] = []
        total = len(wav_paths)
        with httpx.Client() as client:
            for idx, wp in enumerate(wav_paths):
                rms = rms_values[idx] if rms_values is not None else 1.0
                if rms < 1e-3:
                    out.append("")
                else:
                    out.append(_query_llama_server_audio(client, base_url, wp))
                progress(0.15 + 0.85 * ((idx + 1) / total), f"Audar-Q4: {idx + 1}/{total}")
        return out
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        free_cuda_memory()


def score_transcript_pair(a: str, b: str) -> dict[str, Any]:
    """Apply Step 6 Arabic agreement rule (`accepted` / `human_choice` / `difficult`)."""
    a_s, b_s = (a or "").strip(), (b or "").strip()
    if not a_s or not b_s:
        return {"csv_status": "difficult", "gui_status": "difficult", "cer": 1.0, "diff_words": 999}
    cmp = compare_transcripts(a_s, b_s)
    gui_st = cmp["status"]
    csv_st = "human_choice" if gui_st == "needs_review" else gui_st
    return {
        "csv_status": csv_st,
        "gui_status": gui_st,
        "cer": round(float(cmp["cer"]), 4),
        "diff_words": int(cmp["diff_words"]),
    }


def _resolve_seg_wav(vdir: Path, raw: str) -> Path:
    p = Path(str(raw))
    for cand in (p if p.is_absolute() else None, vdir / p, vdir / "segments" / p.name):
        if cand is not None and cand.is_file():
            return cand
    raise FileNotFoundError(f"Missing segment WAV '{raw}' in {vdir}")


def _load_ckpt(ckpt_path: Path, df_asr: pd.DataFrame, col: str) -> list[str] | None:
    if not ckpt_path.is_file():
        return None
    c_df = pd.read_csv(ckpt_path, dtype={"segment_id": str})
    if [str(x) for x in c_df["segment_id"]] != [str(x) for x in df_asr["segment_id"]]:
        return None
    return [str(x) if pd.notnull(x) else "" for x in c_df[col].tolist()]


def _save_ckpt(ckpt_path: Path, df_asr: pd.DataFrame, col: str, values: list[str]) -> None:
    pd.DataFrame(
        {
            "_row_idx": df_asr["_row_idx"].tolist(),
            "segment_id": df_asr["segment_id"].tolist(),
            "label": df_asr["label"].tolist(),
            col: values,
        }
    ).to_csv(ckpt_path, index=False, encoding="utf-8-sig")


def run_local_dual_asr(
    video_dir: str | Path,
    conformer_model: str = DEFAULT_CONFORMER_MODEL,
    conformer_decoder: str = DEFAULT_CONFORMER_DECODER,
    audar_repo: str = DEFAULT_AUDAR_REPO,
    audar_quant: str = DEFAULT_AUDAR_GGUF,
    audar_mmproj: str = DEFAULT_AUDAR_MMPROJ,
    device: str | None = None,
    force: bool = False,
    progress: ProgressCb = _noop_progress,
) -> pd.DataFrame:
    """Run local Step 6 Dual ASR (`speech` + `mixed` only) and write all corpus CSVs/transcripts."""
    import soundfile as sf

    vdir = Path(video_dir)
    src_csv = next(
        (vdir / n for n in ("segments_snr.csv", "segments_student1.csv", "segments.csv") if (vdir / n).is_file()),
        None,
    )
    if src_csv is None:
        raise FileNotFoundError(f"No segments CSV found in {vdir}")

    snr_df = pd.read_csv(src_csv, dtype={"segment_id": str})
    snr_df["_row_idx"] = np.arange(len(snr_df))
    lbl_norm = snr_df["label"].astype(str).str.strip().str.lower()
    mask = lbl_norm.isin({"speech", "mixed"})

    df_asr = snr_df[mask].copy().reset_index(drop=True)
    df_non = snr_df[~mask].copy().reset_index(drop=True)

    if len(df_asr):
        wav_paths: list[Path] = []
        rms_vals: list[float] = []
        for _, row in df_asr.iterrows():
            rel = row.get("audio_file") if pd.notnull(row.get("audio_file")) else row.get("wav_file")
            wp = _resolve_seg_wav(vdir, str(rel))
            samples, _ = sf.read(str(wp), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            wav_paths.append(wp)
            rms_vals.append(float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0)

        ckpt_a = vdir / "asr_conformer_ckpt.csv"
        txts_a = None if force else _load_ckpt(ckpt_a, df_asr, "model_A_transcript")
        if txts_a is None:
            txts_a = transcribe_conformer(
                wav_paths,
                rms_values=rms_vals,
                model_id=conformer_model,
                decoder_type=conformer_decoder,
                device=device,
                progress=lambda f, m="": progress(0.05 + 0.42 * f, m),
            )
            _save_ckpt(ckpt_a, df_asr, "model_A_transcript", txts_a)

        ckpt_b = vdir / "asr_audar_ckpt.csv"
        txts_b = None if force else _load_ckpt(ckpt_b, df_asr, "model_B_transcript")
        if txts_b is None:
            txts_b = transcribe_audar_quantized(
                wav_paths,
                rms_values=rms_vals,
                repo_id=audar_repo,
                gguf_file=audar_quant,
                mmproj_file=audar_mmproj,
                progress=lambda f, m="": progress(0.48 + 0.45 * f, m),
            )
            _save_ckpt(ckpt_b, df_asr, "model_B_transcript", txts_b)

        scored = [score_transcript_pair(a, b) for a, b in zip(txts_a, txts_b)]
        statuses = [s["csv_status"] for s in scored]
        df_asr["model_A_transcript"] = txts_a
        df_asr["model_B_transcript"] = txts_b
        df_asr["final_transcript"] = [a if st == "accepted" else "" for a, st in zip(txts_a, statuses)]
        df_asr["status"] = statuses
        df_asr["automatic_status"] = statuses
        df_asr["review_status"] = ["accepted" if st == "accepted" else "pending" for st in statuses]
        df_asr["cer"] = [s["cer"] for s in scored]
        df_asr["n_diff_words"] = [s["diff_words"] for s in scored]
        df_asr["second_model"] = "audar_q4"
        df_asr["changed_by"] = ""
        df_asr["changed_at"] = ""
        df_asr["change_log"] = "[]"

    if len(df_non):
        for col in ("model_A_transcript", "model_B_transcript", "final_transcript", "second_model", "changed_by", "changed_at"):
            df_non[col] = ""
        df_non["cer"] = np.nan
        df_non["n_diff_words"] = np.nan
        df_non["change_log"] = "[]"
        df_non["status"] = "no_asr"
        df_non["automatic_status"] = "no_asr"
        df_non["review_status"] = "skipped"

    parts = [p for p in (df_asr, df_non) if len(p) > 0]
    merged = pd.concat(parts, ignore_index=True).sort_values("_row_idx").reset_index(drop=True)
    app_df = merged[[c for c in APP_COLS if c in merged.columns]].copy()
    app_df.to_csv(vdir / "segments.csv", index=False, encoding="utf-8-sig")
    app_df.to_csv(vdir / "segments_asr.csv", index=False, encoding="utf-8-sig")

    full_df = merged.drop(columns=["_row_idx"], errors="ignore").copy()
    s1_path = vdir / "segments_student1.csv"
    diag_path = vdir / "snr_diagnostics.csv"
    if s1_path.is_file():
        s1_df = pd.read_csv(s1_path, dtype={"segment_id": str}).reset_index(drop=True)
        if len(s1_df) == len(full_df):
            if "status" in s1_df.columns:
                full_df["qc_status"] = s1_df["status"].values
            for c in [c for c in s1_df.columns if c not in full_df.columns]:
                full_df[c] = s1_df[c].values
    if diag_path.is_file():
        d_df = pd.read_csv(diag_path, dtype={"segment_id": str}).reset_index(drop=True)
        if len(d_df) == len(full_df):
            for c in ("snr_energy_db", "snr_wada_db", "snr_disagree"):
                if c in d_df.columns:
                    full_df[c] = d_df[c].values
    full_df.to_csv(vdir / "segments_full.csv", index=False, encoding="utf-8-sig")

    tdir = vdir / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    for row in df_asr.itertuples():
        tp = tdir / f"segment_{row.segment_id}.txt"
        if row.status == "accepted":
            tp.write_text(str(row.final_transcript), encoding="utf-8")
        else:
            tp.write_text(
                f"[{row.status}]\nModel A (Conformer): {row.model_A_transcript}\nModel B (Audar): {row.model_B_transcript}\n",
                encoding="utf-8",
            )
    progress(1.0, "Dual ASR finished")
    return app_df


def transcribe_clip_dual_asr(
    samples: np.ndarray,
    sr: int = SR,
    conformer_model: str = DEFAULT_CONFORMER_MODEL,
    conformer_decoder: str = DEFAULT_CONFORMER_DECODER,
    audar_repo: str = DEFAULT_AUDAR_REPO,
    audar_quant: str = DEFAULT_AUDAR_GGUF,
    audar_mmproj: str = DEFAULT_AUDAR_MMPROJ,
    device: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Run Conformer + Audar quantized sequentially on a single audio clip array."""
    import soundfile as sf

    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    with tempfile.TemporaryDirectory(prefix="sautlab_clip_") as tmp:
        wav_path = Path(tmp) / "clip.wav"
        sf.write(str(wav_path), samples.astype(np.float32), sr, subtype="PCM_16")
        a_txt = transcribe_conformer(
            [wav_path],
            rms_values=[rms],
            model_id=conformer_model,
            decoder_type=conformer_decoder,
            device=device,
        )[0]
        b_txt = transcribe_audar_quantized(
            [wav_path],
            rms_values=[rms],
            repo_id=audar_repo,
            gguf_file=audar_quant,
            mmproj_file=audar_mmproj,
        )[0]
    scored = score_transcript_pair(a_txt, b_txt)
    return a_txt, b_txt, scored
