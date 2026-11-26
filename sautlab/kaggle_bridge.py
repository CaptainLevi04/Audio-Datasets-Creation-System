"""Shared Kaggle <-> Local GUI bridge utilities.

Provides:
1. Lossless FLAC audio conditioning & local hydration (FLAC -> 16 kHz PCM_16 WAV + segment slicing)
   to cut Kaggle -> PC transfer size by ~4-5x without duplicating audio.
2. Compact and metadata-only archive builders / extractors.
3. Unified Step 6 Dual ASR runner (Whisper large-v3 + Audar-ASR-V1-Turbo on speech & mixed ONLY)
   shared by both `main-runner.ipynb` and `sautlab.remote_server`.
"""
from __future__ import annotations

import gc
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000

APP_COLS = [
    "start", "end", "label", "snr", "quality",
    "model_A_transcript", "model_B_transcript", "final_transcript",
    "status", "review_status", "automatic_status",
    "segment_id", "audio_file",
    "changed_by", "changed_at", "change_log",
]

BLANK_META = {
    "model_A_transcript": "", "model_B_transcript": "", "final_transcript": "",
    "cer": np.nan, "n_diff_words": np.nan, "second_model": "",
    "changed_by": "", "changed_at": "", "change_log": "[]",
}

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})

ProgressCb = Callable[[float, str], None]


def _noop_progress(_f: float, _msg: str = "") -> None:
    pass


# ---------------------------------------------------------------------
# 1. Lossless FLAC compression & local WAV/segment hydration
# ---------------------------------------------------------------------

def ensure_conditioned_flac(video_dir: str | Path) -> Path:
    """Create `audio_conditioned.flac` from `audio.wav` if missing."""
    vdir = Path(video_dir)
    flac_path = vdir / "audio_conditioned.flac"
    wav_path = vdir / "audio.wav"
    if flac_path.exists() and flac_path.stat().st_size > 1000:
        return flac_path
    if not wav_path.exists():
        raise FileNotFoundError(f"Neither audio_conditioned.flac nor audio.wav found in {vdir}")

    from speech_corpus.audio import extract_audio_conditioned

    try:
        return extract_audio_conditioned(wav_path, flac_path, sample_rate=SR)
    except Exception:
        data, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        sf.write(str(flac_path), data, sr, format="FLAC")
        return flac_path


def hydrate_video_dir(video_dir: str | Path, slice_segments: bool = True) -> Path:
    """Decode `audio_conditioned.flac` -> `audio.wav` and optionally slice `segments/*.wav`."""
    vdir = Path(video_dir)
    wav_path = vdir / "audio.wav"
    flac_path = vdir / "audio_conditioned.flac"

    if not wav_path.exists() and flac_path.exists():
        from speech_corpus.audio import read_wav, write_wav

        samples, sr = read_wav(flac_path, target_sr=SR)
        write_wav(wav_path, samples, sample_rate=sr)

    if slice_segments and wav_path.exists():
        csv_path = next(
            (vdir / n for n in ("segments.csv", "segments_snr.csv", "segments_student1.csv") if (vdir / n).exists()),
            None,
        )
        if csv_path is not None:
            df = pd.read_csv(csv_path, dtype={"segment_id": str})
            seg_dir = vdir / "segments"
            expected = len(df)
            have = len(list(seg_dir.glob("segment_*.wav"))) if seg_dir.is_dir() else 0
            if expected > 0 and have < expected:
                from speech_corpus.audio import read_wav, write_wav
                from speech_corpus.cut_engine_v2 import apply_hann_fade

                samples, sr = read_wav(wav_path, target_sr=SR)
                seg_dir.mkdir(parents=True, exist_ok=True)
                for i, r in enumerate(df.sort_values("start").itertuples(), 1):
                    sid = str(getattr(r, "segment_id", f"{i:04d}")).strip()
                    sid = sid.zfill(4) if sid.isdigit() else f"{i:04d}"
                    out_wav = seg_dir / f"segment_{sid}.wav"
                    if out_wav.exists():
                        continue
                    a = max(0, int(round(float(r.start) * sr)))
                    b = min(len(samples), int(round(float(r.end) * sr)))
                    write_wav(out_wav, apply_hann_fade(samples[a:b].copy(), sr=sr), sample_rate=sr)
    return wav_path


# ---------------------------------------------------------------------
# 2. Compact / metadata-only archive creation & extraction
# ---------------------------------------------------------------------

def build_corpus_archive(
    data_root: str | Path,
    zip_path: str | Path,
    mode: str = "compact",  # "compact" | "full" | "metadata"
    video_ids: list[str] | None = None,
) -> Path:
    """Package corpus outputs cleanly.

    - `compact`: includes `audio_conditioned.flac` + CSVs/JSONs/PNGs/MD/transcripts, skipping
      `audio.wav`, `segments/*.wav`, and `video.mp4` (~4-5x smaller; hydrated locally in ~1s).
    - `full`: includes `audio.wav` and `segments/*.wav` (skips `audio_conditioned.flac` & `video.mp4`).
    - `metadata`: includes ONLY CSVs/JSONs/PNGs/MD/transcripts (no audio files, <1 MB).
    """
    data_root = Path(data_root)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "compact":
        for vdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
            if video_ids is not None and vdir.name not in video_ids:
                continue
            if (vdir / "audio.wav").exists() or (vdir / "audio_conditioned.flac").exists():
                ensure_conditioned_flac(vdir)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(data_root.rglob("*")):
            if not f.is_file() or f.name.endswith(".zip"):
                continue
            rel_to_root = f.relative_to(data_root)
            if video_ids is not None and rel_to_root.parts and rel_to_root.parts[0] not in video_ids:
                # skip unrelated video folders when exporting a specific video
                if (data_root / rel_to_root.parts[0]).is_dir():
                    continue

            if f.name in {"video.mp4", ".cookies.txt"} or "segments_staging" in rel_to_root.parts:
                continue
            if mode == "compact":
                if f.name == "audio.wav" or "segments" in rel_to_root.parts:
                    continue
            elif mode == "full":
                if f.name == "audio_conditioned.flac":
                    continue
            elif mode == "metadata":
                if f.name in {"audio.wav", "audio_conditioned.flac"} or "segments" in rel_to_root.parts or f.suffix.lower() in {".wav", ".flac", ".mp4", ".mkv", ".webm"}:
                    continue
            else:
                raise ValueError(f"Unknown archive mode: {mode}")

            arc = Path("corpus_data") / rel_to_root
            zf.write(f, arcname=arc)
    return zip_path


def unpack_corpus_archive(
    zip_path: str | Path,
    target_data_root: str | Path,
    hydrate: bool = True,
) -> list[str]:
    """Extract a Kaggle output zip into `target_data_root` (safely handling root layout) and hydrate."""
    zip_path = Path(zip_path)
    target = Path(target_data_root)
    target.mkdir(parents=True, exist_ok=True)

    touched_videos: set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            if not parts or ".." in parts:
                continue
            # Strip leading "corpus_data/" prefix if present
            if parts[0] == "corpus_data":
                parts = parts[1:]
            if not parts:
                continue
            out_file = target.joinpath(*parts)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(out_file, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if len(parts) >= 2 and parts[0] != "splits":
                touched_videos.add(parts[0])

    if hydrate:
        for vid in sorted(touched_videos):
            vdir = target / vid
            if (vdir / "audio_conditioned.flac").exists() or (vdir / "audio.wav").exists():
                hydrate_video_dir(vdir, slice_segments=True)
    return sorted(touched_videos)


# ---------------------------------------------------------------------
# 3. Arabic Normalization, jiwer Comparison & Dual ASR Runner (Step 6)
# ---------------------------------------------------------------------

def normalize_ar(t) -> str:
    t = "" if t is None or (isinstance(t, float) and t != t) else str(t)
    t = _DIACRITICS.sub("", t).replace("ـ", "").translate(_ALEF_MAP)
    t = t.replace("ة", "ه").replace("ى", "ي")
    t = re.sub(r"[^\w\s]", " ", t).replace("_", " ")
    return " ".join(t.split())


def compare_transcripts(a, b, accept_cer: float = 0.02, max_word_diff: int = 3) -> tuple[str, float, int]:
    import jiwer

    na, nb = normalize_ar(a), normalize_ar(b)
    if not na or not nb:
        return "difficult", 1.0, 999
    w = jiwer.process_words(na, nb)
    n_diff = int(w.substitutions + w.deletions + w.insertions)
    cer = float(jiwer.cer(na, nb))
    if cer <= accept_cer:
        return "accepted", round(cer, 4), n_diff
    status = "human_choice" if n_diff <= max_word_diff else "difficult"
    return status, round(cer, 4), n_diff


def _resolve_segment_wav(video_dir: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    for c in (
        p if p.is_absolute() else None,
        video_dir / p,
        video_dir.parent / p,
        video_dir.parent.parent / p,
        video_dir / "segments" / p.name,
        video_dir / "segments_staging" / p.name,
    ):
        if c is not None and c.exists():
            return c
    raise FileNotFoundError(f"Missing segment WAV '{raw_path}' in {video_dir}")


def run_dual_asr_on_video_dir(
    video_dir: str | Path,
    whisper_id: str = "openai/whisper-large-v3",
    whisper_lang: str = "ar",
    whisper_batch: int = 4,
    whisper_beams: int = 5,
    audar_id: str = "audarai/Audar-ASR-V1-Turbo",
    audar_lang: str = "Arabic",
    audar_batch: int = 8,
    accept_cer: float = 0.02,
    max_word_diff: int = 3,
    limit: int | None = None,
    force: bool = False,
    progress: ProgressCb = _noop_progress,
) -> pd.DataFrame:
    """Run Step 6 Dual ASR strictly on `speech` and `mixed` segments inside `video_dir`."""
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from speech_corpus.audio import read_wav

    vdir = Path(video_dir)
    snr_csv = vdir / "segments_snr.csv"
    if not snr_csv.exists():
        raise FileNotFoundError(f"Missing segments_snr.csv in {vdir}")

    snr_df = pd.read_csv(snr_csv, dtype={"segment_id": str})
    snr_df["_row_idx"] = np.arange(len(snr_df))
    lbl_norm = snr_df["label"].astype(str).str.strip().str.lower()
    asr_mask = lbl_norm.isin({"speech", "mixed"})

    df_asr_all = snr_df[asr_mask].copy().reset_index(drop=True)
    df_non_asr = snr_df[~asr_mask].copy().reset_index(drop=True)

    if limit is not None:
        df_asr = df_asr_all.head(limit).copy().reset_index(drop=True)
        df_skipped = df_asr_all.iloc[limit:].copy().reset_index(drop=True)
    else:
        df_asr = df_asr_all
        df_skipped = pd.DataFrame(columns=snr_df.columns)

    audio_list = []
    for row in df_asr.itertuples():
        x, _ = read_wav(_resolve_segment_wav(vdir, row.audio_file), target_sr=SR)
        audio_list.append(x)
    rms = np.array([float(np.sqrt((a ** 2).mean())) for a in audio_list], dtype=float)

    if len(df_asr) > 0:
        durs = np.array([len(a) / SR for a in audio_list], dtype=float)
        if durs.max() > 30.5:
            raise ValueError(f"Segment exceeds 30.5s (max={durs.max():.2f}s); ASR models support <= 30s.")

        # Model A: Whisper large-v3
        w_ckpt = vdir / "asr_whisper_ckpt.csv"
        loaded_w = False
        if not force and w_ckpt.exists():
            c_df = pd.read_csv(w_ckpt, dtype={"segment_id": str})
            if len(c_df) == len(df_asr) and [str(x) for x in c_df["segment_id"]] == [str(x) for x in df_asr["segment_id"]]:
                df_asr["model_A_transcript"] = c_df["model_A_transcript"].fillna("").tolist()
                loaded_w = True

        if not loaded_w:
            progress(0.55, f"Running Whisper large-v3 on {len(audio_list)} speech/mixed segments...")
            w_dev = "cuda" if torch.cuda.is_available() else "cpu"
            w_dtype = torch.float16 if w_dev == "cuda" else torch.float32
            w_proc = AutoProcessor.from_pretrained(whisper_id)
            w_model = WhisperForConditionalGeneration.from_pretrained(whisper_id, torch_dtype=w_dtype).to(w_dev).eval()
            whisper_txt = []
            try:
                for i in range(0, len(audio_list), whisper_batch):
                    batch = audio_list[i:i + whisper_batch]
                    feats = w_proc(batch, sampling_rate=SR, return_tensors="pt").input_features.to(w_dev, dtype=w_dtype)
                    with torch.inference_mode():
                        ids = w_model.generate(
                            feats, language=whisper_lang, task="transcribe",
                            num_beams=whisper_beams, max_new_tokens=300,
                        )
                    texts = w_proc.batch_decode(ids, skip_special_tokens=True)
                    for k, t in enumerate(texts):
                        whisper_txt.append("" if rms[i + k] < 1e-3 else t.strip())
                    frac = 0.55 + 0.22 * min(1.0, (i + len(batch)) / max(1, len(audio_list)))
                    progress(frac, f"Whisper: {min(i + len(batch), len(audio_list))}/{len(audio_list)}")
            finally:
                del w_model, w_proc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            df_asr["model_A_transcript"] = whisper_txt
            df_asr[["_row_idx", "segment_id", "label", "model_A_transcript"]].to_csv(
                w_ckpt, index=False, encoding="utf-8-sig"
            )

        # Model B: Audar-ASR-V1-Turbo
        a_ckpt = vdir / "asr_audar_ckpt.csv"
        loaded_a = False
        if not force and a_ckpt.exists():
            c_df = pd.read_csv(a_ckpt, dtype={"segment_id": str})
            if len(c_df) == len(df_asr) and [str(x) for x in c_df["segment_id"]] == [str(x) for x in df_asr["segment_id"]]:
                df_asr["model_B_transcript"] = c_df["model_B_transcript"].fillna("").tolist()
                loaded_a = True

        if not loaded_a:
            from qwen_asr import Qwen3ASRModel

            progress(0.78, f"Running Audar-ASR-V1-Turbo on {len(audio_list)} speech/mixed segments...")
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                a_dtype = torch.bfloat16 if major >= 8 else torch.float16
                a_device = "cuda:0"
            else:
                a_dtype = torch.float32
                a_device = "cpu"
            a_model = Qwen3ASRModel.from_pretrained(
                audar_id, dtype=a_dtype, device_map=a_device,
                max_inference_batch_size=audar_batch, max_new_tokens=256,
            )
            a_model.model.to(a_dtype)
            audar_txt = []
            try:
                for i in range(0, len(audio_list), audar_batch):
                    batch = [(a, SR) for a in audio_list[i:i + audar_batch]]
                    out = a_model.transcribe(audio=batch, language=[audar_lang] * len(batch))
                    audar_txt.extend(r.text.strip() for r in out)
                    frac = 0.78 + 0.16 * min(1.0, (i + len(batch)) / max(1, len(audio_list)))
                    progress(frac, f"Audar: {min(i + len(batch), len(audio_list))}/{len(audio_list)}")
            finally:
                del a_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            df_asr["model_B_transcript"] = [("" if r < 1e-3 else t) for t, r in zip(audar_txt, rms)]
            df_asr[["_row_idx", "segment_id", "label", "model_B_transcript"]].to_csv(
                a_ckpt, index=False, encoding="utf-8-sig"
            )

        # Agreement rule
        progress(0.95, "Comparing transcripts and merging results...")
        results = [
            compare_transcripts(a, b, accept_cer=accept_cer, max_word_diff=max_word_diff)
            for a, b in zip(df_asr["model_A_transcript"], df_asr["model_B_transcript"])
        ]
        statuses = [r[0] for r in results]
        df_asr["automatic_status"] = statuses
        df_asr["status"] = statuses
        df_asr["review_status"] = ["accepted" if s == "accepted" else "pending" for s in statuses]
        df_asr["final_transcript"] = [a if s == "accepted" else "" for a, s in zip(df_asr["model_A_transcript"], statuses)]
        df_asr["cer"] = [r[1] for r in results]
        df_asr["n_diff_words"] = [r[2] for r in results]
        df_asr["second_model"] = "audar"
        df_asr["changed_by"] = ""
        df_asr["changed_at"] = ""
        df_asr["change_log"] = "[]"

    for extra_df, st, rev_st in ((df_non_asr, "no_asr", "skipped"), (df_skipped, "not_run", "pending")):
        if len(extra_df):
            for col, val in BLANK_META.items():
                extra_df[col] = val
            extra_df["status"] = st
            extra_df["automatic_status"] = st
            extra_df["review_status"] = rev_st

    parts = [p for p in (df_asr, df_non_asr, df_skipped) if len(p) > 0]
    merged_df = pd.concat(parts, ignore_index=True).sort_values("_row_idx").reset_index(drop=True)

    df_out = merged_df[[c for c in APP_COLS if c in merged_df.columns]].copy()
    df_out.to_csv(vdir / "segments.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(vdir / "segments_asr.csv", index=False, encoding="utf-8-sig")

    full_df = merged_df.copy()
    s1_path = vdir / "segments_student1.csv"
    diag_path = vdir / "snr_diagnostics.csv"
    if s1_path.exists():
        s1_df = pd.read_csv(s1_path, dtype={"segment_id": str}).reset_index(drop=True)
        if len(s1_df) == len(full_df):
            if "status" in s1_df.columns:
                full_df["qc_status"] = s1_df["status"].values
            for c in [c for c in s1_df.columns if c not in full_df.columns]:
                full_df[c] = s1_df[c].values
    if diag_path.exists():
        d_df = pd.read_csv(diag_path, dtype={"segment_id": str}).reset_index(drop=True)
        if len(d_df) == len(full_df):
            for c in ["snr_energy_db", "snr_wada_db", "snr_disagree"]:
                if c in d_df.columns:
                    full_df[c] = d_df[c].values
    full_df.drop(columns=["_row_idx"], errors="ignore").to_csv(
        vdir / "segments_full.csv", index=False, encoding="utf-8-sig"
    )

    txt_dir = vdir / "transcripts"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for r in df_asr.itertuples():
        txt_path = txt_dir / f"segment_{r.segment_id}.txt"
        if r.status == "accepted":
            txt_path.write_text(str(r.final_transcript), encoding="utf-8")
        else:
            txt_path.write_text(
                f"[{r.status}]\nModel A (Whisper): {r.model_A_transcript}\nModel B (Audar): {r.model_B_transcript}\n",
                encoding="utf-8",
            )
    return df_out
