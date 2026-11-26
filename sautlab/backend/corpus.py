"""Backend adapter: connects the GUI to the `speech_corpus` package.

Works on the same folder layout as runner.ipynb (one folder per video):

    <data_dir>/<video_id>/
        info.json  audio.wav  audio_conditioned.flac
        regions.csv  regions.meta.json
        segments/segment_0001.wav ...
        segments_student1.csv      raw output of the cut engine
        segments_snr.csv           + snr / quality
        snr_diagnostics.csv        energy / WADA SNR per segment
        segments.csv               the 16 review columns (what the GUI edits)
        segments_asr.csv           untouched copy of the automatic ASR output
        segments_full.csv          review columns + extra columns
        transcripts/segment_0001.txt
        changes.csv                audit trail (who / what / when)
        gui_state.json             GUI-only data (confidence, notes, ...)

Runs Steps 1-6 locally, including low-VRAM sequential Dual ASR
(NVIDIA Arabic Conformer + quantized Audar-ASR-V1-Turbo), and also supports importing
external `segments.csv` files via `import_asr`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from ..local_asr import (
    DEFAULT_AUDAR_GGUF,
    DEFAULT_CONFORMER_DECODER,
    DEFAULT_CONFORMER_MODEL,
    free_cuda_memory,
    run_local_dual_asr,
    transcribe_clip_dual_asr,
)
from ..models import PIPELINE_STEPS, Segment, VideoInfo
from ..review_csv import audio_name, read_review_csv, read_rows, write_review_csv
from .base import Backend, Progress

SR = 16000
SNR_DISAGREE_DB = 8.0


def _span(start, end) -> tuple[float, float]:
    return (round(float(start), 2), round(float(end), 2))


def _stamp(path: Path) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime))
    except OSError:
        return ""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _safe_id(text: str) -> str:
    """Folder name for a local file. Arabic-only names have no ASCII letters, so they get a short hash."""
    stem = Path(text).stem
    core = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "_", stem)).strip("_-")[:32]
    if len(re.sub(r"[^A-Za-z0-9]", "", core)) < 3:
        core = "audio_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    return core


class CorpusBackend(Backend):
    asr_names = ("Conformer", "Audar")

    def __init__(
        self,
        data_dir: str | Path = "data",
        classifier: str = "auto",  # auto | ast | heuristic
        vad: str = "auto",  # auto | silero | energy
        snr_threshold: float = 25.0,
        snr_combine: str = "mean",
        word_verification: bool = False,
        cookies_file: str | None = None,
        device: str | None = None,
        run_asr: bool | None = None,
        conformer_model: str = DEFAULT_CONFORMER_MODEL,
        conformer_decoder: str = DEFAULT_CONFORMER_DECODER,
        audar_quant: str = DEFAULT_AUDAR_GGUF,
    ):
        try:
            import speech_corpus  # noqa: F401
        except Exception as e:  # missing package or a missing dependency of it
            raise ImportError(
                "The speech_corpus package could not be imported "
                f"({type(e).__name__}: {e}).\nRun the GUI from the project root and install requirements.txt."
            ) from e
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.classifier = classifier
        self.vad = vad
        self.snr_threshold = float(snr_threshold)
        self.snr_combine = snr_combine
        self.word_verification = word_verification
        self.cookies_file = cookies_file
        self.device = device
        self.run_asr = (classifier != "heuristic") if run_asr is None else bool(run_asr)
        self.conformer_model = conformer_model
        self.conformer_decoder = conformer_decoder
        self.audar_quant = audar_quant
        self.run_ablation = False

        self._lock = threading.RLock()
        self._infos: dict[str, VideoInfo] = {}
        self._running: dict[str, set[str]] = {}
        self._persisted: dict[str, int] = {}
        self._audio: tuple[str, np.ndarray, int] | None = None
        self._flags_cache: dict[str, tuple[float, dict]] = {}
        self._dur_cache: dict[str, tuple[float, float]] = {}

    # folders and bookkeeping
    def _dir(self, video_id: str) -> Path:
        d = self.root / video_id
        if not d.is_dir():
            raise FileNotFoundError(f"No folder for video '{video_id}' in {self.root}")
        return d

    def _video_dirs(self) -> list[Path]:
        out = []
        for p in sorted(self.root.iterdir()):
            # YouTube ids may start with "_" or "-", so only hidden folders are skipped
            if p.is_dir() and not p.name.startswith(".") and ((p / "audio.wav").exists() or (p / "info.json").exists()):
                out.append(p)
        return out

    def _duration(self, vdir: Path, info: dict) -> float:
        wav = vdir / "audio.wav"
        if wav.exists():
            mt = wav.stat().st_mtime
            cached = self._dur_cache.get(str(wav))
            if cached and cached[0] == mt:
                return cached[1]
            try:
                import soundfile as sf

                d = float(sf.info(str(wav)).duration)
                self._dur_cache[str(wav)] = (mt, d)
                return d
            except Exception:
                pass
        return float(info.get("duration_s") or 0.0)

    def _csv_flags(self, path: Path) -> dict:
        """Cheap look into segments.csv: does it hold SNR, ASR, how many are accepted."""
        flags = {"snr": False, "asr": False, "text": 0, "accepted": 0}
        if not path.exists():
            return flags
        mt = path.stat().st_mtime
        hit = self._flags_cache.get(str(path))
        if hit and hit[0] == mt:
            return hit[1]
        try:
            for row in read_rows(path):
                if (row.get("snr") or row.get("snr_db") or "").strip():
                    flags["snr"] = True
                if (row.get("model_A_transcript") or "").strip():
                    flags["asr"] = True
                if (row.get("label") or "").strip().lower() in ("speech", "mixed"):
                    flags["text"] += 1
                    if (row.get("status") or "").strip().lower() == "accepted":
                        flags["accepted"] += 1
        except Exception:
            pass
        self._flags_cache[str(path)] = (mt, flags)
        return flags

    def _state(self, vdir: Path) -> dict:
        return _read_json(vdir / "gui_state.json")

    def _update_state(self, vdir: Path, **kw) -> None:
        st = self._state(vdir)
        st.update(kw)
        _write_json(vdir / "gui_state.json", st)

    def _steps(self, vid: str, vdir: Path) -> dict:
        steps = {k: {"state": "pending", "info": ""} for k in PIPELINE_STEPS}

        def done(key: str, path: Path) -> None:
            steps[key] = {"state": "done", "info": _stamp(path)}

        for p in (vdir / "info.json", vdir / "audio.wav"):
            if p.exists():
                done("download", p)
                break
        if (vdir / "audio.wav").exists():
            done("extract", vdir / "audio.wav")
        if (vdir / "regions.csv").exists():
            done("region_labels", vdir / "regions.csv")
        seg_csv = next((vdir / n for n in ("segments_student1.csv", "segments.csv") if (vdir / n).exists()), None)
        if seg_csv is not None and (vdir / "segments").is_dir():
            done("segmentation", seg_csv)
        main = vdir / "segments.csv"
        flags = self._csv_flags(main)
        if (vdir / "segments_snr.csv").exists() or flags["snr"]:
            done("snr", vdir / "segments_snr.csv" if (vdir / "segments_snr.csv").exists() else main)
        if flags["asr"]:
            done("dual_asr", main)
        st = self._state(vdir)
        if st.get("human_review_done"):
            steps["human_review"] = {"state": "done", "info": str(st.get("human_review_at", ""))[-8:]}
        if st.get("exported_at"):
            steps["csv_export"] = {"state": "done", "info": str(st["exported_at"])[-8:]}
        for step in self._running.get(vid, ()):
            steps[step] = {"state": "running", "info": ""}
        return steps

    def _refresh(self, vdir: Path) -> VideoInfo:
        vid = vdir.name
        info = _read_json(vdir / "info.json")
        with self._lock:
            v = self._infos.get(vid) or VideoInfo(id=vid, url="", title=vid, channel="", duration=0.0, license="")
            v.title = info.get("title") or vid
            v.channel = info.get("channel") or "-"
            v.license = info.get("license") or "-"
            v.url = info.get("webpage_url") or info.get("url") or ""
            v.duration = self._duration(vdir, info)
            v.audio_path = str(vdir / "audio.wav") if (vdir / "audio.wav").exists() else None
            v.steps = self._steps(vid, vdir)
            fl = self._csv_flags(vdir / "segments.csv")
            v.progress = (fl["accepted"] / fl["text"]) if fl["text"] else 0.0
            self._infos[vid] = v
            return v

    def _set_running(self, vid: str, step: str, on: bool) -> None:
        with self._lock:
            run = self._running.setdefault(vid, set())
            (run.add if on else run.discard)(step)
            if vid in self._infos:
                self._refresh(self._dir(vid))

    # Backend API
    def list_videos(self) -> list[VideoInfo]:
        dirs = self._video_dirs()
        with self._lock:
            for gone in set(self._infos) - {d.name for d in dirs}:
                self._infos.pop(gone, None)
        return [self._refresh(d) for d in dirs]

    def download(self, url: str, progress: Progress) -> VideoInfo:
        import speech_corpus as sc

        vid = sc.extract_video_id(url)
        vdir = self.root / vid
        progress(0.05, "Downloading audio with yt-dlp...")
        sc.download_video(url, vdir, cookies_file=self.cookies_file)
        progress(1.0, "Download finished")
        return self._refresh(vdir)

    def add_local_file(self, path: str, progress: Progress) -> VideoInfo:
        import speech_corpus as sc

        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {src}")
        vid = _safe_id(src.name)
        n = 2
        while (self.root / vid / "info.json").exists() and _read_json(self.root / vid / "info.json").get("url") != str(src.resolve()):
            vid = f"{_safe_id(src.name)}_{n}"
            n += 1
        vdir = self.root / vid
        progress(0.1, "Converting to 16 kHz mono WAV...")
        sc.ingest_local_file(src, vdir)
        progress(1.0, "Done")
        return self._refresh(vdir)

    def extract_audio(self, video_id: str, sample_rate: int, channels: int, progress: Progress) -> str:
        import speech_corpus as sc

        vdir = self._dir(video_id)
        wav = vdir / "audio.wav"
        if not wav.exists():
            raise FileNotFoundError(f"{wav} is missing. Add the video again.")
        if sample_rate != SR or channels != 1:
            raise ValueError("The corpus pipeline needs 16 kHz mono 16-bit WAV.")
        progress(0.3, "Writing conditioned FLAC...")
        try:
            sc.extract_audio_conditioned(wav, vdir / "audio_conditioned.flac")
            info = _read_json(vdir / "info.json")
            info["audio_info"] = sc.wav_info(wav)
            info["audio_file"] = "audio.wav"
            info["audio_conditioned_file"] = "audio_conditioned.flac"
            _write_json(vdir / "info.json", info)
        except Exception as e:  # the FLAC copy is optional, the GUI only needs audio.wav
            progress(0.9, f"FLAC skipped: {e}")
        progress(1.0, "Audio ready")
        self._refresh(vdir)
        return str(wav)

    def audio_path(self, video_id: str) -> str:
        return str(self._dir(video_id) / "audio.wav")

    def load_audio(self, video_id: str) -> tuple[np.ndarray, int]:
        import speech_corpus as sc

        with self._lock:
            if self._audio and self._audio[0] == video_id:
                return self._audio[1], self._audio[2]
        samples, sr = sc.read_wav(self.audio_path(video_id), target_sr=SR)
        with self._lock:
            self._audio = (video_id, samples, sr)
        return samples, sr

    # reading segments
    def _diagnostics(self, vdir: Path) -> dict[str, dict]:
        """segment_id -> energy / WADA SNR, from snr_diagnostics.csv or segments_full.csv."""
        for name in ("snr_diagnostics.csv", "segments_full.csv"):
            p = vdir / name
            if not p.exists():
                continue
            out = {}
            for row in read_rows(p):
                sid = (row.get("segment_id") or "").strip()
                if sid and (row.get("snr_energy_db") or row.get("snr_wada_db")):
                    out[sid.zfill(4)] = row
            if out:
                return out
        return {}

    @staticmethod
    def _num(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def load_segments(self, video_id: str) -> list[Segment]:
        vdir = self._dir(video_id)
        path = next((vdir / n for n in ("segments.csv", "segments_snr.csv", "segments_student1.csv") if (vdir / n).exists()), None)
        if path is None:
            return []
        segs = read_review_csv(path)
        rows = read_rows(path)
        ids = {_span(r.get("start") or 0, r.get("end") or 0): (r.get("segment_id") or "").strip().zfill(4) for r in rows}

        # raw cut-engine confidence of each segment
        conf: dict[tuple, float] = {}
        raw = vdir / "segments_student1.csv"
        if raw.exists():
            for r in read_rows(raw):
                c = self._num(r.get("confidence"))
                if c is not None and r.get("start") and r.get("end"):
                    conf[_span(r["start"], r["end"])] = c
        diag = self._diagnostics(vdir)
        side = {_span(d["start"], d["end"]): d for d in self._state(vdir).get("segments", []) if "start" in d and "end" in d}

        for s in segs:
            key = _span(s.start, s.end)
            if key in conf:
                s.confidence = round(conf[key], 3)
            d = diag.get(ids.get(key, ""))
            if d:
                s.snr_energy = self._num(d.get("snr_energy_db"))
                s.snr_wada = self._num(d.get("snr_wada_db"))
            sd = side.get(key)
            if sd:
                for name in ("confidence", "snr_energy", "snr_wada", "quality_auto", "decided_by", "notes", "auto_status"):
                    if sd.get(name) not in (None, ""):
                        setattr(s, name, sd[name])
            if s.quality_auto is None:
                s.quality_auto = s.quality
        self._persisted[video_id] = len(self.load_log(video_id))
        return segs

    def load_log(self, video_id: str) -> list[dict]:
        path = self._dir(video_id) / "changes.csv"
        if not path.exists():
            return []
        out = []
        for r in read_rows(path):
            if r.get("reason") == "gui":
                detail = r.get("new_value", "")
            else:
                detail = f"{r.get('field', '')}: {r.get('old_value', '')} -> {r.get('new_value', '')}"
            sid = (r.get("segment_id") or "").strip()
            out.append(
                {
                    "time": (r.get("timestamp") or "")[-8:],
                    "user": r.get("user", ""),
                    "segment": int(sid) if sid.isdigit() and int(sid) > 0 else None,
                    "action": r.get("field", ""),
                    "detail": detail,
                    "kind": "edit",
                }
            )
        return out

    # saving
    def _sync_wavs(self, vdir: Path, segs: list[Segment], audio: np.ndarray, sr: int) -> None:
        """Make segments/*.wav match the segment list (only touched clips are rewritten)."""
        from speech_corpus.audio import write_wav
        from speech_corpus.cut_engine_v2 import apply_hann_fade

        spans = [[round(s.start, 3), round(s.end, 3)] for s in segs]
        seg_dir = vdir / "segments"
        state = self._state(vdir)
        old_spans = state.get("wav_spans")
        if old_spans == spans and all((vdir / audio_name(i)).exists() for i in range(1, len(segs) + 1)):
            return
        reusable: dict[tuple, Path] = {}
        if old_spans:
            for i, sp in enumerate(old_spans, 1):
                p = vdir / audio_name(i)
                if p.exists():
                    reusable[(sp[0], sp[1])] = p
        staging = vdir / "segments_staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for i, s in enumerate(segs, 1):
            dst = staging / Path(audio_name(i)).name
            src = reusable.get((round(s.start, 3), round(s.end, 3)))
            if src is not None:
                shutil.copy2(src, dst)
                continue
            a, b = max(0, int(round(s.start * sr))), min(len(audio), int(round(s.end * sr)))
            write_wav(dst, apply_hann_fade(audio[a:b].copy(), sr=sr), sample_rate=sr)
        if seg_dir.exists():
            old = vdir / "_segments_old_gui"
            shutil.rmtree(old, ignore_errors=True)
            seg_dir.rename(old)
            staging.rename(seg_dir)
            shutil.rmtree(old, ignore_errors=True)
        else:
            staging.rename(seg_dir)
        self._update_state(vdir, wav_spans=spans)

    def _write_outputs(self, vdir: Path, vid: str, segs: list[Segment]) -> None:
        main = vdir / "segments.csv"
        flags = self._csv_flags(main)
        # keep the Kaggle output untouched the first time the GUI replaces it
        if main.exists() and flags["asr"] and not (vdir / "segments_asr.csv").exists():
            shutil.copy2(main, vdir / "segments_asr.csv")
        if main.exists() and not (vdir / "segments_snr.csv").exists() and flags["snr"] and not flags["asr"]:
            shutil.copy2(main, vdir / "segments_snr.csv")

        write_review_csv(main, segs)
        extras = {
            "video_id": lambda s: vid,
            "duration": lambda s: f"{s.duration:.3f}",
            "confidence": lambda s: s.confidence,
            "snr_energy_db": lambda s: s.snr_energy,
            "snr_wada_db": lambda s: s.snr_wada,
            "snr_disagree": lambda s: (
                bool(s.snr_energy is not None and s.snr_wada is not None and abs(s.snr_energy - s.snr_wada) > SNR_DISAGREE_DB)
            ),
            "notes": lambda s: s.notes,
            "decided_by": lambda s: s.decided_by,
            "second_model": lambda s: "audar_q4",
        }
        write_review_csv(vdir / "segments_full.csv", segs, extras)

        # snr_diagnostics.csv in the runner's format
        ordered = sorted(segs, key=lambda s: s.start)
        with open(vdir / "snr_diagnostics.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["segment_id", "label", "snr_energy_db", "snr_wada_db", "snr", "quality", "snr_disagree"])
            for i, s in enumerate(ordered, 1):
                dis = s.snr_energy is not None and s.snr_wada is not None and abs(s.snr_energy - s.snr_wada) > SNR_DISAGREE_DB
                w.writerow([f"{i:04d}", s.label, s.snr_energy, s.snr_wada, s.snr_mean, s.quality or "", dis])

        # transcripts, same convention as the runner
        tdir = vdir / "transcripts"
        shutil.rmtree(tdir, ignore_errors=True)
        tdir.mkdir()
        for i, s in enumerate(ordered, 1):
            if s.label not in ("speech", "mixed"):
                continue
            p = tdir / f"segment_{i:04d}.txt"
            if s.status == "accepted":
                p.write_text(s.final_text, encoding="utf-8")
            else:
                st = "difficult" if s.status == "difficult" else "human_choice"
                p.write_text(
                    f"[{st}]\nModel A ({self.asr_names[0]}): {s.asr_a}\nModel B ({self.asr_names[1]}): {s.asr_b}\n",
                    encoding="utf-8",
                )

    def save_segments(self, video_id: str, segments: list[Segment], log: list[dict]) -> None:
        vdir = self._dir(video_id)
        segs = sorted(segments, key=lambda s: s.start)
        audio, sr = self.load_audio(video_id)
        self._sync_wavs(vdir, segs, audio, sr)
        self._write_outputs(vdir, video_id, segs)

        from speech_corpus.segmentation import log_human_change

        done = self._persisted.get(video_id, 0)
        for e in log[done:]:
            log_human_change(vdir, e.get("user", ""), e.get("segment") or "", e.get("action", ""), "", e.get("detail", ""), "gui")
        self._persisted[video_id] = len(log)

        review_done = bool(segs) and all(s.status == "accepted" for s in segs)
        prev = self._state(vdir)
        state = {
            "segments": [
                {
                    "start": s.start, "end": s.end, "confidence": s.confidence,
                    "snr_energy": s.snr_energy, "snr_wada": s.snr_wada, "quality_auto": s.quality_auto,
                    "decided_by": s.decided_by, "notes": s.notes, "auto_status": s.auto_status,
                }
                for s in segs
            ],
            "wav_spans": prev.get("wav_spans"),
            "human_review_done": review_done,
            "human_review_at": time.strftime("%Y-%m-%d %H:%M:%S") if review_done else "",
            "exported_at": prev.get("exported_at", ""),
        }
        _write_json(vdir / "gui_state.json", state)
        self._refresh(vdir)

    def mark_exported(self, video_id: str) -> None:
        vdir = self._dir(video_id)
        self._update_state(vdir, exported_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._refresh(vdir)

    # pipeline
    def _classifier_kind(self) -> str:
        if self.classifier == "heuristic":
            return "heuristic"
        have_ast = True
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            have_ast = False
        if self.classifier == "ast" and not have_ast:
            raise RuntimeError("The AST classifier needs torch and transformers (pip install torch transformers).")
        return "ast" if have_ast else "heuristic"

    def run_pipeline(self, video_id: str, progress: Progress) -> list[Segment]:
        import pandas as pd
        import speech_corpus as sc
        from speech_corpus.labelling import classify_audio as classify_any
        from speech_corpus.snr import compute_snr_for_segments, estimate_segment_snr

        vdir = self._dir(video_id)
        audio, sr = self.load_audio(video_id)
        dur = len(audio) / sr
        steps = ("region_labels", "segmentation", "snr", "dual_asr")
        try:
            # step 3: regions
            self._set_running(video_id, "region_labels", True)
            progress(0.05, "Detecting pauses...")
            _, pauses_df = sc.detect_speech_activity_overlay(
                audio=audio, sr=sr, silence_threshold_s=0.25, min_pause_ms=120, vad_backend=self.vad
            )
            regions_csv = vdir / "regions.csv"
            if sc.validate_cached_regions(regions_csv, expected_source_id=vdir.name, expected_duration=dur, expected_version="2.2"):
                regions_df = pd.read_csv(regions_csv)
            else:
                kind = self._classifier_kind()
                note = ""
                if kind == "heuristic" and self.classifier != "heuristic":
                    note = " - torch/transformers not found, using the less accurate heuristic"
                progress(0.10, f"Classifying regions ({kind}){note}...")
                if kind == "ast":
                    regions_df, windows_df = sc.classify_audio(
                        y=audio, sr=sr, win_s=10.0, hop_s=5.0, batch_size=32, device=self.device, pauses_df=pauses_df
                    )
                else:
                    regions_df, windows_df = classify_any(audio, sr=sr, backend="heuristic")
                if regions_df is None or len(regions_df) == 0:
                    raise RuntimeError("Region classification returned no regions.")
                sc.save_regions(regions_df, regions_csv, source_id=vdir.name, duration=dur, version="2.2")
                if isinstance(windows_df, pd.DataFrame) and not windows_df.empty:
                    windows_df.to_csv(vdir / "windows_ast.csv", index=False)
            free_cuda_memory()
            self._set_running(video_id, "region_labels", False)

            # step 4: segmentation
            self._set_running(video_id, "segmentation", True)
            progress(0.25, "Cutting segments at pauses...")
            segments_df = sc.cut_speech_and_mixed_segments(
                audio=audio, regions_df=regions_df, out_dir=vdir, sr=sr, pauses=pauses_df,
                use_word_verification=self.word_verification, run_ablation=self.run_ablation,
            )
            seg_dir, staging = vdir / "segments", vdir / "segments_staging"
            if not seg_dir.exists() and staging.exists():  # QC failed: keep the cuts like the runner does
                staging.rename(seg_dir)
                for col in ("audio_file", "wav_file"):
                    if col in segments_df.columns:
                        segments_df[col] = segments_df[col].map(
                            lambda x: x.replace("segments_staging/", "segments/") if isinstance(x, str) else x
                        )
            elif not seg_dir.exists():
                raise RuntimeError("The cut engine produced no segments.")
            segments_df.to_csv(vdir / "segments_student1.csv", index=False)
            try:
                qc = sc.validate_segments_integrity(segments_df, video_dir=vdir, total_dur=dur, sr=sr)
                sc.generate_qc_markdown_report(segments_df, report_path=vdir / "qc_report.md", video_dir=vdir, qc=qc)
            except Exception as e:  # the report is a bonus, never stop the pipeline for it
                progress(0.35, f"QC report skipped: {e}")
            free_cuda_memory()
            self._set_running(video_id, "segmentation", False)

            # step 5: SNR
            self._set_running(video_id, "snr", True)
            progress(0.42, "Estimating SNR...")
            src = pd.read_csv(vdir / "segments_student1.csv", dtype={"segment_id": str})
            snr_df, _ = compute_snr_for_segments(
                src, base_dir=vdir, threshold_db=self.snr_threshold, combine=self.snr_combine, save_csv=True, verbose=False
            )
            snr_df.to_csv(vdir / "segments_snr.csv", index=False, encoding="utf-8-sig")
            diag = []
            for row in src.itertuples():
                rel = row.wav_file if isinstance(getattr(row, "wav_file", None), str) else row.audio_file
                clip, csr = sc.read_wav(vdir / str(rel), target_sr=sr)
                est = estimate_segment_snr(clip, csr, self.snr_threshold, self.snr_combine)
                diag.append(
                    {
                        "segment_id": row.segment_id, "label": row.label,
                        "snr_energy_db": est["snr_energy_db"], "snr_wada_db": est["snr_wada_db"],
                        "snr": est["snr"], "quality": est["quality"], "snr_disagree": est["snr_disagree"],
                    }
                )
            pd.DataFrame(diag).to_csv(vdir / "snr_diagnostics.csv", index=False, encoding="utf-8-sig")
            self._set_running(video_id, "snr", False)

            # Fresh cut replaces previous transcripts and checkpoints.
            shutil.rmtree(vdir / "transcripts", ignore_errors=True)
            (vdir / "gui_state.json").unlink(missing_ok=True)
            (vdir / "segments_asr.csv").unlink(missing_ok=True)
            (vdir / "asr_conformer_ckpt.csv").unlink(missing_ok=True)
            (vdir / "asr_audar_ckpt.csv").unlink(missing_ok=True)
            (vdir / "segments.csv").unlink(missing_ok=True)

            # step 6: Local Dual ASR (Conformer + Quantized Audar) sequentially in <= 4 GB VRAM
            if self.run_asr:
                self._set_running(video_id, "dual_asr", True)
                run_local_dual_asr(
                    vdir,
                    conformer_model=self.conformer_model,
                    conformer_decoder=self.conformer_decoder,
                    audar_quant=self.audar_quant,
                    device=self.device,
                    force=True,
                    progress=lambda f, m="": progress(0.50 + 0.48 * f, m),
                )
                self._set_running(video_id, "dual_asr", False)
        finally:
            for st in steps:
                self._set_running(video_id, st, False)

        self._flags_cache.clear()
        progress(1.0, "Pipeline finished")
        segs = self.load_segments(video_id)
        self._update_state(vdir, wav_spans=[[s.start, s.end] for s in segs])
        self._refresh(vdir)
        return segs

    def reprocess_segment(self, video_id: str, segment: Segment) -> Segment:
        """Re-estimate SNR (and Dual ASR when enabled) for the current cut."""
        from speech_corpus.snr import estimate_segment_snr

        audio, sr = self.load_audio(video_id)
        a, b = int(round(segment.start * sr)), int(round(segment.end * sr))
        clip = audio[a:b]
        est = estimate_segment_snr(clip, sr, self.snr_threshold, self.snr_combine)
        segment.snr = est["snr"]
        segment.snr_energy = est["snr_energy_db"]
        segment.snr_wada = est["snr_wada_db"]
        segment.quality = segment.quality_auto = est["quality"]
        if segment.has_text and self.run_asr:
            a_txt, b_txt, scored = transcribe_clip_dual_asr(
                clip,
                sr=sr,
                conformer_model=self.conformer_model,
                conformer_decoder=self.conformer_decoder,
                audar_quant=self.audar_quant,
                device=self.device,
            )
            segment.asr_a = a_txt
            segment.asr_b = b_txt
            segment.auto_status = scored["csv_status"]
            segment.status = scored["gui_status"]
            segment.final_text = a_txt if segment.status == "accepted" else ""
            segment.decided_by = "auto" if segment.status == "accepted" else "human"
        elif segment.has_text and (segment.asr_a or segment.asr_b):
            segment.status = "needs_review"
        return segment

    def import_asr(self, video_id: str, csv_path: str) -> list[Segment]:
        vdir = self._dir(video_id)
        src = Path(csv_path)
        rows = read_rows(src)
        if not rows or "model_A_transcript" not in rows[0]:
            raise ValueError("This CSV has no model_A_transcript column. Use the segments.csv written by the Kaggle runner.")
        ends = [float(r["end"]) for r in rows if (r.get("end") or "").strip()]
        info = _read_json(vdir / "info.json")
        total = self._duration(vdir, info)
        if ends and total and max(ends) > total + 1.0:
            raise ValueError(
                f"The CSV runs to {max(ends):.1f} s but this video is only {total:.1f} s long. Is it the right video?"
            )
        main = vdir / "segments.csv"
        if main.exists():
            shutil.copy2(main, vdir / f"segments_before_asr_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy2(src, main)
        shutil.copy2(src, vdir / "segments_asr.csv")
        self._flags_cache.clear()
        (vdir / "gui_state.json").unlink(missing_ok=True)  # spans may differ, rebuild on next save
        segs = self.load_segments(video_id)
        self._refresh(vdir)
        return segs
