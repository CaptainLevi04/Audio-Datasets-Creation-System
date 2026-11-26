"""Mock backend: synthetic audio and fake Arabic transcripts so the GUI can be
built and tested without the real pipeline."""
from __future__ import annotations

import json
import random
import zlib
import time
import wave
from pathlib import Path

import numpy as np

from ..models import PIPELINE_STEPS, Segment, VideoInfo, fmt_time
from ..textutil import compare_transcripts
from .base import Backend, Progress

SR = 16000

SENTENCES = [
    "هذا مثال على استخدام الذكاء الاصطناعي في التعليم الحديث",
    "يعتمد التعرف على الكلام على نماذج التعلم العميق والشبكات العصبية",
    "سنتحدث اليوم عن تاريخ الثقافة المصرية وأثرها في العالم العربي",
    "تطورت التكنولوجيا الحديثة بشكل كبير خلال العقود الماضية",
    "تعد اللغة العربية من أغنى اللغات من حيث المفردات والتراكيب",
    "يقوم النظام بتحويل الصوت إلى نص مكتوب بدقة عالية",
    "الأسرة هي النواة الأولى لبناء المجتمع وتنمية القيم الإنسانية",
    "أصبحت الهواتف الذكية جزءا أساسيا من حياتنا اليومية",
    "في هذه الحلقة نستضيف مجموعة من الخبراء والمتخصصين في المجال",
    "تنتشر الحضارات القديمة على ضفاف الأنهار الكبيرة في العالم",
    "من المهم أن نحافظ على البيئة من أجل الأجيال القادمة",
    "يحتاج الطلاب إلى التدريب العملي بجانب الدراسة النظرية",
    "تختلف اللهجات العربية من بلد إلى آخر لكنها تشترك في الجذور",
    "قدمت المنظمة تقريرا مفصلا عن نتائج المشروع الأخير",
    "ازاي نقدر نستفيد من الوقت ونحقق اللي احنا عايزينه",
    "شو رأيك نبدأ بالموضوع الاول قبل ما نكمل الحديث",
]
VOCAB = sorted({w for s in SENTENCES for w in s.split()})

SAMPLE_VIDEOS = [
    ("محاضرة عن الذكاء الاصطناعي", "Jazeera Documentary", "Creative Commons", 8.0, 11),
    ("الثقافة المصرية", "Cairo Talks", "Creative Commons", 6.5, 22),
    ("التكنولوجيا الحديثة", "Tech Arabia", "Public Domain", 7.25, 33),
]
STEP_TIMES = ["00:00:00"] * 8


def _formant_gain(freq: np.ndarray) -> np.ndarray:
    g = np.zeros_like(freq)
    for f0, bw, a in ((500, 220, 1.0), (1500, 350, 0.6), (2500, 450, 0.4), (3500, 500, 0.2)):
        g += a * np.exp(-0.5 * ((freq - f0) / bw) ** 2)
    return g + 0.02


def synth_speech(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    walk = np.cumsum(rng.normal(0, 1, n // 400 + 2))
    f0_pts = 150 + 25 * np.tanh(walk / 6)
    f0 = np.interp(np.arange(n), np.arange(len(f0_pts)) * 400, f0_pts)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    sig = np.zeros(n)
    for k in range(1, 22):
        sig += _formant_gain(k * f0) * np.sin(k * phase) / np.sqrt(k)
    syl_rate = rng.uniform(3.5, 5.5)
    env = np.clip(np.sin(2 * np.pi * syl_rate * t + rng.uniform(0, 6)) + 0.35, 0, 1) ** 0.7
    # short pauses
    pause = np.ones(n)
    for _ in range(max(1, n // (SR * 3))):
        c = rng.integers(0, n)
        w = int(rng.uniform(0.12, 0.3) * SR)
        pause[max(0, c - w // 2) : c + w // 2] = 0.03
    fric = np.diff(rng.normal(0, 1, n + 1)) * (rng.random(n) > 0.9995).cumsum() % 1
    return sig * env * pause * 0.5 + fric * 0.01


def synth_music(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    roots = [220.0, 261.63, 196.0, 174.61]
    out = np.zeros(n)
    seg_len = int(2.0 * SR)
    for i, start in enumerate(range(0, n, seg_len)):
        root = roots[(i + rng.integers(0, 2)) % len(roots)]
        end = min(n, start + seg_len)
        tt = t[start:end]
        chord = np.zeros(end - start)
        for ratio in (1.0, 1.25, 1.5, 2.0):
            for h in range(1, 7):
                chord += np.sin(2 * np.pi * root * ratio * h * tt) / h**1.3
        out[start:end] = chord * (0.6 + 0.4 * np.exp(-3 * (tt - tt[0])))
    beat = (np.sin(2 * np.pi * 60 * t) * np.exp(-12 * (t % 0.5))) * 1.2
    return (out * 0.12 + beat * 0.25)


def synth_segment(label: str, dur: float, seed: int, noise_db: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(int(dur * SR), 1)
    if label == "speech":
        x = synth_speech(rng, n)
    elif label == "music":
        x = synth_music(rng, n)
    elif label == "mixed":
        x = synth_speech(rng, n) * 0.9 + synth_music(rng, n) * 0.8
    else:
        x = np.zeros(n)
    noise = rng.normal(0, 1, n) * 10 ** (noise_db / 20)
    return x + noise


def _make_text(rng: random.Random, dur: float) -> str:
    target = max(4, int(dur * 2.0))
    words: list[str] = []
    while len(words) < target:
        words += rng.choice(SENTENCES).split()
    return " ".join(words[:target])


def _perturb(rng: random.Random, text: str, n_changes: int) -> str:
    words = text.split()
    for idx in rng.sample(range(len(words)), min(n_changes, len(words))):
        if rng.random() < 0.25 and len(words) > 4:
            words[idx] = ""
        else:
            words[idx] = rng.choice(VOCAB)
    return " ".join(w for w in words if w)


class MockBackend(Backend):
    def __init__(self, workspace: str | Path = "workspace", minutes: float | None = None):
        self.root = Path(workspace)
        self.root.mkdir(parents=True, exist_ok=True)
        self.minutes = minutes  # overrides every sample video's length
        self._videos: dict[str, VideoInfo] = {}
        self._audio_cache: dict[str, np.ndarray] = {}
        self._load_or_create_project()

    # project index
    def _index_path(self) -> Path:
        return self.root / "project.json"

    def _load_or_create_project(self) -> None:
        if self._index_path().exists():
            data = json.loads(self._index_path().read_text(encoding="utf-8"))
            for d in data["videos"]:
                v = VideoInfo.from_dict(d)
                self._videos[v.id] = v
            return
        for i, (title, channel, lic, minutes, seed) in enumerate(SAMPLE_VIDEOS):
            minutes = self.minutes or minutes
            vid = f"video{i + 1}"
            v = VideoInfo(
                id=vid,
                url=f"https://www.youtube.com/watch?v=mock{seed:04d}",
                title=title,
                channel=channel,
                duration=minutes * 60,
                license=lic,
            )
            self._set_all_steps(v, upto=6 if i == 0 else (3 if i == 1 else 0))
            self._videos[vid] = v
        self._save_index()

    def _save_index(self) -> None:
        data = {"videos": [v.to_dict() for v in self._videos.values()]}
        self._index_path().write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def _set_all_steps(self, v: VideoInfo, upto: int) -> None:
        t0 = 0.0
        for i, step in enumerate(PIPELINE_STEPS):
            if i < upto:
                t0 += v.duration * 0.001 + 1
                v.steps[step] = {"state": "done", "info": fmt_time(v.duration + t0)}
            elif i == upto and upto < len(PIPELINE_STEPS) - 2:
                v.steps[step] = {"state": "pending", "info": ""}
            else:
                v.steps[step] = {"state": "pending", "info": ""}

    # Backend API
    def list_videos(self) -> list[VideoInfo]:
        return list(self._videos.values())

    def download(self, url: str, progress: Progress) -> VideoInfo:
        for i in range(10):
            time.sleep(0.08)
            progress((i + 1) / 10, "Downloading...")
        n = len(self._videos) + 1
        rng = random.Random(url)
        v = VideoInfo(
            id=f"video{n}",
            url=url,
            title=f"فيديو جديد {n}",
            channel="Mock Channel",
            duration=(self.minutes or rng.choice([4.0, 5.0, 6.0])) * 60,
            license="Creative Commons",
        )
        self._set_all_steps(v, upto=1)
        self._videos[v.id] = v
        self._save_index()
        return v

    def extract_audio(self, video_id: str, sample_rate: int, channels: int, progress: Progress) -> str:
        for i in range(5):
            time.sleep(0.1)
            progress((i + 1) / 5, "Extracting audio...")
        v = self._videos[video_id]
        self._ensure_audio(video_id)
        v.steps["extract"] = {"state": "done", "info": fmt_time(v.duration + 1)}
        self._save_index()
        return self.audio_path(video_id)

    def audio_path(self, video_id: str) -> str:
        self._ensure_audio(video_id)
        return str(self.root / f"{video_id}.wav")

    def load_audio(self, video_id: str) -> tuple[np.ndarray, int]:
        self._ensure_audio(video_id)
        return self._audio_cache[video_id], SR

    def load_segments(self, video_id: str) -> list[Segment]:
        path = self.root / f"{video_id}_segments.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return [Segment.from_dict(d) for d in data["segments"]]
        return self._plan_segments(video_id)

    def load_log(self, video_id: str) -> list[dict]:
        path = self.root / f"{video_id}_segments.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")).get("log", [])
        return []

    def save_segments(self, video_id: str, segments: list[Segment], log: list[dict]) -> None:
        path = self.root / f"{video_id}_segments.json"
        data = {"segments": [s.to_dict() for s in segments], "log": log}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        v = self._videos[video_id]
        texts = [s for s in segments if s.has_text]
        if texts:
            v.progress = sum(s.status == "accepted" for s in texts) / len(texts)
        self._save_index()

    def run_pipeline(self, video_id: str, progress: Progress) -> list[Segment]:
        v = self._videos[video_id]
        steps = PIPELINE_STEPS[2:6]
        for i, step in enumerate(steps):
            v.steps[step] = {"state": "running", "info": ""}
            progress(i / len(steps), step)
            time.sleep(0.6)
            v.steps[step] = {"state": "done", "info": fmt_time(v.duration + 5 * (i + 1))}
        segs = self._plan_segments(video_id)
        progress(1.0, "done")
        self._save_index()
        return segs

    def reprocess_segment(self, video_id: str, segment: Segment) -> Segment:
        time.sleep(0.4)
        rng = random.Random(f"{video_id}{segment.id}{segment.start:.2f}{segment.end:.2f}")
        if segment.has_text:
            base = _make_text(rng, segment.duration)
            segment.asr_a = base
            segment.asr_b = base if rng.random() < 0.7 else _perturb(rng, base, 2)
            cmp = compare_transcripts(segment.asr_a, segment.asr_b)
            segment.status = cmp["status"]
            segment.final_text = segment.asr_a if cmp["agree"] else ""
            segment.decided_by = "auto"
        return segment

    # synthetic data
    def _ensure_audio(self, video_id: str) -> None:
        if video_id in self._audio_cache:
            return
        v = self._videos[video_id]
        wav_path = self.root / f"{video_id}.wav"
        n_total = int(v.duration * SR)
        audio = np.zeros(n_total, dtype=np.float32)
        plan = self._plan_segments(video_id)
        for s in plan:
            a, b = int(s.start * SR), min(int(s.end * SR), n_total)
            noise_db = -62 + (30 - (s.snr_energy or 25)) * 1.1
            chunk = synth_segment(s.label, (b - a) / SR, zlib.crc32(f"{video_id}{s.id}".encode()) & 0xFFFF, noise_db)
            audio[a:b] = chunk[: b - a]
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak * 0.85).astype(np.float32)
        self._audio_cache[video_id] = audio
        if not wav_path.exists():
            pcm = (audio * 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm.tobytes())

    def _plan_segments(self, video_id: str) -> list[Segment]:
        v = self._videos[video_id]
        rng = random.Random(video_id)
        segs: list[Segment] = []
        t = 0.0
        nid = 1

        def add(label: str, dur: float) -> None:
            nonlocal t, nid
            end = min(v.duration, t + dur)
            if end - t < 0.5:
                return
            s = Segment(id=nid, start=round(t, 3), end=round(end, 3), label=label)
            s.confidence = round(rng.uniform(0.72, 0.99), 2)
            if label in ("speech", "mixed"):
                base = rng.uniform(22, 32) if label == "speech" else rng.uniform(12, 22)
                s.snr_energy = round(base + rng.uniform(-2, 2), 1)
                s.snr_wada = round(base + rng.uniform(-3, 1), 1)
                s.quality = s.quality_auto = "clean" if s.snr_mean >= 20 else "noisy"
                text = _make_text(rng, s.duration)
                r = rng.random()
                s.asr_a = text
                if r < 0.62:
                    s.asr_b = text
                elif r < 0.87:
                    s.asr_b = _perturb(rng, text, rng.randint(1, 3))
                else:
                    s.asr_b = _perturb(rng, text, 6)
                cmp = compare_transcripts(s.asr_a, s.asr_b)
                s.status = cmp["status"]
                if cmp["agree"]:
                    s.final_text = s.asr_a
            nid += 1
            t = end
            segs.append(s)

        add("silence", rng.uniform(2, 6))
        while t < v.duration - 1:
            r = rng.random()
            if r < 0.14:
                add("music", rng.uniform(6, 22))
            elif r < 0.20:
                add("silence", rng.uniform(2, 7))
            else:
                label = "speech" if r < 0.78 else "mixed"
                for _ in range(rng.randint(1, 4)):
                    add(label, rng.uniform(5, 15))
        return segs
