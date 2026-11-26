from __future__ import annotations

from dataclasses import dataclass, field, asdict

LABELS = ("speech", "music", "mixed", "silence")
LABEL_TITLES = {"speech": "Speech", "music": "Music", "mixed": "Mixed", "silence": "Silence"}
TEXT_LABELS = ("speech", "mixed")

STATUSES = ("accepted", "needs_review", "difficult")
STATUS_TITLES = {
    "accepted": "Accepted",
    "needs_review": "Needs Review",
    "difficult": "Difficult",
}

PIPELINE_STEPS = (
    "download",
    "extract",
    "region_labels",
    "segmentation",
    "snr",
    "dual_asr",
    "human_review",
    "csv_export",
)
STEP_TITLES = {
    "download": "Download",
    "extract": "Extract",
    "region_labels": "Region Labels",
    "segmentation": "Segmentation",
    "snr": "SNR",
    "dual_asr": "Dual ASR",
    "human_review": "Human Review",
    "csv_export": "CSV Export",
}


@dataclass
class Segment:
    id: int
    start: float
    end: float
    label: str = "speech"
    confidence: float = 1.0
    snr: float | None = None  # combined value used for clean/noisy
    snr_energy: float | None = None
    snr_wada: float | None = None
    quality: str | None = None  # "clean" / "noisy" / None
    quality_auto: str | None = None  # value computed by the backend
    asr_a: str = ""  # model A transcript (Whisper)
    asr_b: str = ""  # model B transcript (Audar)
    final_text: str = ""
    status: str = "accepted"
    auto_status: str = ""  # status given by the automatic ASR comparison
    decided_by: str = "auto"  # auto / a / b / edited / human
    notes: str = ""
    changed_by: str = ""
    changed_at: str = ""
    change_log: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def has_text(self) -> bool:
        return self.label in TEXT_LABELS

    @property
    def snr_mean(self) -> float | None:
        if self.snr is not None:
            return self.snr
        vals = [v for v in (self.snr_energy, self.snr_wada) if v is not None]
        return sum(vals) / len(vals) if vals else None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        d = dict(d)
        for old, new in (("whisper", "asr_a"), ("mms", "asr_b")):  # older saved files
            if old in d and new not in d:
                d[new] = d.pop(old)
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class VideoInfo:
    id: str
    url: str
    title: str
    channel: str
    duration: float
    license: str
    thumbnail: str | None = None
    audio_path: str | None = None
    steps: dict = field(default_factory=dict)  # step -> {"state": str, "info": str}
    progress: float = 0.0  # share of segments already reviewed, 0..1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoInfo":
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in names})


def fmt_time(t: float, ms: bool = False) -> str:
    t = max(0.0, float(t))
    total_ms = int(round(t * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}.{milli:03d}" if ms else base


def parse_time(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        parts = text.split(":")
        secs = 0.0
        for p in parts:
            secs = secs * 60 + float(p)
        return secs
    except ValueError:
        return None
