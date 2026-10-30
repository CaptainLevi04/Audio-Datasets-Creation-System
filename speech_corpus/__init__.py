"""Speech Corpus Toolkit - Audio Pipeline.
"""

from .audio import (
    DEFAULT_SR,
    check_ffmpeg,
    extract_audio,
    extract_audio_hq,
    read_wav,
    slice_wav,
    wav_info,
    write_wav,
)
from .downloader import download_video, is_creative_commons, process_local_file
from .labelling import (
    COLORS,
    LABELS,
    build_mock_audio,
    classify_audio,
    evaluate_ground_truth,
    heuristic_probs,
    panns_probs,
    probs_to_regions,
)
from .pipeline import run_mock_pipeline, run_student1_pipeline
from .segmentation import (
    cut_speech_and_mixed_segments,
    detect_pauses_energy,
    detect_pauses_silero,
    slice_region_into_chunks,
)
from .video_cutter import (
    cut_video_from_timestamp_file,
    cut_video_segments_ffmpeg,
    cut_video_segments_moviepy,
)
from .visualization import RegionEditor, plot_segments_overview, plot_spectrogram_regions

__version__ = "1.0.0"

__all__ = [
    "download_video",
    "is_creative_commons",
    "process_local_file",
    "extract_audio",
    "extract_audio_hq",
    "read_wav",
    "write_wav",
    "slice_wav",
    "wav_info",
    "DEFAULT_SR",
    "check_ffmpeg",
    "classify_audio",
    "heuristic_probs",
    "panns_probs",
    "probs_to_regions",
    "build_mock_audio",
    "evaluate_ground_truth",
    "LABELS",
    "COLORS",
    "plot_spectrogram_regions",
    "plot_segments_overview",
    "RegionEditor",
    "detect_pauses_energy",
    "detect_pauses_silero",
    "slice_region_into_chunks",
    "cut_speech_and_mixed_segments",
    "cut_video_segments_ffmpeg",
    "cut_video_segments_moviepy",
    "cut_video_from_timestamp_file",
    "run_student1_pipeline",
    "run_mock_pipeline",
]
