import json
from pathlib import Path
from typing import Literal
import pandas as pd

from .audio import (
    DEFAULT_SR, extract_audio, extract_audio_hq, extract_audio_conditioned,
    extract_source_archive, read_wav, wav_info, write_wav
)
from .downloader import download_video, process_local_file
from .labelling import build_mock_audio, classify_audio
from .cut_engine_v2 import cut_speech_and_mixed_segments
from .video_cutter import cut_video_segments_ffmpeg
from .visualization import plot_segments_overview, plot_spectrogram_regions


def print_segments_table(segments_df: pd.DataFrame, max_rows: int = 20):
    sub = segments_df.head(max_rows)
    print("\n" + "-" * 105)
    print(f"  SEGMENTS SUMMARY (First {len(sub)} of {len(segments_df)} total)")
    print("-" * 105)
    header = f"{'ID':<6} | {'Start':<7} | {'End':<7} | {'Dur':<6} | {'Label':<8} | {'Tier':<6} | {'Cert':<5} | {'Priority':<8} | {'File'}"
    print(header)
    print("-" * 105)
    for row in sub.itertuples():
        lbl = getattr(row, "primary_label", getattr(row, "label", "speech"))
        tier = getattr(row, "cut_tier", "HIGH")
        cert = getattr(row, "cut_certainty", getattr(row, "certainty", 1.0))
        prio = getattr(row, "review_priority", "LOW")
        print(f"{row.segment_id:<6} | {row.start:<7.2f} | {row.end:<7.2f} | {row.duration:<6.2f} | {lbl:<8} | {tier:<6} | {cert:<5.2f} | {prio:<8} | {row.wav_file}")
    print("-" * 105)


def run_student1_pipeline(source: str,
                          data_dir: str | Path = "data",
                          is_url: bool = True,
                          backend: Literal["ast", "heuristic", "panns"] = "ast",
                          vad_backend: Literal["auto", "silero", "energy"] = "auto",
                          max_height: int = 360,
                          cut_video: bool = False,
                          save_spectrogram: bool = True,
                          device: str | None = None) -> dict:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPEECH CORPUS PIPELINE (STEPS 1-4)")
    print("=" * 70)

    # Step 1: Ingest
    print("\n[Step 1/4] Ingesting video...")
    if is_url:
        tmp_dir = data_dir / "_incoming"
        raw_video_path, meta = download_video(source, tmp_dir, max_height=max_height)
        video_dir = data_dir / meta["video_id"]
        video_dir.mkdir(parents=True, exist_ok=True)
        final_video_path = video_dir / raw_video_path.name
        if raw_video_path.resolve() != final_video_path.resolve() and raw_video_path.exists():
            raw_video_path.replace(final_video_path)
        try:
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
        except OSError:
            pass
    else:
        file_path = Path(source)
        video_dir = data_dir / file_path.stem
        video_dir.mkdir(parents=True, exist_ok=True)
        final_video_path, meta = process_local_file(file_path, video_dir)

    # Step 2: Audio extraction
    print("\n[Step 2/4] Extracting 16 kHz mono 16-bit WAV & Conditioned FLAC...")
    wav_path = video_dir / "audio.wav"
    flac_path = video_dir / "audio_conditioned.flac"

    extract_audio(final_video_path, wav_path)
    extract_audio_conditioned(final_video_path, flac_path)
    audio_meta = wav_info(wav_path)

    record = dict(meta)
    record.update({
        "video_file": final_video_path.name,
        "audio_file": wav_path.name,
        "audio_conditioned_file": flac_path.name,
        "audio_info": audio_meta,
        "dir": str(video_dir),
    })

    with open(video_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"  Duration: {audio_meta['duration_sec'] / 60.0:.2f} min | Channel: {meta['channel']}")

    # Step 3: Classification
    print(f"\n[Step 3/4] Region classification (speech, music, mixed, silence) via [{backend.upper()}]...")
    audio_samples, sr = read_wav(wav_path)
    total_dur = len(audio_samples) / sr
    regions_csv = video_dir / "regions.csv"

    from .ast_model import validate_cached_regions, save_regions
    if validate_cached_regions(regions_csv, expected_source_id=video_dir.name, expected_duration=total_dur, expected_version="2.2"):
        print(f"  Valid cached regions found (v2.2): {regions_csv}")
        regions_df = pd.read_csv(regions_csv)
        raw_preds = None
    else:
        regions_df, raw_preds = classify_audio(audio_samples, sr=sr, backend=backend, device=device)
        save_regions(regions_df, regions_csv, source_id=video_dir.name, duration=total_dur, version="2.2")
        print(f"  Regions identified and saved (v2.2): {len(regions_df)}")

    # If window-level predictions DataFrame exists, save windows_5s.csv
    if isinstance(raw_preds, pd.DataFrame):
        windows_csv = video_dir / "windows_5s.csv"
        raw_preds.to_csv(windows_csv, index=False)
        print(f"  Window-level predictions (5s) saved: {windows_csv}")

    # Full audio macro-view
    if save_spectrogram:
        full_spec_path = video_dir / "full_spectrogram.png"
        plot_spectrogram_regions(
            audio_samples, regions_df, sr=sr,
            title=f"Full Audio Spectrogram - {meta['title'][:40]}",
            save_path=full_spec_path
        )

    # Step 4: Pause-aware segmentation
    print("\n[Step 4/4] Segmentation at real pauses (5-15s target)...")
    segments_df = cut_speech_and_mixed_segments(
        audio=audio_samples,
        regions_df=regions_df,
        out_dir=video_dir,
        sr=sr,
        vad_backend=vad_backend
    )

    # Generate composite visual report (First 20 segments + visual table)
    # Saved as both spectrogram.png and segments_overview.png
    if save_spectrogram:
        spec_path = video_dir / "spectrogram.png"
        overview_path = video_dir / "segments_overview.png"
        plot_segments_overview(
            audio_samples, segments_df, sr=sr, max_segments=20,
            save_path=spec_path
        )
        if spec_path.resolve() != overview_path.resolve():
            import shutil
            shutil.copy2(spec_path, overview_path)

    if cut_video and final_video_path.exists():
        print("  Generating matching video subclips...")
        cut_video_segments_ffmpeg(final_video_path, segments_df, video_dir)

    print_segments_table(segments_df, max_rows=20)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED: {len(segments_df)} segments generated in {video_dir / 'segments'}")
    print(f"Visual Report Image: {video_dir / 'spectrogram.png'}")
    print(f"Segments Index:      {video_dir / 'segments.csv'}")
    print("=" * 70)

    return {
        "status": "success",
        "video_dir": str(video_dir),
        "metadata": record,
        "num_regions": len(regions_df),
        "num_segments": len(segments_df),
        "regions_csv": str(regions_csv),
        "segments_csv": str(video_dir / "segments.csv"),
        "spectrogram_png": str(video_dir / "spectrogram.png"),
        "segments": segments_df,
    }


def run_student1(url: str,
                 profile: dict | None = None,
                 data_dir: str | Path = "data",
                 device: str | None = None) -> pd.DataFrame:
    """Standard function API for Student 3 (UI App / Streamlit / Gradio) & Live Demo.
    
    Accepts video URL and profile dict, runs the optimized Steps 1-4 pipeline,
    and returns the manifest DataFrame (saved to segments.csv).
    """
    res = run_student1_pipeline(
        source=url,
        data_dir=data_dir,
        is_url=True,
        backend="ast",
        vad_backend="auto",
        device=device
    )
    return res["segments"]


def run_mock_pipeline(out_dir: str | Path = "data/mock_demo",
                      backend: Literal["ast", "heuristic", "panns"] = "ast") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    extended_timeline = [
        ("silence", 2.0),
        ("speech", 18.0),
        ("music", 8.0),
        ("mixed", 22.0),
        ("silence", 2.5),
        ("speech", 24.0),
        ("mixed", 16.0),
        ("music", 6.0),
        ("speech", 26.0),
        ("silence", 3.0),
        ("mixed", 20.0),
        ("speech", 15.0),
    ]
    
    print("Generating synthetic test audio signal...")
    y_mock, gt_df = build_mock_audio(seed=42, timeline=extended_timeline)

    mock_wav = out_dir / "audio.wav"
    write_wav(mock_wav, y_mock, sample_rate=DEFAULT_SR)

    # Step 3
    print(f"Running Step 3 classification via [{backend.upper()}]...")
    regions_df, raw_preds = classify_audio(y_mock, sr=DEFAULT_SR, backend=backend)
    regions_df.to_csv(out_dir / "regions.csv", index=False)
    if isinstance(raw_preds, pd.DataFrame):
        raw_preds.to_csv(out_dir / "windows_5s.csv", index=False)

    full_spec = out_dir / "full_spectrogram.png"
    plot_spectrogram_regions(y_mock, regions_df, gt_df=gt_df, sr=DEFAULT_SR, save_path=full_spec)

    # Step 4
    print("Running Step 4 pause detection & 5-15s segmentation...")
    segments_df = cut_speech_and_mixed_segments(
        audio=y_mock,
        regions_df=regions_df,
        out_dir=out_dir,
        sr=DEFAULT_SR,
        vad_backend="energy"
    )

    spec_path = out_dir / "spectrogram.png"
    plot_segments_overview(
        y_mock, segments_df, sr=DEFAULT_SR, max_segments=20,
        save_path=spec_path
    )
    overview_path = out_dir / "segments_overview.png"
    import shutil
    shutil.copy2(spec_path, overview_path)

    print_segments_table(segments_df, max_rows=20)

    print("\n" + "=" * 70)
    print(f"MOCK PIPELINE FINISHED: {len(segments_df)} segments generated in {out_dir / 'segments'}")
    print(f"Visual Report Image: {spec_path}")
    print("=" * 70)

    return {
        "status": "success",
        "out_dir": str(out_dir),
        "num_regions": len(regions_df),
        "num_segments": len(segments_df),
        "segments_csv": str(out_dir / "segments.csv"),
        "spectrogram_png": str(spec_path),
        "segments": segments_df,
    }
