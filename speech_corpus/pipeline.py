import json
from pathlib import Path
from typing import Literal
import pandas as pd

from .audio import DEFAULT_SR, extract_audio, extract_audio_hq, read_wav, wav_info, write_wav
from .downloader import download_video, process_local_file
from .labelling import build_mock_audio, classify_audio
from .segmentation import cut_speech_and_mixed_segments
from .video_cutter import cut_video_segments_ffmpeg
from .visualization import plot_segments_overview, plot_spectrogram_regions


def print_segments_table(segments_df: pd.DataFrame, max_rows: int = 20):
    sub = segments_df.head(max_rows)
    print("\n" + "-" * 75)
    print(f"  SEGMENTS SUMMARY (First {len(sub)} of {len(segments_df)} total)")
    print("-" * 75)
    header = f"{'ID':<6} | {'Start (s)':<10} | {'End (s)':<10} | {'Dur (s)':<8} | {'Label':<8} | {'Conf':<6} | {'File'}"
    print(header)
    print("-" * 75)
    for row in sub.itertuples():
        print(f"{row.segment_id:<6} | {row.start:<10.2f} | {row.end:<10.2f} | {row.duration:<8.2f} | {row.source_region_label:<8} | {row.confidence:<6.2f} | {row.wav_file}")
    print("-" * 75)


def run_student1_pipeline(source: str,
                          data_dir: str | Path = "data",
                          is_url: bool = True,
                          backend: Literal["heuristic", "panns"] = "heuristic",
                          vad_backend: Literal["auto", "silero", "energy"] = "auto",
                          max_height: int = 480,
                          cut_video: bool = False,
                          save_spectrogram: bool = True) -> dict:
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
        raw_video_path.replace(final_video_path)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
    else:
        file_path = Path(source)
        video_dir = data_dir / file_path.stem
        video_dir.mkdir(parents=True, exist_ok=True)
        final_video_path, meta = process_local_file(file_path, video_dir)

    # Step 2: Audio extraction
    print("\n[Step 2/4] Extracting 16 kHz mono 16-bit WAV...")
    wav_path = video_dir / "audio.wav"
    flac_path = video_dir / "audio_hq.flac"

    extract_audio(final_video_path, wav_path)
    extract_audio_hq(final_video_path, flac_path)
    audio_meta = wav_info(wav_path)

    record = dict(meta)
    record.update({
        "video_file": final_video_path.name,
        "audio_file": wav_path.name,
        "audio_hq_file": flac_path.name,
        "audio_info": audio_meta,
        "dir": str(video_dir),
    })

    with open(video_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"  Duration: {audio_meta['duration_sec'] / 60.0:.2f} min | Channel: {meta['channel']}")

    # Step 3: Classification
    print("\n[Step 3/4] Region classification (speech, music, mixed, silence)...")
    audio_samples, sr = read_wav(wav_path)
    regions_df, probs = classify_audio(audio_samples, sr=sr, backend=backend)

    regions_csv = video_dir / "regions.csv"
    regions_df.to_csv(regions_csv, index=False)
    print(f"  Regions identified: {len(regions_df)}")

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
    }


def run_mock_pipeline(out_dir: str | Path = "data/mock_demo") -> dict:
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
    print("Running Step 3 classification...")
    regions_df, probs = classify_audio(y_mock, sr=DEFAULT_SR, backend="heuristic")
    regions_df.to_csv(out_dir / "regions.csv", index=False)

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
    }
