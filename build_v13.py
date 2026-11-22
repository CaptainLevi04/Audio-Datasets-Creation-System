import json
from pathlib import Path

def build_v13():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().splitlines()]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().splitlines()]
        })

    # =========================================================================
    # Notebook Structure: Student 1 Improvement Plan (v13: Cut Engine v2)
    # =========================================================================

    # Cell 0: Header
    add_md(r"""# 🎙️ Speech Corpus Pipeline: Student 1 (v13: Cut Engine v2 & Global DP Optimizer)
### End-to-End Implementation of the Student 1 Improvement Plan (P0–P2)
1. **Global DP Optimizer (Cut Engine v2):** Shortest-path dynamic programming over voice streams; guarantees 5.0s–15.0s priority window (18.0s fallback ceiling).
2. **Word-Timestamp Verification:** Verified boundary protection via `faster-whisper` (word_timestamps=True); 0 cuts inside words.
3. **Strict 100–200ms Margin Rule:** Leaves true silence on both edges; 8ms Hann fade strictly inside margin.
4. **Frozen `segments.csv` Schema & Review Queue:** `cut_certainty`, `cut_tier` (HIGH/MEDIUM/LOW), `word_check_passed`, `needs_review`.
5. **Interactive Inspectors with Real-Time Synced Cursor:** 60 FPS animated audio playback tracking across Waveform & Energy curves.
6. **Phase 6 Cleaned:** Circular pseudo-label fine-tuning cell removed; clean top-to-bottom run.""")

    # Cell 1: Imports & Setup
    add_code(r"""import sys
from pathlib import Path

# Add repository root to path
REPO_PATH = Path('Audio-Datasets-Creation-System')
if REPO_PATH.exists() and str(REPO_PATH.resolve()) not in sys.path:
    sys.path.insert(0, str(REPO_PATH.resolve()))

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Audio, HTML

from speech_corpus.audio import read_wav, write_wav, extract_audio, extract_audio_hq, DEFAULT_SR
from speech_corpus.downloader import download_video, find_cached_video
from speech_corpus.labelling import classify_audio, build_mock_audio
from speech_corpus.segmentation import (
    cut_speech_and_mixed_segments,
    detect_speech_activity_overlay,
    compute_instantaneous_energy_vad_overlay,
    create_second_by_second_overlay,
    log_human_change
)
from speech_corpus.cut_engine_v2 import (
    Cand,
    plan_cuts,
    cut_certainty,
    get_tier,
    apply_margin_and_fade,
    apply_hann_fade,
    pause_score,
    reject_inside_words
)
from speech_corpus.inspector import (
    launch_region_inspector,
    launch_vad_overlay_inspector,
    launch_timeline_pointer_inspector,
    launch_segment_inspector
)
from speech_corpus.quality_control import (
    validate_segments_integrity,
    split_corpus_by_source,
    generate_qc_markdown_report
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Environment Ready! • Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("✅ Cut Engine v2 Loaded: Global DP Optimizer, Word Verification & Frozen Schema.")""")

    # Cell 2: Step 1 & 2 Markdown
    add_md(r"""## 📥 Step 1 & 2: Video Download & High-Quality Audio Extraction
- **Audio Extraction:** High-quality 16 kHz Mono WAV (standardized for speech models) + lossless 24-bit FLAC archive.
- **Deduplication:** Automatic cache lookup avoids redundant downloads.""")

    # Cell 3: Step 1 & 2 Code
    add_code(r"""USE_MOCK_DEMO = False  # Set to True for an instant offline demo
YOUTUBE_URL = "https://www.youtube.com/watch?v=7fIN_zj4Yno"  # Replace with any YouTube URL
MAX_HEIGHT = 360  # Fast download (360p or 480p)
DATA_DIR = Path("data")

if USE_MOCK_DEMO:
    print("⚡ Generating synthetic test audio (speech, music, mixed, silence)...")
    video_dir = DATA_DIR / "demo_run"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_path = video_dir / "audio.wav"
    audio_samples = build_mock_audio(duration_s=60.0, sr=DEFAULT_SR)
    write_wav(audio_path, audio_samples, sample_rate=DEFAULT_SR)
    meta = {"id": "demo_run", "title": "Synthetic Audio Test", "duration": 60.0}
else:
    cached_id, cached_audio = find_cached_video(YOUTUBE_URL, data_dir=DATA_DIR)
    if cached_id and cached_audio:
        print(f"⚡ Found cached audio for video [{cached_id}]: {cached_audio}")
        video_dir = cached_audio.parent
        audio_path = cached_audio
        audio_samples, sr = read_wav(audio_path)
    else:
        print(f"📥 Downloading and extracting audio from: {YOUTUBE_URL}")
        video_path, meta = download_video(YOUTUBE_URL, out_dir=DATA_DIR, max_height=MAX_HEIGHT)
        video_dir = video_path.parent
        audio_path = video_dir / "audio.wav"
        flac_path = video_dir / "audio_hq.flac"
        extract_audio_hq(video_path, wav_out=audio_path, flac_out=flac_path, sr=DEFAULT_SR)
        audio_samples, sr = read_wav(audio_path)

dur_s = len(audio_samples) / DEFAULT_SR
print(f"🎵 Audio Ready: {dur_s:.1f}s ({dur_s/60:.2f} mins) @ {DEFAULT_SR} Hz | Saved in {video_dir}")""")

    # Cell 4: Step 3 Markdown
    add_md(r"""## 🧠 Step 3: AST Multi-Class Classification & Urgency Scoring
- Classifies audio into 4 content classes: `speech`, `music`, `mixed`, `silence`.
- Calculates margin-based Certainty and Urgency.
- **Smart Cache:** If `regions.csv` exists, loads instantly. Set `FORCE_RECLASSIFY = True` to re-run.""")

    # Cell 5: Step 3 Code
    add_code(r"""regions_csv = video_dir / "regions.csv"
windows_csv = video_dir / "windows_5s.csv"
FORCE_RECLASSIFY = False  # Set to True to re-run neural inference from scratch

if not FORCE_RECLASSIFY and regions_csv.exists() and windows_csv.exists():
    print(f"⚡ Loading cached AST classification from {regions_csv}...")
    regions_df = pd.read_csv(regions_csv)
    windows_df = pd.read_csv(windows_csv)
else:
    print("🧠 Running Audio Spectrogram Transformer (AST) classification...")
    regions_df, windows_df = classify_audio(audio_samples, sr=DEFAULT_SR, backend="ast", batch_size=32)
    regions_df.to_csv(regions_csv, index=False)
    windows_df.to_csv(windows_csv, index=False)
    print(f"Saved: {regions_csv} and {windows_csv}")

print(f"Detected {len(regions_df)} acoustic regions across {dur_s:.1f}s:")
display(regions_df)""")

    # Cell 6: Step 3 Spectrogram Markdown
    add_md(r"""## 📊 Full Audio Spectrogram Overview
Generates and displays the macro spectrogram across the full recording duration.""")

    # Cell 7: Step 3 Spectrogram Code
    add_code(r"""spec_path = video_dir / "full_spectrogram.png"
if not spec_path.exists():
    try:
        from speech_corpus.visualization import plot_audio_timeline
        print("Generating timeline visual report...")
        plot_audio_timeline(audio_samples, regions_df, out_path=spec_path, sr=DEFAULT_SR)
    except Exception as e:
        print(f"Note: Timeline visualizer: {e}")

if spec_path.exists():
    display(HTML(f'<img src="{spec_path.as_posix()}" style="max-width:100%; border:1px solid #ddd; border-radius:6px;" />'))""")

    # Cell 8: Inspector 1 Markdown
    add_md(r"""### 🔍 Interactive Inspector 1: Review & Override Step 3 Macro Regions
Audit and edit AST classification regions before cutting. Edits are logged to `changes.csv` and automatically propagate to Step 4.""")

    # Cell 9: Inspector 1 Code
    add_code(r"""# 🔍 Cell 1 of 4: Inspect & Override Step 3 Macro Regions
launch_region_inspector(
    audio_samples=audio_samples,
    regions_df=regions_df,
    video_dir=video_dir,
    sr=DEFAULT_SR
)""")

    # Cell 10: Step 3.5 Overlay Markdown
    add_md(r"""## 🎙️ Step 3.5: Continuous Energy-Assisted Silence Overlay (`SPEECH` vs `SILENCE`)
Unlike macro-region labelling (5–15s chunks), this overlay evaluates audio at **millisecond-level resolution (10ms frames)**:
- **Energy-Assisted:** Uses Short-Time RMS Log Energy $E_{\text{dB}}(t)$ to detect physical vocal cords vibrations vs acoustic silence.
- **`SILENCE` = `NO SPEECH`:** Marks exact pauses and breath gaps, separate from macroscopic background silence.""")

    # Cell 11: Step 3.5 Overlay Code
    add_code(r"""# 🎙️ Step 3.5: Continuous Energy-Assisted Speech vs Silence Overlay
SILENCE_THRESHOLD_S = 0.25  # 250ms threshold: minimum pause duration for a 100% safe cut

print("🔍 Extracting millisecond-level Voice Activity & Short-Time Energy...")
vad_timeline_df, pauses_df = detect_speech_activity_overlay(
    audio=audio_samples,
    sr=DEFAULT_SR,
    silence_threshold_s=SILENCE_THRESHOLD_S,
    min_pause_ms=120,
    vad_backend="auto"
)

# Compute continuous 10ms frame energy overlay
instant_intervals_df, times_arr, energy_db_arr, thresh_db = compute_instantaneous_energy_vad_overlay(
    audio=audio_samples,
    sr=DEFAULT_SR,
    vad_timeline_df=vad_timeline_df
)

total_dur = len(audio_samples) / DEFAULT_SR
sp_time = instant_intervals_df[instant_intervals_df["state"] == "SPEECH"]["duration"].sum() if not instant_intervals_df.empty else 0.0
sil_time = instant_intervals_df[instant_intervals_df["state"] == "SILENCE"]["duration"].sum() if not instant_intervals_df.empty else 0.0

print()
print("📊 Continuous Acoustic Overlay Summary:")
print(f"• Total Audio Duration : {total_dur/60:.1f} mins ({total_dur:.1f}s)")
print(f"• Active Speech Energy : {sp_time/60:.1f} mins ({sp_time/total_dur*100:.1f}%)")
print(f"• Acoustic Silence Gap : {sil_time/60:.1f} mins ({sil_time/total_dur*100:.1f}%)")
print(f"• Silence Energy Floor : {thresh_db:.1f} dB")
print(f"• Continuous Intervals : {len(instant_intervals_df)} exact acoustic segments")

# Save tables
instant_intervals_df.to_csv(video_dir / "instant_overlay.csv", index=False)
vad_timeline_df.to_csv(video_dir / "vad_timeline.csv", index=False)
pauses_df.to_csv(video_dir / "vad_pauses.csv", index=False)

print()
print("First 10 continuous acoustic intervals (millisecond bounds):")
display(instant_intervals_df.head(10))""")

    # Cell 12: Inspector 2 Markdown
    add_md(r"""### 📍 Interactive Inspector 2: Continuous Voice Activity & Energy Pointer (60 FPS Animated Playback)
- **🎬 60 FPS Real-Time Cursor Tracking:** When you press **Play**, the red pointer line sweeps smoothly across both the Waveform and the Short-Time Energy curve in real time with the audio!
- **⚡ Instantaneous Dynamic Badge:** Dynamically switches between `🟢 NO SILENCE [ACTIVE SPEECH]` and `⚪ SILENCE [NO SPEECH / PAUSE]` as audio playback crosses millisecond acoustic boundaries.
- **🎯 Click-to-Seek:** Click anywhere directly on the plot to jump playback & cursor to that exact instant!
- **🌐 Global Scrubber:** Navigate anywhere across the full recording using the slider, jump buttons (`-15s`, `+15s`), or `[⏭ Next Silence]`.""")

    # Cell 13: Inspector 2 Code
    add_code(r"""# 🔍 Cell 2 of 4: Continuous Voice Activity & Energy Pointer Inspector
launch_timeline_pointer_inspector(
    audio_samples=audio_samples,
    vad_timeline_df=vad_timeline_df,
    pauses_df=pauses_df,
    sr=DEFAULT_SR
)""")

    # Cell 14: Inspector 3 Markdown
    add_md(r"""### 🔍 Interactive Inspector 3: No-Speech Pauses & Boundary Safe Zones
Inspect pause candidates, listen to word transitions, and verify safe cut boundaries.""")

    # Cell 15: Inspector 3 Code
    add_code(r"""# 🔍 Cell 3 of 4: Inspect No-Speech Pauses & Boundary Safe Zones
launch_vad_overlay_inspector(
    audio_samples=audio_samples,
    vad_timeline_df=vad_timeline_df,
    pauses_df=pauses_df,
    sr=DEFAULT_SR
)""")

    # Cell 16: Step 4 Cut Engine v2 Markdown
    add_md(r"""## ✂️ Step 4: Cut Engine v2 (Global DP Optimizer & Word Verification)
### 🎯 Implementation of Student 1 Improvement Plan (P0 Specification)
1. **Global Shortest-Path DP:** Replaces greedy per-pause cutter with global dynamic programming over candidates.
2. **Priority Window (5.0s – 15.0s):** Free with soft prior centered on 10s ($LAM=0.3$).
3. **Fallback Window (15.0s – 18.0s):** Used deliberately only when no 5–15s path exists ($penalty=10.0$); always flagged for review.
4. **Whisper Word-Timestamp Verification:** Candidates overlapping words are rejected; survivors verified ($clearance=30$ms).
5. **Strict 100–200ms Silence Margin Rule:** Each side keeps 100–200ms true silence; 8ms Hann fade applied inside margin.
6. **Frozen Schema:** Emits extended `segments.csv` with `cut_certainty`, `cut_tier`, `word_check_passed`, `needs_review`.""")

    # Cell 17: Step 4 Cut Engine v2 Code
    add_code(r"""print("✂️ Executing Cut Engine v2 (Global DP Optimizer & Word Verification)...")
print("• Priority Window : 5.0s - 15.0s (18.0s fallback ceiling)")
print("• Margin Rule     : 100ms - 200ms true silence on each side")
print("• Soft Fading     : 8ms Hann window applied strictly inside margin")
print("• Word Protection : faster-whisper word-boundary overlap rejection")
print()

segments_df = cut_speech_and_mixed_segments(
    audio=audio_samples,
    regions_df=regions_df,
    out_dir=video_dir,
    sr=DEFAULT_SR,
    silence_threshold_s=SILENCE_THRESHOLD_S,
    extended_max_dur=18.0,
    pauses=pauses_df,
    vad_backend="auto",
    use_word_verification=True
)

# Display first 10 segments with frozen schema
display(segments_df.head(10))""")

    # Cell 18: Inspector 4 Markdown
    add_md(r"""### 🔍 Interactive Inspector 4: Inspect Segments & LOW-Certainty Review Queue
Review cut segments with prominent red alerts on LOW certainty cuts, verify word safety, listen to boundaries, and override classifications. All overrides are logged to `changes.csv`.""")

    # Cell 19: Inspector 4 Code
    add_code(r"""# 🔍 Cell 4 of 4: Inspect Segments & LOW-Certainty Review Queue
launch_segment_inspector(
    audio_samples=audio_samples,
    segments_df=segments_df,
    video_dir=video_dir,
    sr=DEFAULT_SR
)""")

    # Cell 20: Step 5 Quality Control Markdown
    add_md(r"""## 📋 Step 5: Quality Control (QC) & Ablation Report
Evaluates corpus integrity, margin compliance (100–200ms), certainty tiers, review budget, and generates the **Ablation Table (Old Cutter vs. Cut Engine v2)** for the final report.""")

    # Cell 21: Step 5 Quality Control Code
    add_code(r"""qc_results = validate_segments_integrity(
    segments_df,
    audio_samples=audio_samples,
    sr=DEFAULT_SR,
    min_dur=5.0,
    max_dur=18.0
)

# Generate comprehensive QC report with Ablation analysis
report_path = video_dir / "qc_report.md"
qc_md = generate_qc_markdown_report(segments_df, report_path=report_path)

print()
print("First 20 lines of QC Report:")
print("\n".join(qc_md.splitlines()[:20]))""")

    # Cell 22: Step 6 Dataset Splits Markdown
    add_md(r"""## 📂 Step 6: Corpus Packaging & Train / Val / Test Splits
Splits the generated segments (70% Train, 15% Val, 15% Test) with source-disjoint partitioning to prevent data leakage.""")

    # Cell 23: Step 6 Dataset Splits Code
    add_code(r"""splits = split_corpus_by_source(
    segments_df,
    out_dir=video_dir / "splits",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
)

print(f"Corpus Packaging Complete: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
print(f"Handoff ready for Student 2 (SNR / Noise Filtering) and Student 3 (Web App).")""")

    # Build notebook JSON structure with kernelspec metadata
    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Verify every single code cell with Python compiler
    for idx, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            compile(src, f"cell_{idx}", "exec")
            print(f"Cell {idx:02d} verified syntax: OK")

    # Determine workspace root and repo directory
    curr_dir = Path.cwd()
    if curr_dir.name == "Audio-Datasets-Creation-System":
        repo_dir = curr_dir
        ws_root = curr_dir.parent
    else:
        ws_root = curr_dir
        repo_dir = curr_dir / "Audio-Datasets-Creation-System"

    p1 = ws_root / "student1_v13.ipynb"
    p1.write_text(json.dumps(nb_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {p1.resolve()}")

    if repo_dir.exists():
        p2 = repo_dir / "student1_v13.ipynb"
        p2.write_text(json.dumps(nb_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {p2.resolve()}")

if __name__ == "__main__":
    build_v13()
