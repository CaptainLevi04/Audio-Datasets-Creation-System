#!/usr/bin/env python3
"""Speech Corpus CLI Runner - Student 1 (Audio Pipeline: Steps 1-4).
Usage:
  python run_pipeline.py --demo
  python run_pipeline.py --url "https://www.youtube.com/watch?v=..." --out-dir data
  python run_pipeline.py --file "my_audio_or_video.mp4" --out-dir data
"""

import argparse
import sys
from pathlib import Path

from speech_corpus import run_mock_pipeline, run_student1_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Student 1 Audio Pipeline: Download, Audio Extraction, Labelling, and 5-15s Segmentation."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run an instant offline demonstration with synthetic mock audio (no downloads or ffmpeg needed).",
    )
    group.add_argument(
        "--url",
        type=str,
        help="Online video URL (e.g. YouTube) to download and process.",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to a local video or audio file to process.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Destination directory for processed corpora (default: data).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ast", "heuristic", "panns"],
        default="ast",
        help="Region classification backend for Step 3 (default: ast).",
    )
    parser.add_argument(
        "--vad",
        type=str,
        choices=["auto", "silero", "energy"],
        default="auto",
        help="VAD/Pause detection backend for Step 4 (default: auto).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=360,
        help="Maximum video height for yt-dlp download (default: 360, fast & small).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for AST neural model execution (default: auto).",
    )
    parser.add_argument(
        "--cut-video",
        action="store_true",
        help="Also cut video clips (.mp4) alongside audio chunks (.wav) in Step 4.",
    )

    args = parser.parse_args()

    if args.demo:
        demo_dir = Path(args.out_dir) / "demo_run"
        print(f"Running instant mock demo in: {demo_dir}")
        res = run_mock_pipeline(out_dir=demo_dir, backend=args.backend)
        print("\nDemo completed successfully!")
        print(f"Regions and segments created in: {res['out_dir']}")
        return

    source = args.url if args.url else args.file
    is_url = bool(args.url)
    device = None if args.device == "auto" else args.device

    res = run_student1_pipeline(
        source=source,
        data_dir=args.out_dir,
        is_url=is_url,
        backend=args.backend,
        vad_backend=args.vad,
        max_height=args.max_height,
        cut_video=args.cut_video,
        device=device,
    )
    print("\nExecution finished with status:", res["status"])


if __name__ == "__main__":
    main()
