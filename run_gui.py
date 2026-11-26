#!/usr/bin/env python3
"""SautLab: desktop annotation GUI for the speech corpus pipeline.

    python run_gui.py                       # real backend, data in ./corpus_data
    python run_gui.py --data-dir D:/corpus  # another data folder
    python run_gui.py --mock                # fake data, no models or downloads needed

The data folder has the same layout as the runner notebook (one folder per video).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="SautLab annotation GUI")
    ap.add_argument("--data-dir", default="corpus_data", help="folder with one sub-folder per video (default: corpus_data)")
    ap.add_argument("--user", default="", help="annotator name written in every change (default: your login name)")
    ap.add_argument("--classifier", choices=["auto", "ast", "heuristic"], default="auto",
                    help="region classifier for Step 3 (auto = AST when torch and transformers are installed)")
    ap.add_argument("--vad", choices=["auto", "silero", "energy"], default="auto", help="pause detector for Step 4")
    ap.add_argument("--snr-threshold", type=float, default=25.0, help="clean/noisy cut-off in dB (default 25)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None, help="device for the AST model")
    ap.add_argument("--cookies", default=None, help="YouTube cookies.txt, if YouTube blocks the download")
    ap.add_argument("--kaggle-url", default=os.environ.get("SAUTLAB_KAGGLE_URL"),
                    help="remote Kaggle GPU server URL from main-runner.ipynb (or set SAUTLAB_KAGGLE_URL)")
    ap.add_argument("--endpoint-slug", default="sautlab-endpoint",
                    help="private Kaggle dataset slug for Auto-Beacon discovery (default: sautlab-endpoint)")
    ap.add_argument("--no-auto-beacon", action="store_true",
                    help="disable automatic connection to private Kaggle Auto-Beacon")
    ap.add_argument("--import-zip", default=None,
                    help="unpack a Kaggle output .zip (compact or full) into --data-dir and hydrate locally")
    ap.add_argument("--watch-dir", default=None,
                    help="folder to watch for corpus_final_output*.zip (default: ~/Downloads and current dir)")
    ap.add_argument("--no-auto-import", action="store_true",
                    help="do not auto-import the latest Kaggle output zip from Downloads on startup")
    ap.add_argument("--no-watch", action="store_true",
                    help="do not watch Downloads for new Kaggle output zips while the GUI is open")
    ap.add_argument("--mock", action="store_true", help="use fake data instead of the real backend")
    args = ap.parse_args()

    import getpass

    user = args.user.strip() or getpass.getuser()

    from PySide6.QtWidgets import QApplication, QMessageBox

    from sautlab.main_window import MainWindow
    from sautlab.theme import STYLESHEET

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    backend_factory = None
    watch_dirs = None
    try:
        if args.mock:
            from sautlab.backend.mock import MockBackend

            backend = MockBackend(args.data_dir if args.data_dir != "corpus_data" else "mock_workspace")
        else:
            from sautlab.backend.remote_kaggle import RemoteKaggleBackend
            from sautlab.kaggle_bridge import (
                auto_import_latest_zip,
                default_downloads_dirs,
                record_auto_import,
                resolve_live_kaggle_url,
                unpack_corpus_archive,
            )

            watch_dirs = [Path(args.watch_dir)] if args.watch_dir else default_downloads_dirs()
            if args.import_zip:
                vids = unpack_corpus_archive(args.import_zip, args.data_dir, hydrate=True)
                record_auto_import(args.import_zip, args.data_dir, vids)
            elif not args.no_auto_import:
                zp, vids = auto_import_latest_zip(args.data_dir, search_dirs=watch_dirs)
                if zp is not None:
                    print(f"[SautLab] Auto-imported {zp.name} -> {len(vids)} video(s): {', '.join(vids)}")

            kaggle_url = args.kaggle_url
            if not kaggle_url and not args.no_auto_beacon:
                live_url, health = resolve_live_kaggle_url(slug=args.endpoint_slug, timeout_s=4.0)
                if live_url:
                    kaggle_url = live_url
                    dev = (health or {}).get("device", "Kaggle")
                    print(f"[SautLab] Auto-connected to Kaggle GPU beacon: {live_url} ({dev})")

            def backend_factory(folder: str | Path) -> RemoteKaggleBackend:
                cur_url = getattr(backend, "remote_url", kaggle_url)
                return RemoteKaggleBackend(
                    folder,
                    remote_url=cur_url,
                    auto_beacon=not args.no_auto_beacon,
                    endpoint_slug=args.endpoint_slug,
                    classifier=args.classifier,
                    vad=args.vad,
                    snr_threshold=args.snr_threshold,
                    cookies_file=args.cookies,
                    device=args.device,
                )

            backend = RemoteKaggleBackend(
                args.data_dir,
                remote_url=kaggle_url,
                auto_beacon=not args.no_auto_beacon,
                endpoint_slug=args.endpoint_slug,
                classifier=args.classifier,
                vad=args.vad,
                snr_threshold=args.snr_threshold,
                cookies_file=args.cookies,
                device=args.device,
            )
    except ImportError as e:
        QMessageBox.critical(None, "SautLab", f"{e}\n\nInstall the requirements with:\n  pip install -r requirements-gui.txt\n\nOr start with --mock to try the GUI with fake data.")
        return 1
    win = MainWindow(
        backend,
        user,
        backend_factory=backend_factory,
        watch_downloads=None if (args.mock or args.no_watch) else watch_dirs,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
