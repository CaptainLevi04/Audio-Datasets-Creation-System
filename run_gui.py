#!/usr/bin/env python3
"""SautLab: desktop annotation GUI for the speech corpus pipeline.

    python run_gui.py                       # real backend, data in ./corpus_data
    python run_gui.py --data-dir D:/corpus  # another data folder
    python run_gui.py --mock                # fake data, no models or downloads needed

The data folder has the same layout as the runner notebook (one folder per video).
"""
import argparse
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
    ap.add_argument("--snr-threshold", type=float, default=20.0, help="clean/noisy cut-off in dB (default 20)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None, help="device for neural models (AST / Conformer)")
    ap.add_argument("--cookies", default=None, help="YouTube cookies.txt, if YouTube blocks the download")
    ap.add_argument("--no-asr", action="store_true", help="run Steps 3-5 only and skip local Dual ASR")
    ap.add_argument("--conformer-model", default=None, help="Hugging Face ID or local .nemo path for NVIDIA Arabic Conformer")
    ap.add_argument("--audar-quant", default=None, help="Audar-ASR-V1-Turbo GGUF filename (default: Audar-ASR-V1-Turbo-Q4_K_M.gguf)")
    ap.add_argument("--mock", action="store_true", help="use fake data instead of the real backend")
    args = ap.parse_args()

    import getpass

    user = args.user.strip() or getpass.getuser()

    from PySide6.QtWidgets import QApplication, QMessageBox

    from sautlab.main_window import MainWindow
    from sautlab.theme import STYLESHEET

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    try:
        if args.mock:
            from sautlab.backend.mock import MockBackend

            backend = MockBackend(args.data_dir if args.data_dir != "corpus_data" else "mock_workspace")
        else:
            from sautlab.backend.corpus import CorpusBackend

            extra_kw = {"run_asr": not args.no_asr}
            if args.conformer_model:
                extra_kw["conformer_model"] = args.conformer_model
            if args.audar_quant:
                extra_kw["audar_quant"] = args.audar_quant
            backend = CorpusBackend(
                args.data_dir, classifier=args.classifier, vad=args.vad, snr_threshold=args.snr_threshold,
                cookies_file=args.cookies, device=args.device, **extra_kw,
            )
    except ImportError as e:
        QMessageBox.critical(None, "SautLab", f"{e}\n\nInstall the requirements with:\n  pip install -r requirements-gui.txt\n\nOr start with --mock to try the GUI with fake data.")
        return 1
    win = MainWindow(backend, user)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
