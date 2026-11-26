"""Renders the main window offscreen to a PNG (used to compare with the reference design)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from sautlab.backend.mock import MockBackend
from sautlab.main_window import MainWindow
from sautlab.theme import STYLESHEET


def pump(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def main(out: str, workspace: str, minutes: float) -> None:
    app = QApplication([])
    app.setStyleSheet(STYLESHEET)
    win = MainWindow(MockBackend(workspace, minutes))
    win.resize(1600, 1000)
    win.show()
    t0 = time.time()
    while time.time() - t0 < 60:
        pump(200)
        if win.audio.audio is not None:
            break
    pump(500)
    c = win.c
    ids = c.conflict_ids()
    if ids:
        c.set_selection([ids[0]])
    pump(400)
    win.grab().save(out)
    print("saved", out, "segments", len(c.segments))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
