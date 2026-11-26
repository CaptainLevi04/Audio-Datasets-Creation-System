from __future__ import annotations

from functools import lru_cache

from PySide6.QtGui import QIcon

try:
    import qtawesome as qta
except Exception:  # the GUI still runs, just without icons
    qta = None


@lru_cache(maxsize=256)
def icon(name: str, color: str = "#e8eeff") -> QIcon:
    if qta is None:
        return QIcon()
    try:
        return qta.icon(f"mdi6.{name}", color=color)
    except Exception:
        return QIcon()
