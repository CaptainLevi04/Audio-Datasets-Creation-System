from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..icons import icon
from ..models import fmt_time, parse_time


class Card(QFrame):
    """Rounded panel with an optional icon + title row. Add content to `body`."""

    def __init__(self, title: str = "", icon_name: str | None = None, margins=(12, 10, 12, 10), spacing: int = 8):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(spacing)
        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        if title:
            if icon_name:
                ic = QLabel()
                ic.setPixmap(icon(icon_name, "#9db4ff").pixmap(18, 18))
                self.header.addWidget(ic)
            lbl = QLabel(title)
            lbl.setObjectName("cardTitle")
            self.header.addWidget(lbl)
            self.header.addStretch(1)
            self.body.addLayout(self.header)


class TimeSpinBox(QDoubleSpinBox):
    """Seconds shown as hh:mm:ss.mmm; typing accepts hh:mm:ss.mmm, mm:ss or plain seconds."""

    def __init__(self):
        super().__init__()
        self.setDecimals(3)
        self.setRange(0.0, 86400.0)
        self.setSingleStep(0.1)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QDoubleSpinBox.UpDownArrows)

    def textFromValue(self, value: float) -> str:
        return fmt_time(value, True)

    def valueFromText(self, text: str) -> float:
        t = parse_time(text)
        return t if t is not None else self.value()

    def validate(self, text: str, pos: int):
        return (QValidator.State.Acceptable, text, pos)


def tool_button(icon_name: str, tip: str = "", color: str = "#e8eeff", size: int = 20, checkable: bool = False) -> QToolButton:
    b = QToolButton()
    b.setIcon(icon(icon_name, color))
    from PySide6.QtCore import QSize

    b.setIconSize(QSize(size, size))
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setCursor(Qt.PointingHandCursor)
    return b


def hbox(*widgets, spacing: int = 6, margins=(0, 0, 0, 0)) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(*margins)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        else:
            lay.addWidget(w)
    return lay


def muted_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("muted")
    return lbl


def spacer_widget() -> QWidget:
    w = QWidget()
    w.setSizePolicy(w.sizePolicy().horizontalPolicy(), w.sizePolicy().verticalPolicy())
    return w
