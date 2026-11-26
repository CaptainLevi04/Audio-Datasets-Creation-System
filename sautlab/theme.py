from __future__ import annotations

from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QLinearGradient, QBrush, QPen
from PySide6.QtCore import Qt, QRectF

BG = "#091024"
PANEL = "#0f1a33"
PANEL_ALT = "#0c1529"
BORDER = "#1b2a49"
BORDER_STRONG = "#2a3d66"
TEXT = "#e8eeff"
MUTED = "#8b9bbd"
ACCENT = "#2d6bf3"
ACCENT_HOVER = "#4380ff"
SELECT_ROW = "#173366"

LABEL_COLORS = {
    "speech": "#22c55e",
    "music": "#3b82f6",
    "mixed": "#f59e0b",
    "silence": "#8e99ad",
}
STATUS_COLORS = {
    "accepted": "#22c55e",
    "needs_review": "#f59e0b",
    "difficult": "#ef4444",
}


def qcolor(hex_color: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_color)
    c.setAlphaF(alpha)
    return c


def dot_icon(color: str, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


def logo_pixmap(size: int = 34) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    heights = [0.35, 0.65, 1.0, 0.7, 0.45]
    bar_w = size / 8.5
    gap = bar_w * 0.8
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    x = (size - total) / 2
    for i, h in enumerate(heights):
        g = QLinearGradient(0, 0, 0, size)
        g.setColorAt(0, QColor("#ffb020"))
        g.setColorAt(1, QColor("#ff6a1a") if i % 2 == 0 else QColor("#2d8cff"))
        p.setBrush(QBrush(g))
        p.setPen(Qt.NoPen)
        bh = size * h * 0.9
        p.drawRoundedRect(QRectF(x, (size - bh) / 2, bar_w, bh), bar_w / 2, bar_w / 2)
        x += bar_w + gap
    p.end()
    return pm


STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 12px;
    color: {TEXT};
}}
QMainWindow, QWidget#root {{ background: {BG}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; padding: 4px; }}

QFrame#card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel {{ background: transparent; }}
QLabel#cardTitle {{ font-size: 13px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#ok {{ color: #34d399; }}

QLineEdit, QPlainTextEdit, QTextEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QPlainTextEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled {{ color: {MUTED}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {SELECT_ROW};
    outline: none;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}

QPushButton {{
    background: #16233f;
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 5px 12px;
}}
QPushButton:hover {{ background: #1d2d52; border-color: #3a5288; }}
QPushButton:pressed {{ background: #14213c; }}
QPushButton:disabled {{ color: #5a6a8a; background: #111b32; border-color: {BORDER}; }}
QPushButton#primary {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    font-weight: 600;
    color: white;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: #20386d; border-color: #20386d; color: #7f93bf; }}
QPushButton#warn {{
    background: #f59e0b;
    border: 1px solid #f59e0b;
    color: #1a1204;
    font-weight: 600;
}}
QPushButton#warn:hover {{ background: #ffb02e; }}
QPushButton#success {{ border: 1px solid #1f8f4d; color: #34d399; background: #0f2a22; }}
QPushButton#success:hover {{ background: #143a2d; }}
QPushButton#danger {{ border: 1px solid #7f2a2a; color: #ff7b7b; background: #2a1215; }}
QPushButton#danger:hover {{ background: #3a171b; }}
QPushButton#flat, QToolButton#flat {{
    background: transparent; border: none; padding: 4px 6px;
}}
QToolButton {{ background: transparent; border: none; border-radius: 5px; padding: 4px; }}
QToolButton:hover {{ background: #1d2d52; }}
QToolButton:checked {{ background: #1d3a78; }}
QToolButton#menuBtn {{ padding: 6px 10px; }}
QToolButton#menuBtn::menu-indicator {{ image: none; }}
QToolButton#round {{ background: {ACCENT}; border-radius: 17px; padding: 0; }}
QToolButton#round:hover {{ background: {ACCENT_HOVER}; }}

QMenu {{ background: {PANEL}; border: 1px solid {BORDER_STRONG}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {SELECT_ROW}; }}
QMenu::item:disabled {{ color: #5a6a8a; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

QTableView {{
    background: {PANEL};
    alternate-background-color: {PANEL};
    border: none;
    gridline-color: transparent;
    selection-background-color: {SELECT_ROW};
    selection-color: {TEXT};
    outline: none;
}}
QTableView::item {{ padding: 0 8px; border: none; }}
QTableView::item:selected {{ background: {SELECT_ROW}; }}
QHeaderView::section {{
    background: {PANEL};
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {PANEL}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #2a3d66; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #3a5288; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #2a3d66; border-radius: 4px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSlider::groove:horizontal {{ height: 4px; background: #24365d; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #dbe6ff; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
}}
QProgressBar {{ background: #1a2a4d; border: none; border-radius: 2px; max-height: 4px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 6px; }}
QListWidget::item:selected {{ background: {SELECT_ROW}; }}
QListWidget::item:hover:!selected {{ background: #13213f; }}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: #223660; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

QFrame#asrCard {{
    background: #12203f;
    border: 1px solid {BORDER_STRONG};
    border-radius: 7px;
}}
QFrame#asrCard[chosen="true"] {{ border: 1px solid {ACCENT}; background: #13265a; }}
QLabel#chip {{
    background: #0f2a22; color: #34d399; border: 1px solid #1f8f4d;
    border-radius: 11px; padding: 3px 14px; font-weight: 600;
}}
QLabel#chip[tone="noisy"] {{ background: #2a1f0c; color: #fbbf24; border-color: #a16207; }}
QLabel#chip[tone="none"] {{ background: #17223e; color: {MUTED}; border-color: {BORDER_STRONG}; }}
"""
