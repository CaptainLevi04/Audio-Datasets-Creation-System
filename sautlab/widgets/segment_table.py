from __future__ import annotations

from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableView,
    QToolButton,
)

from .. import theme
from ..controller import ProjectController
from ..icons import icon
from ..models import LABEL_TITLES, LABELS, STATUS_TITLES, Segment, fmt_time
from .common import Card, tool_button

COLUMNS = ["#", "Start", "End", "Duration", "Label", "SNR", "Clean/Noisy", "Status"]
SORT_ROLE = Qt.UserRole + 1


def short_time(t: float) -> str:
    return fmt_time(t, True)[:-2]


class SegmentModel(QAbstractTableModel):
    def __init__(self, controller: ProjectController):
        super().__init__()
        self.c = controller

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.c.segments)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLUMNS[section]
        return None

    def segment(self, row: int) -> Segment:
        return self.c.segments[row]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        s = self.c.segments[index.row()]
        col = index.column()
        if role == Qt.DisplayRole:
            if col == 0:
                return str(index.row() + 1)
            if col == 1:
                return short_time(s.start)
            if col == 2:
                return short_time(s.end)
            if col == 3:
                return f"{s.duration:.1f} s"
            if col == 4:
                return LABEL_TITLES[s.label]
            if col == 5:
                return f"{s.snr_mean:.1f} dB" if s.snr_mean is not None and s.has_text else "–"
            if col == 6:
                return (s.quality or "–").capitalize() if s.has_text else "–"
            if col == 7:
                return STATUS_TITLES[s.status]
        elif role == Qt.DecorationRole:
            if col == 4:
                return icon_dot(theme.LABEL_COLORS[s.label])
            if col == 7:
                return icon_dot(theme.STATUS_COLORS[s.status])
        elif role == Qt.ForegroundRole:
            if col == 7:
                return QBrush(QColor(theme.STATUS_COLORS[s.status]))
            if col == 6 and s.has_text and s.quality == "noisy":
                return QBrush(QColor("#fbbf24"))
            if col in (1, 2):
                return QBrush(QColor("#7fb0ff"))
            if col == 0:
                return QBrush(QColor(theme.MUTED))
        elif role == SORT_ROLE:
            return {
                0: index.row(),
                1: s.start,
                2: s.end,
                3: s.duration,
                4: s.label,
                5: s.snr_mean if s.snr_mean is not None and s.has_text else -1.0,
                6: s.quality or "",
                7: s.status,
            }[col]
        elif role == Qt.TextAlignmentRole:
            return int(Qt.AlignVCenter | Qt.AlignLeft)
        return None

    def reload(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def row_changed(self, sid: int) -> None:
        r = self.c.index_of(sid)
        if r >= 0:
            self.dataChanged.emit(self.index(r, 0), self.index(r, len(COLUMNS) - 1))


_dot_cache: dict[str, object] = {}


def icon_dot(color: str):
    if color not in _dot_cache:
        _dot_cache[color] = theme.dot_icon(color, 11)
    return _dot_cache[color]


class SegmentFilter(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.text = ""
        self.label = "all"
        self.status = "all"
        self.setSortRole(SORT_ROLE)
        self.setDynamicSortFilter(False)

    def set_filters(self, text=None, label=None, status=None) -> None:
        if text is not None:
            self.text = text.strip().lower()
        if label is not None:
            self.label = label
        if status is not None:
            self.status = status
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent) -> bool:
        src = self.sourceModel()
        s = src.segment(row)
        if self.label != "all" and s.label != self.label:
            return False
        if self.status == "conflicts" and s.status not in ("needs_review", "difficult"):
            return False
        if self.status in ("accepted", "needs_review", "difficult") and s.status != self.status:
            return False
        if self.text:
            hay = " ".join(
                (str(row + 1), s.label, STATUS_TITLES[s.status], s.asr_a, s.asr_b, s.final_text, s.notes)
            ).lower()
            if self.text not in hay:
                return False
        return True


class SegmentTable(Card):
    addClicked = Signal()
    splitClicked = Signal()
    mergeClicked = Signal()
    deleteClicked = Signal()
    playSegment = Signal(int)
    reprocessRequested = Signal(int)

    def __init__(self, controller: ProjectController):
        super().__init__("", None, margins=(12, 10, 12, 8), spacing=8)
        self.c = controller
        self._syncing = False

        self.title = QLabel("Segments (0)")
        self.title.setObjectName("cardTitle")
        self.title.setMinimumWidth(128)
        ic = QLabel()
        ic.setPixmap(icon("format-list-bulleted", "#9db4ff").pixmap(18, 18))

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search segments...")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(60)
        self.label_combo = QComboBox()
        self.label_combo.addItem("All Labels", "all")
        for lab in LABELS:
            self.label_combo.addItem(icon_dot(theme.LABEL_COLORS[lab]), LABEL_TITLES[lab], lab)
        self.status_combo = QComboBox()
        for text, key in (
            ("All Segments", "all"),
            ("Conflicts Only", "conflicts"),
            ("Needs Review", "needs_review"),
            ("Difficult", "difficult"),
            ("Accepted", "accepted"),
        ):
            self.status_combo.addItem(text, key)
        self.status_combo.setToolTip("Show all segments or only the ones that need a human decision")
        self.sort_combo = QComboBox()
        for text, col in (("Start Time", 1), ("Duration", 3), ("SNR", 5), ("Label", 4), ("Status", 7)):
            self.sort_combo.addItem(text, col)
        self.prev_conflict = tool_button("chevron-up", "Previous conflict (Shift+N)", size=18)
        self.next_conflict = tool_button("chevron-down", "Next conflict (N)", size=18)

        self.b_add = self._btn("plus", "Add", "primary")
        self.b_split = self._btn("content-cut", "Split")
        self.b_merge = self._btn("call-merge", "Merge")
        self.b_delete = self._btn("delete-outline", "Delete", "danger")

        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(ic)
        bar.addWidget(self.title)
        bar.addSpacing(6)
        bar.addWidget(self.search, 1)
        bar.addWidget(self.label_combo)
        bar.addWidget(self.status_combo)
        bar.addWidget(self.prev_conflict)
        bar.addWidget(self.next_conflict)
        bar.addWidget(self.sort_combo)
        bar.addSpacing(4)
        for b in (self.b_add, self.b_split, self.b_merge, self.b_delete):
            bar.addWidget(b)
        self.body.addLayout(bar)

        self.model = SegmentModel(controller)
        self.proxy = SegmentFilter()
        self.proxy.setSourceModel(self.model)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setShowGrid(False)
        self.view.setAlternatingRowColors(False)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(30)
        self.view.setSortingEnabled(False)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        hh = self.view.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hh.setSectionsClickable(True)
        hh.setSortIndicatorShown(True)
        hh.setSectionResizeMode(QHeaderView.Interactive)
        for i, w in enumerate((46, 110, 110, 86, 110, 90, 110, 130)):
            self.view.setColumnWidth(i, w)
        self.body.addWidget(self.view, 1)

        self.search.textChanged.connect(lambda t: self._filters(text=t))
        self.label_combo.currentIndexChanged.connect(
            lambda: self._filters(label=self.label_combo.currentData())
        )
        self.status_combo.currentIndexChanged.connect(
            lambda: self._filters(status=self.status_combo.currentData())
        )
        self.sort_combo.currentIndexChanged.connect(lambda: self._sort(self.sort_combo.currentData()))
        hh.sectionClicked.connect(self._header_clicked)
        self.prev_conflict.clicked.connect(lambda: self.c.select_next_conflict(-1))
        self.next_conflict.clicked.connect(lambda: self.c.select_next_conflict(1))
        self.b_add.clicked.connect(self.addClicked)
        self.b_split.clicked.connect(self.splitClicked)
        self.b_merge.clicked.connect(self.mergeClicked)
        self.b_delete.clicked.connect(self.deleteClicked)
        self.view.selectionModel().selectionChanged.connect(self._table_selection)
        self.view.doubleClicked.connect(self._double_clicked)
        self.view.customContextMenuRequested.connect(self._context_menu)

        self.c.structureChanged.connect(self._structure)
        self.c.segmentChanged.connect(self._changed)
        self.c.selectionChanged.connect(self._controller_selection)
        self._sort_col = 1
        self._sort_order = Qt.AscendingOrder

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        w = self.width()
        self.sort_combo.setVisible(w >= 1000)
        compact = w < 800
        for b in (self.b_add, self.b_split, self.b_merge, self.b_delete):
            b.setText("" if compact else b.property("full"))
            b.setToolTip(b.property("full") if compact else "")

    def _btn(self, icon_name: str, text: str, name: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setProperty("full", text)
        b.setIcon(icon(icon_name, "#ffffff" if name == "primary" else "#e8eeff"))
        if name:
            b.setObjectName(name)
        b.setCursor(Qt.PointingHandCursor)
        return b

    # filtering / sorting
    def _filters(self, **kw) -> None:
        self.proxy.set_filters(**kw)
        self._update_title()
        self._controller_selection()

    def _sort(self, col: int) -> None:
        self._sort_col = col
        self._sort_order = Qt.AscendingOrder
        self._apply_sort()

    def _header_clicked(self, col: int) -> None:
        if col == 0:
            col = 1
        if col == self._sort_col:
            self._sort_order = Qt.DescendingOrder if self._sort_order == Qt.AscendingOrder else Qt.AscendingOrder
        else:
            self._sort_col, self._sort_order = col, Qt.AscendingOrder
        self._apply_sort()

    def _apply_sort(self) -> None:
        self.proxy.sort(self._sort_col, self._sort_order)
        self.view.horizontalHeader().setSortIndicator(self._sort_col, self._sort_order)

    def _update_title(self) -> None:
        total = len(self.c.segments)
        shown = self.proxy.rowCount()
        self.title.setText(f"Segments ({total})" if shown == total else f"Segments ({shown} of {total})")

    # model sync
    def _structure(self) -> None:
        self.model.reload()
        self._apply_sort()
        self._update_title()

    def _changed(self, sid: int) -> None:
        self.model.row_changed(sid)
        if self.proxy.status != "all" or self.proxy.text:
            self.proxy.invalidateFilter()
            self._update_title()

    def _table_selection(self) -> None:
        if self._syncing:
            return
        rows = sorted({self.proxy.mapToSource(i).row() for i in self.view.selectionModel().selectedRows()})
        ids = [self.c.segments[r].id for r in rows if r < len(self.c.segments)]
        cur = self.view.currentIndex()
        if cur.isValid():
            cur_id = self.c.segments[self.proxy.mapToSource(cur).row()].id
            if cur_id in ids:
                ids.remove(cur_id)
                ids.insert(0, cur_id)
        self._syncing = True
        self.c.set_selection(ids)
        self._syncing = False

    def _controller_selection(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        sel = QItemSelection()
        first = None
        for sid in self.c.selected:
            r = self.c.index_of(sid)
            idx = self.proxy.mapFromSource(self.model.index(r, 0))
            if idx.isValid():
                last = self.proxy.index(idx.row(), len(COLUMNS) - 1)
                sel.select(idx, last)
                first = first or idx
        sm = self.view.selectionModel()
        sm.select(sel, QItemSelectionModel.ClearAndSelect)
        if first is not None:
            sm.setCurrentIndex(first, QItemSelectionModel.NoUpdate)
            self.view.scrollTo(first, QAbstractItemView.EnsureVisible)
        self._syncing = False

    def _double_clicked(self, index) -> None:
        r = self.proxy.mapToSource(index).row()
        self.playSegment.emit(self.c.segments[r].id)

    def _context_menu(self, pos) -> None:
        if not self.c.selected:
            return
        sid = self.c.selected[0]
        ids = list(self.c.selected)
        m = QMenu(self)
        m.addAction(icon("play", "#e8eeff"), "Play segment", lambda: self.playSegment.emit(sid))
        m.addSeparator()
        lab = m.addMenu("Set label")
        for key in LABELS:
            lab.addAction(icon_dot(theme.LABEL_COLORS[key]), LABEL_TITLES[key], lambda k=key: self.c.set_label(ids, k))
        st = m.addMenu("Set status")
        for key in ("accepted", "needs_review", "difficult"):
            st.addAction(icon_dot(theme.STATUS_COLORS[key]), STATUS_TITLES[key], lambda k=key: self.c.set_status(ids, k))
        m.addSeparator()
        m.addAction(icon("content-cut", "#e8eeff"), "Split at playhead", self.splitClicked.emit)
        m.addAction(icon("call-merge", "#e8eeff"), "Merge", self.mergeClicked.emit)
        m.addAction(icon("refresh", "#e8eeff"), "Re-run SNR and ASR", lambda: self.reprocessRequested.emit(sid))
        m.addSeparator()
        m.addAction(icon("delete-outline", "#ff7b7b"), "Delete", self.deleteClicked.emit)
        m.exec(self.view.viewport().mapToGlobal(pos))
