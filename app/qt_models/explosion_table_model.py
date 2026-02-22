# app/qt_models/explosion_table_model.py
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from app.state.project_context import ProjectContext


COLUMNS = [
    "PN", "REV", "QTY", "UOM",
    "Manufacturer", "Manufacturer code", "Description",
    "Parent", "Depth"
]


class ExplosionTableModel(QAbstractTableModel):
    def __init__(self, ctx: Optional[ProjectContext] = None) -> None:
        super().__init__()
        self._ctx = ctx
        self._rows: List[object] = []  # list[ExplosionEdge]

    def set_context(self, ctx: ProjectContext) -> None:
        self.beginResetModel()
        self._ctx = ctx
        self._rows = []
        self.endResetModel()

    def set_edges(self, edges: List[object]) -> None:
        self.beginResetModel()
        self._rows = list(edges)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return COLUMNS[section]
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or self._ctx is None:
            return None
        if role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None

        e = self._rows[index.row()]
        col = index.column()

        child_code = getattr(e, "child_code", "") or ""
        child_rev = getattr(e, "child_rev", "") or ""
        qty = getattr(e, "qty", None)
        parent_code = getattr(e, "parent_code", "") or ""
        depth = getattr(e, "depth", None)

        info = self._ctx.part_master.get(child_code)
        uom = self._ctx.uom_by_code.get(child_code, "")

        vals = [
            child_code,
            child_rev,
            "" if qty is None else str(qty),
            uom,
            "" if info is None else info.manufacturer,
            "" if info is None else info.manufacturer_code,
            "" if info is None else info.description,
            parent_code,
            "" if depth is None else str(depth),
        ]
        return vals[col]
