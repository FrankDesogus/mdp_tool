# app/qt_models/totals_table_model.py
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from app.state.project_context import ProjectContext


COLUMNS = ["PN", "QTY total", "UOM", "Description", "Manufacturer", "Manufacturer code"]


class TotalsTableModel(QAbstractTableModel):
    def __init__(self, ctx: Optional[ProjectContext] = None) -> None:
        super().__init__()
        self._ctx = ctx
        self._rows: List[Tuple[str, object]] = []

    def set_context(self, ctx: ProjectContext) -> None:
        self.beginResetModel()
        self._ctx = ctx
        # righe = qty_by_code
        self._rows = sorted(ctx.qty_by_code.items(), key=lambda x: x[0])
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

        code, qty = self._rows[index.row()]
        info = self._ctx.part_info_for(code)
        uom = self._ctx.uom_by_code.get(code, "")

        vals = [
            code,
            "" if qty is None else str(qty),
            uom,
            "" if info is None else info.description,
            "" if info is None else info.manufacturer,
            "" if info is None else info.manufacturer_code,
        ]
        return vals[index.column()]

    def code_at_row(self, row: int) -> str:
        if row < 0 or row >= len(self._rows):
            return ""
        return self._rows[row][0]
