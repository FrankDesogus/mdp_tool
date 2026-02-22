# app/qt_models/issues_table_model.py
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from core.use_case.analyze_folder_pdf import Issue, AnalyzeFolderPdfResult


COLUMNS = ["Level", "Code", "Message", "Path"]


class IssuesTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._issues: List[Issue] = []

    def set_result(self, r: Optional[AnalyzeFolderPdfResult]) -> None:
        self.beginResetModel()
        self._issues = [] if r is None else list(r.issues or [])
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._issues)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return COLUMNS[section]
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        if role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None

        i = self._issues[index.row()]
        path = "" if i.path is None else str(i.path)

        vals = [i.level, i.code, i.message, path]
        return vals[index.column()]
