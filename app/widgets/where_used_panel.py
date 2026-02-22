# app/widgets/where_used_panel.py
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PySide6.QtCore import Qt

from app.state.project_context import ProjectContext


class WhereUsedPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._ctx: ProjectContext | None = None

        self._title = QLabel("Where-used")
        self._title.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._parents_list = QListWidget()
        self._occ_list = QListWidget()

        lay = QVBoxLayout(self)
        lay.addWidget(self._title)
        lay.addWidget(QLabel("Padri possibili:"))
        lay.addWidget(self._parents_list, 1)
        lay.addWidget(QLabel("Occorrenze (parent / depth / path):"))
        lay.addWidget(self._occ_list, 2)

    def set_context(self, ctx: ProjectContext) -> None:
        self._ctx = ctx
        self.clear()

    def clear(self) -> None:
        self._title.setText("Where-used")
        self._parents_list.clear()
        self._occ_list.clear()

    def show_code(self, code: str) -> None:
        if self._ctx is None or not code:
            self.clear()
            return

        self._title.setText(f"Where-used — {code}")

        self._parents_list.clear()
        for p in sorted(self._ctx.parents_by_child.get(code, set())):
            self._parents_list.addItem(QListWidgetItem(p))

        self._occ_list.clear()
        for e in self._ctx.occurrences_by_code.get(code, []):
            parent = getattr(e, "parent_code", "") or ""
            depth = getattr(e, "depth", "")
            path = getattr(e, "path", "")
            self._occ_list.addItem(QListWidgetItem(f"{parent} | depth={depth} | {path}"))
