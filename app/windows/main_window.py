# app/windows/main_window.py
from __future__ import annotations

from pathlib import Path
import logging
import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTreeView, QTableView, QTabWidget,
    QVBoxLayout, QToolBar, QLabel, QAbstractItemView, QLineEdit,
    QPushButton, QHBoxLayout, QPlainTextEdit, QProgressBar, QMessageBox
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt

from app.controllers.project_controller import ProjectController
from app.state.project_context import ProjectContext
from app.qt_models.explosion_table_model import ExplosionTableModel
from app.qt_models.totals_table_model import TotalsTableModel
from app.qt_models.issues_table_model import IssuesTableModel
from app.widgets.where_used_panel import WhereUsedPanel


PBS_NODE_ID_ROLE = Qt.UserRole + 1
PBS_CODE_ROLE = Qt.UserRole + 2

_LOG = logging.getLogger(__name__)
_KEY_DEBUG_ENV = "BOM_KEY_DEBUG"
_KEY_DEBUG_TARGET = "166104001"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("mdp_tool — BOM Explorer (Qt6)")

        self._ctx: ProjectContext | None = None
        self._selected_folder: Path | None = None

        # Controller
        self._controller = ProjectController(self)
        self._controller.context_ready.connect(self._on_context_ready)
        self._controller.analysis_failed.connect(self._on_analysis_failed)
        self._controller.analysis_progress.connect(self._on_progress)
        self._controller.analysis_log.connect(self._on_log)
        self._controller.analysis_started.connect(self._on_started)
        self._controller.analysis_finished.connect(self._on_finished)

        # Toolbar top
        tb = QToolBar("Main")
        self.addToolBar(tb)

        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Seleziona una cartella progetto…")
        self.folder_edit.setReadOnly(True)

        self.btn_choose = QPushButton("Seleziona cartella")
        self.btn_analyze = QPushButton("Analizza cartella")
        self.btn_analyze.setEnabled(False)

        self.btn_choose.clicked.connect(self._choose_folder)
        self.btn_analyze.clicked.connect(self._analyze_selected)

        top_lay.addWidget(QLabel("Cartella:"))
        top_lay.addWidget(self.folder_edit, 1)
        top_lay.addWidget(self.btn_choose)
        top_lay.addWidget(self.btn_analyze)

        tb.addWidget(top)

        # Status bar: progress + status
        self._status = QLabel("Pronto.")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(220)
        self._progress.setVisible(False)

        self.statusBar().addWidget(self._status, 1)
        self.statusBar().addPermanentWidget(self._progress)

        # Left: Tree (PBS in PBS_EXCEL; BOM in PDF_ONLY fallback)
        self.tree = QTreeView()
        self.tree.setHeaderHidden(False)
        self._tree_model = QStandardItemModel()
        self._tree_model.setHorizontalHeaderLabels(["PBS"])
        self.tree.setModel(self._tree_model)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setUniformRowHeights(True)

        # Right: Tabs
        self.tabs = QTabWidget()

        # Tab 1: Explosion
        self.explosion_table = QTableView()
        self.explosion_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._explosion_model = ExplosionTableModel()
        self.explosion_table.setModel(self._explosion_model)
        self.tabs.addTab(self.explosion_table, "Explosion (selezione)")

        # Tab 2: Totals + where used
        totals_container = QWidget()
        totals_layout = QVBoxLayout(totals_container)

        self.totals_table = QTableView()
        self.totals_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._totals_model = TotalsTableModel()
        self.totals_table.setModel(self._totals_model)

        self.where_used = WhereUsedPanel()

        totals_layout.addWidget(self.totals_table, 3)
        totals_layout.addWidget(self.where_used, 2)
        self.tabs.addTab(totals_container, "Totalizzazione")

        # Tab 3: Diagnostics + Log live
        diag_container = QWidget()
        diag_layout = QVBoxLayout(diag_container)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Log analisi…")

        self.issues_table = QTableView()
        self.issues_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._issues_model = IssuesTableModel()
        self.issues_table.setModel(self._issues_model)

        diag_layout.addWidget(QLabel("Log live:"))
        diag_layout.addWidget(self.log_box, 2)
        diag_layout.addWidget(QLabel("Issues (a fine analisi):"))
        diag_layout.addWidget(self.issues_table, 3)

        self.tabs.addTab(diag_container, "Diagnostics")

        # Splitter layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(splitter)
        self.setCentralWidget(root)

        # Wiring selections
        self.tree.selectionModel().selectionChanged.connect(self._on_tree_selection_changed)
        self.totals_table.selectionModel().selectionChanged.connect(self._on_totals_selection_changed)

    # ---------------- UI actions ----------------
    def _choose_folder(self) -> None:
        folder = self._controller.choose_folder_dialog()
        if not folder:
            return
        self._selected_folder = folder
        self.folder_edit.setText(str(folder))
        self.btn_analyze.setEnabled(True)
        self._status.setText("Cartella selezionata. Premi 'Analizza cartella'.")

    def _analyze_selected(self) -> None:
        if not self._selected_folder:
            return
        self.log_box.clear()
        self.tabs.setCurrentIndex(2)  # diagnostics
        self._controller.analyze_folder(self._selected_folder)

    # ---------------- Controller signals ----------------
    def _on_started(self, msg: str) -> None:
        self.btn_choose.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText(msg)

    def _on_finished(self) -> None:
        self.btn_choose.setEnabled(True)
        self.btn_analyze.setEnabled(self._selected_folder is not None)
        self._progress.setVisible(False)

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(max(0, min(100, pct)))
        if msg:
            self._status.setText(msg)

    def _on_log(self, line: str) -> None:
        self.log_box.appendPlainText(line)

    def _on_analysis_failed(self, msg: str) -> None:
        self._status.setText(f"Errore: {msg}")
        QMessageBox.critical(self, "Analisi fallita", msg)

    # ---------------- Existing: context -> models ----------------
    def _on_context_ready(self, ctx: ProjectContext) -> None:
        self._ctx = ctx
        self._status.setText(f"OK — Root MFG: {ctx.root_code} | items={len(ctx.qty_by_code)}")

        self._issues_model.set_result(ctx.result)
        self._explosion_model.set_context(ctx)
        self._totals_model.set_context(ctx)
        self.where_used.set_context(ctx)

        self._rebuild_tree(ctx)
        self.tree.expandToDepth(1)

    # ---------------- Tree helpers ----------------
    def _use_pbs_tree(self, ctx: ProjectContext) -> bool:
        """
        PBS_EXCEL: se abbiamo indici PBS id-based, la tree a sinistra deve mostrare PBS vera.
        PDF_ONLY: fallback su manufacturing.
        """
        return bool(getattr(ctx, "pbs_root_id", "")) and bool(getattr(ctx, "pbs_children_by_parent_id", None))

    def _rebuild_tree(self, ctx: ProjectContext) -> None:
        self._tree_model.clear()
        self._tree_model.setHorizontalHeaderLabels(["PBS"])

        if self._use_pbs_tree(ctx):
            root_id = ctx.pbs_root_id or ""
            root_code = (ctx.pbs_code_by_id.get(root_id, "") or "").strip() or (ctx.pbs_root_code or "").strip()
            if not root_code:
                root_code = "<NO PBS ROOT>"

            root_item = QStandardItem(f"{root_code}  (PBS ROOT)")
            root_item.setData(root_id, PBS_NODE_ID_ROLE)
            root_item.setData(root_code, PBS_CODE_ROLE)
            self._tree_model.appendRow(root_item)

            self._add_children_iter_pbs_id(ctx, root_item, root_id)

        else:
            # PDF_ONLY / legacy: mostra manufacturing tree (BOM)
            root_code = (ctx.root_code or "").strip() or "<NO ROOT>"
            root_item = QStandardItem(f"{root_code}  (MFG ROOT)")
            root_item.setData("", PBS_NODE_ID_ROLE)   # no id
            root_item.setData(root_code, PBS_CODE_ROLE)
            self._tree_model.appendRow(root_item)

            self._add_children_recursive_bom(ctx, root_item, root_code)

        self.tree.setModel(self._tree_model)

    def _add_children_iter_pbs_id(self, ctx: ProjectContext, parent_item: QStandardItem, parent_id: str) -> None:
        """
        Builder PBS robusto:
          - usa node_id (occurrence-based)
          - iterativo (niente recursion depth)
        """
        stack: list[tuple[QStandardItem, str]] = [(parent_item, parent_id)]

        while stack:
            qitem, pid = stack.pop()

            for cid in ctx.pbs_children_by_parent_id.get(pid, []):
                ccode = (ctx.pbs_code_by_id.get(cid, "") or "").strip()
                if not ccode:
                    ccode = "<NO CODE>"

                item = QStandardItem(ccode)
                item.setData(cid, PBS_NODE_ID_ROLE)
                item.setData(ccode, PBS_CODE_ROLE)
                qitem.appendRow(item)

                stack.append((item, cid))

    def _add_children_recursive_bom(self, ctx: ProjectContext, parent_item: QStandardItem, parent_code: str) -> None:
        # manufacturing tree (legacy alias children_by_parent)
        for e in ctx.children_by_parent.get(parent_code, []):
            child = getattr(e, "child_code", "") or ""
            if not child:
                continue
            item = QStandardItem(child)
            item.setData("", PBS_NODE_ID_ROLE)  # no id
            item.setData(child, PBS_CODE_ROLE)
            parent_item.appendRow(item)
            self._add_children_recursive_bom(ctx, item, child)

    # ---------------- Selection logic ----------------
    def _on_tree_selection_changed(self, *_args) -> None:
        if self._ctx is None:
            return
        idx = self.tree.currentIndex()
        if not idx.isValid():
            return
        item = self._tree_model.itemFromIndex(idx)

        pbs_node_id = item.data(PBS_NODE_ID_ROLE) or ""
        code = item.data(PBS_CODE_ROLE) or ""
        code = str(code).strip()
        if not code:
            return

        # Se stiamo mostrando PBS tree, il click NON implica necessariamente che esista explosion manufacturing
        if self._use_pbs_tree(self._ctx):
            has_mfg = (
                code == (self._ctx.root_code or "")
                or code in self._ctx.bom_children_by_parent
                or code in self._ctx.bom_occurrences_by_code
            )

            if not has_mfg:
                # PBS-only / documentale / non esploso
                self._explosion_model.set_edges([])
                self.tabs.setCurrentIndex(2)  # Diagnostics

                # Log PBS-only, includendo node_id per disambiguare occorrenze
                self._status.setText(f"PBS-only: '{code}' (node {pbs_node_id}) non ha esplosione manufacturing.")
                self.log_box.appendPlainText(f"[INFO] PBS-only node selected: code={code} node_id={pbs_node_id} (no manufacturing explosion)")
                return

            # ha manufacturing: mostriamo la porzione di explosion a partire da quel code
            edges = self._collect_subtree_edges_bom(self._ctx, code)
            self._explosion_model.set_edges(edges)
            self.tabs.setCurrentIndex(0)
            return

        # Legacy (PDF_ONLY): tree == manufacturing
        edges = self._collect_subtree_edges_legacy(self._ctx, code)
        self._explosion_model.set_edges(edges)
        self.tabs.setCurrentIndex(0)

    def _collect_subtree_edges_bom(self, ctx: ProjectContext, root_code: str) -> list[object]:
        out: list[object] = []
        stack = [root_code]
        while stack:
            parent = stack.pop()
            for e in ctx.bom_children_by_parent.get(parent, []):
                out.append(e)
                child = getattr(e, "child_code", "") or ""
                if child:
                    stack.append(child)
        return out

    def _collect_subtree_edges_legacy(self, ctx: ProjectContext, root_code: str) -> list[object]:
        """
        Compat: usa children_by_parent (alias manufacturing).
        """
        out: list[object] = []
        stack = [root_code]
        while stack:
            parent = stack.pop()
            for e in ctx.children_by_parent.get(parent, []):
                out.append(e)
                child = getattr(e, "child_code", "") or ""
                if child:
                    stack.append(child)
        return out

    def _on_totals_selection_changed(self, *_args) -> None:
        if self._ctx is None:
            return
        idx = self.totals_table.currentIndex()
        if not idx.isValid():
            return
        code = self._totals_model.code_at_row(idx.row())
        if (os.getenv(_KEY_DEBUG_ENV, "").strip() == "1") and (_KEY_DEBUG_TARGET in (code or "")):
            lookup_key = (code or "").strip()
            _LOG.info(
                "[KEY_DEBUG][totals-selection] totals_selection_key=%s where_used_lookup_key=%s index_contains_key=%s",
                code,
                lookup_key,
                lookup_key in (self._ctx.parents_by_child if self._ctx else {}),
            )
        self.where_used.show_code(code)
