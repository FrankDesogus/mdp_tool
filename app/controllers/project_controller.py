# app/controllers/project_controller.py
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QObject, Signal
from PySide6.QtWidgets import QFileDialog

from app.workers.analyze_worker import AnalyzeWorker, AnalyzeRequest
from app.state.project_context import ProjectContext


class ProjectController(QObject):
    context_ready = Signal(ProjectContext)
    analysis_failed = Signal(str)

    # NEW: progress/log lifecycle
    analysis_started = Signal(str)
    analysis_finished = Signal()
    analysis_progress = Signal(int, str)   # pct, msg
    analysis_log = Signal(str)            # log line

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: AnalyzeWorker | None = None

    def choose_folder_dialog(self) -> Path | None:
        folder = QFileDialog.getExistingDirectory(None, "Seleziona cartella progetto")
        return None if not folder else Path(folder)

    def analyze_folder(self, folder: Path) -> None:
        self._cleanup_thread()

        self.analysis_started.emit("Avvio analisi…")
        self.analysis_progress.emit(0, "Preparazione…")
        self.analysis_log.emit(f"[INFO] Folder: {folder}")

        self._thread = QThread()

        # ✅ QUI la modifica che evita il crash QObject.__init__(AnalyzeRequest)
        self._worker = AnalyzeWorker(request=AnalyzeRequest(folder=folder))
        self._worker.moveToThread(self._thread)

        # wiring
        self._thread.started.connect(self._worker.run)

        self._worker.progress.connect(self.analysis_progress)
        self._worker.log.connect(self.analysis_log)

        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _on_finished(self, result) -> None:
        try:
            folder = getattr(result, "base_dir", None)
            if folder is None:
                # fallback: prova dal worker/request
                req = getattr(self._worker, "request", None) if self._worker else None
                folder = getattr(req, "folder", None)

            ctx = ProjectContext.from_any_result(
                folder=Path(folder) if folder else Path("."),
                result=result,
                mode=getattr(self._worker, "mode_detected", "") if self._worker else "",
            )
            self.context_ready.emit(ctx)
            self.analysis_finished.emit()
        except Exception as e:
            self.analysis_failed.emit(f"{type(e).__name__}: {e}")
            self.analysis_finished.emit()

    def _on_failed(self, msg: str) -> None:
        self.analysis_failed.emit(msg)
        self.analysis_finished.emit()

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
