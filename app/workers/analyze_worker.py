# app/workers/analyze_worker.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.use_case.analyze_folder_pdf import AnalyzeFolderPdfUseCase, AnalyzeFolderPdfResult
from core.use_case.analyze_folder import AnalyzeFolderUseCase, AnalyzeFolderResult


@dataclass(frozen=True)
class AnalyzeRequest:
    folder: Path


class AnalyzeWorker(QObject):
    """
    Worker eseguito su QThread.
    - auto-detect mode (PDF_ONLY vs PBS_EXCEL)
    - chiama il use case corretto
    - emette progress/log durante l'analisi
    """

    # Restiamo compatibili col tuo controller attuale: finished emette SOLO result
    finished = Signal(object)  # AnalyzeFolderPdfResult | AnalyzeFolderResult
    failed = Signal(str)

    progress = Signal(int, str)  # pct, msg
    log = Signal(str)

    def __init__(self, *, request: AnalyzeRequest, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._request = request
        self.mode_detected: str = ""

    @property
    def request(self) -> AnalyzeRequest:
        # utile per controller (_on_finished fallback)
        return self._request

    @Slot()
    def run(self) -> None:
        try:
            folder = self._request.folder

            # auto-detect
            mode = self._detect_mode(folder)
            self.mode_detected = mode
            self.log.emit(f"[INFO] Mode detected: {mode}")

            def log_cb(level: str, msg: str) -> None:
                # normalize
                lvl = (level or "INFO").upper()
                self.log.emit(f"[{lvl}] {msg}")

            def progress_cb(done: int, total: int, msg: str) -> None:
                pct = 0 if total <= 0 else int(done * 100 / total)
                self.progress.emit(pct, msg)

            if mode == "PDF_ONLY":
                uc = AnalyzeFolderPdfUseCase()
                res: AnalyzeFolderPdfResult = uc.run(folder, progress_cb=progress_cb, log_cb=log_cb)  # type: ignore
                self.finished.emit(res)
            else:
                uc = AnalyzeFolderUseCase()
                res: AnalyzeFolderResult = uc.run(folder, progress_cb=progress_cb, log_cb=log_cb)  # type: ignore
                self.finished.emit(res)

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _detect_mode(self, folder: Path) -> str:
        # euristica semplice:
        # - se trovo segnali forti di PBS/PARTL excel -> PBS_EXCEL
        # - altrimenti se trovo PDF e non Excel -> PDF_ONLY
        # - se misto: preferiamo PBS_EXCEL
        excel = list(folder.rglob("*.xls*"))
        pdf = list(folder.rglob("*.pdf"))

        excel_names = [p.name.upper() for p in excel]
        has_partl = any("PARTL" in n for n in excel_names)
        has_pbs = any(("PBS" in n) or ("MDP" in n) for n in excel_names)

        if has_partl or has_pbs:
            return "PBS_EXCEL"
        if pdf and not excel:
            return "PDF_ONLY"
        return "PBS_EXCEL"
