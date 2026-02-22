# core/parsers/excel_session.py
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, TypeVar

import pythoncom
import pywintypes
import win32com.client


T = TypeVar("T")

# HRESULT tipici in automazione Office quando l'istanza muore o si disconnette
RPC_E_DISCONNECTED = 0x80010108
RPC_S_SERVER_UNAVAILABLE = 0x800706BA

def _hresult(exc: Exception) -> Optional[int]:
    if isinstance(exc, pywintypes.com_error):
        # exc.args spesso: (hresult, text, exc, arg)
        try:
            return int(exc.args[0]) & 0xFFFFFFFF
        except Exception:
            return None
    return None


def _apply_excel_safe_flags(excel) -> None:
    # coerente con quello che già fai, con 2 aggiunte utili
    for attr, val in (
        ("Visible", False),
        ("DisplayAlerts", False),
        ("ScreenUpdating", False),
        ("EnableEvents", False),
        ("AskToUpdateLinks", False),
    ):
        try:
            setattr(excel, attr, val)
        except Exception:
            pass

    # Calculation manual
    try:
        excel.Calculation = -4135  # xlCalculationManual
    except Exception:
        pass

    # Disabilita macro (msoAutomationSecurityForceDisable = 3)
    try:
        excel.AutomationSecurity = 3
    except Exception:
        pass

    # Riduce “dialoghi” e interazioni
    try:
        excel.Interactive = False
    except Exception:
        pass


@dataclass
class ExcelOpenOptions:
    copy_to_local: bool = False
    keep_local_copy: bool = False  # utile in SAFE_MODE per ispezione
    retries: int = 2
    retry_sleep_s: float = 0.4


class ExcelSession:
    """
    Wrapper robusto:
    - 1 istanza Excel (DispatchEx)
    - open workbook con retry + restart su RPC_E_DISCONNECTED
    - opzionale copia su locale prima di aprire
    """

    def __init__(self, restart_every_n: int = 0):
        self._excel = None
        self._opened_count = 0
        self._restart_every_n = restart_every_n

    @property
    def excel(self):
        if self._excel is None:
            self.start()
        return self._excel

    def start(self) -> None:
        pythoncom.CoInitialize()
        self._excel = win32com.client.DispatchEx("Excel.Application")
        _apply_excel_safe_flags(self._excel)

    def quit(self) -> None:
        if self._excel is None:
            return
        try:
            self._excel.Quit()
        except Exception:
            pass
        self._excel = None

    def restart(self) -> None:
        self.quit()
        self.start()

    def _maybe_periodic_restart(self) -> None:
        if self._restart_every_n and self._opened_count >= self._restart_every_n:
            self.restart()
            self._opened_count = 0

    def _open_workbook_once(self, path: str):
        # opzioni “safe” simili a quelle che hai già
        try:
            return self.excel.Workbooks.Open(
                str(path),
                UpdateLinks=0,
                ReadOnly=True,
                IgnoreReadOnlyRecommended=True,
                AddToMru=False,
            )
        except Exception:
            return self.excel.Workbooks.Open(str(path), ReadOnly=True)

    def open_workbook(self, path: Path, opt: ExcelOpenOptions) -> tuple[object, Path]:
        self._maybe_periodic_restart()

        local_path = path
        tmp_dir: Optional[str] = None

        if opt.copy_to_local:
            tmp_dir = tempfile.mkdtemp(prefix="pbs_bom_excel_")
            dst = Path(tmp_dir) / path.name
            shutil.copy2(path, dst)
            local_path = dst

        last_exc: Optional[Exception] = None
        for attempt in range(0, opt.retries + 1):
            try:
                wb = self._open_workbook_once(str(local_path))
                self._opened_count += 1
                return wb, local_path
            except Exception as e:
                last_exc = e
                hr = _hresult(e)
                # Se Excel è morto/disconnesso → restart e retry
                if hr in (RPC_E_DISCONNECTED, RPC_S_SERVER_UNAVAILABLE):
                    self.restart()
                    time.sleep(opt.retry_sleep_s)
                    continue
                # per altri errori, non “mascherare”: rilancia
                raise

        # exhausted retries
        if last_exc:
            raise last_exc
        raise RuntimeError("open_workbook: failure without exception (?)")

    def close_workbook(self, wb) -> None:
        try:
            wb.Close(SaveChanges=False)
        except Exception:
            pass

    def call(self, fn: Callable[[], T], retries: int = 1, sleep_s: float = 0.2) -> T:
        """
        Helper per invocare piccole operazioni COM con retry+restart
        (utile per ws.UsedRange, ws.Range(...).Value, ecc.)
        """
        last_exc: Optional[Exception] = None
        for _ in range(retries + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                hr = _hresult(e)
                if hr in (RPC_E_DISCONNECTED, RPC_S_SERVER_UNAVAILABLE):
                    self.restart()
                    time.sleep(sleep_s)
                    continue
                raise
        assert last_exc is not None
        raise last_exc
