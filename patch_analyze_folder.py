from __future__ import annotations

import re
from pathlib import Path


TARGET = Path("core/use_case/analyze_folder.py")


def must_find(pattern: str, text: str, what: str) -> re.Match:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Pattern non trovato per: {what}")
    return m


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"File non trovato: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")
    text = original

    # ------------------------------------------------------------
    # 1) typing import: aggiungi Callable, Any se manca
    # ------------------------------------------------------------
    # Cerca: from typing import ...
    m = must_find(r"^from typing import .*$", text, "typing import line")
    line = m.group(0)

    if "Callable" not in line or "Any" not in line:
        # aggiunge in coda, mantenendo il resto invariato
        if line.endswith(")"):
            # non previsto qui, ma safe
            pass
        else:
            # aggiungi solo se non ci sono già
            parts = [p.strip() for p in line.replace("from typing import", "").split(",")]
            parts = [p for p in parts if p]
            if "Callable" not in parts:
                parts.append("Callable")
            if "Any" not in parts:
                parts.append("Any")
            newline = "from typing import " + ", ".join(parts)
            text = text.replace(line, newline, 1)

    # ------------------------------------------------------------
    # 2) Sostituisci il blocco _log/_prog sbagliato (scope) con uno safe
    # ------------------------------------------------------------
    bad_block_pattern = r"""
^def\s+_log\(level:\s*str,\s*msg:\s*str\)\s*->\s*None:\s*\n
(?:\s+.*\n)+?
^def\s+_prog\(done:\s*int,\s*total:\s*int,\s*msg:\s*str\)\s*->\s*None:\s*\n
(?:\s+.*\n)+?
(?=^\s*class\s+AnalyzeFolderUseCase:|\Z)
"""
    # Il tuo blocco attuale usa log_cb/progress_cb non definiti a livello modulo.
    # Lo rimpiazziamo con una versione safe e riusabile.
    safe_block = """# -------------------------
# ✅ NEW: helpers log/progress (safe, riusabili)
# -------------------------
LogCb = Callable[[str, str], Any]
ProgressCb = Callable[[int, int, str], Any]


def _log(level: str, msg: str, log_cb: Path | None = None) -> None:
    # NOTA: questo stub è mantenuto solo per compatibilità, verrà sovrascritto subito sotto.
    pass


def _log(level: str, msg: str, log_cb: LogCb | None = None) -> None:
    \"""
    Log helper: non fa nulla se log_cb è None.
    \"""
    if log_cb is not None:
        try:
            log_cb(level, msg)
        except Exception:
            # mai far crashare la pipeline per un logger GUI
            pass


def _prog(done: int, total: int, msg: str, progress_cb: ProgressCb | None = None) -> None:
    \"""
    Progress helper: non fa nulla se progress_cb è None.
    \"""
    if progress_cb is not None:
        try:
            progress_cb(done, total, msg)
        except Exception:
            pass

"""

    # Usiamo DOTALL + MULTILINE con una regex robusta
    m_bad = re.search(bad_block_pattern, text, flags=re.MULTILINE | re.DOTALL | re.VERBOSE)
    if m_bad:
        text = text[: m_bad.start()] + safe_block + text[m_bad.end() :]
    else:
        # Se non troviamo il blocco, inseriamo il safe_block prima della class (fallback)
        m_class = must_find(r"^class\s+AnalyzeFolderUseCase:", text, "class AnalyzeFolderUseCase")
        text = text[: m_class.start()] + safe_block + text[m_class.start() :]

    # ------------------------------------------------------------
    # 3) Inserisci wrapper LOG/PROG + prime progress call all'inizio di run()
    # ------------------------------------------------------------
    # Rimpiazza l'inizio del run:
    # base_dir = ...
    # discovery = discover_folder(base_dir)
    # result = AnalyzeFolderResult(...)
    run_head_pattern = r"""
(\s+def\s+run\(self,\s*folder:\s*str\s*\|\s*Path,\s*progress_cb=None,\s*log_cb=None\)\s*->\s*AnalyzeFolderResult:\s*\n)
(\s+base_dir\s*=\s*Path\(folder\)\.expanduser\(\)\.resolve\(\)\s*\n)
(\s+discovery\s*=\s*discover_folder\(base_dir\)\s*\n)
(\s*\n\s*result\s*=\s*AnalyzeFolderResult\(base_dir=base_dir,\s*discovery=discovery\)\s*\n)
"""
    m_run = re.search(run_head_pattern, text, flags=re.MULTILINE | re.DOTALL | re.VERBOSE)
    if not m_run:
        raise RuntimeError("Non trovo l'inizio di run() con la struttura attesa.")

    head = m_run.group(1)
    base_dir_line = m_run.group(2)
    discovery_line = m_run.group(3)
    result_line = m_run.group(4)

    injected = (
        head
        + base_dir_line
        + """
        # wrappers locali (non rompono altri utilizzi)
        def LOG(level: str, msg: str) -> None:
            _log(level, msg, log_cb=log_cb)

        def PROG(done: int, total: int, msg: str) -> None:
            _prog(done, total, msg, progress_cb=progress_cb)

        PROG(0, 100, "Discovery cartella…")
        LOG("INFO", f"AnalyzeFolderUseCase.run: base_dir={base_dir}")
"""
        + discovery_line
        + """
        LOG("INFO", f"Discovery completata: bom_excel={len(discovery.bom_excel)} bom_pdf={len(discovery.bom_pdf)}")
        PROG(5, 100, "Discovery completata")
"""
        + result_line
    )

    text = text[: m_run.start()] + injected + text[m_run.end() :]

    # ------------------------------------------------------------
    # 4) Inserisci progress/log per PBS select + PBS parse (2 punti chiave)
    # ------------------------------------------------------------
    # Dopo pbs_path = choose_single_pbs(discovery) aggiungiamo log/prog
    text = text.replace(
        "            pbs_path = choose_single_pbs(discovery)\n",
        "            pbs_path = choose_single_pbs(discovery)\n"
        "            LOG(\"INFO\", f\"[PBS] Trovato PBS: {pbs_path.name}\")\n"
        "            PROG(15, 100, \"PBS selezionato\")\n",
        1,
    )

    # Dopo result.pbs = load_pbs(pbs_path) aggiungiamo log/prog
    text = text.replace(
        "            result.pbs = load_pbs(pbs_path)\n",
        "            PROG(18, 100, \"Parsing PBS…\")\n"
        "            result.pbs = load_pbs(pbs_path)\n"
        "            LOG(\"INFO\", f\"[PBS] Parse OK: rows={len(result.pbs.rows) if result.pbs else 0}\")\n"
        "            PROG(22, 100, \"PBS parsato\")\n",
        1,
    )

    # ------------------------------------------------------------
    # 5) Inserisci progress/log parsing BOM per-file (la parte che serve in GUI)
    # ------------------------------------------------------------
    # Dopo bom_pdf_to_open = ... aggiungiamo total_to_parse + done
    anchor = "        bom_pdf_to_open = [p for p in bom_paths_to_open if p.suffix.lower() == \".pdf\"]\n"
    if anchor in text:
        text = text.replace(
            anchor,
            anchor
            + "\n"
            + "        total_to_parse = len(bom_excel_to_open) + len(bom_pdf_to_open)\n"
              "        done_parse = 0\n"
              "        LOG(\"INFO\", f\"[BOM_PARSE] Da aprire: total={total_to_parse} (xls={len(bom_excel_to_open)} pdf={len(bom_pdf_to_open)})\")\n"
              "        PROG(25, 100, f\"Parsing BOM… (0/{total_to_parse})\")\n\n",
            1,
        )
    else:
        raise RuntimeError("Anchor bom_pdf_to_open non trovato (serve per inserire progress parsing).")

    # Dentro loop Excel: subito dopo "for xls_path in bom_excel_to_open:" inseriamo update
    text = text.replace(
        "            for xls_path in bom_excel_to_open:\n",
        "            for xls_path in bom_excel_to_open:\n"
        "                done_parse += 1\n"
        "                base = 25\n"
        "                span = 45\n"
        "                pct = base + (0 if total_to_parse == 0 else int(span * done_parse / total_to_parse))\n"
        "                PROG(pct, 100, f\"Parsing BOM {done_parse}/{total_to_parse}: {xls_path.name}\")\n"
        "                LOG(\"INFO\", f\"[BOM_XLS] Parsing: {xls_path.name}\")\n",
        1,
    )

    # Dentro loop PDF: subito dopo "for pdf_path in bom_pdf_to_open:" inseriamo update
    text = text.replace(
        "            for pdf_path in bom_pdf_to_open:\n",
        "            for pdf_path in bom_pdf_to_open:\n"
        "                done_parse += 1\n"
        "                base = 25\n"
        "                span = 45\n"
        "                pct = base + (0 if total_to_parse == 0 else int(span * done_parse / total_to_parse))\n"
        "                PROG(pct, 100, f\"Parsing BOM {done_parse}/{total_to_parse}: {pdf_path.name}\")\n"
        "                LOG(\"INFO\", f\"[BOM_PDF] Parsing: {pdf_path.name}\")\n",
        1,
    )

    # ------------------------------------------------------------
    # 6) Progress per linking + explode + done
    # ------------------------------------------------------------
    text = text.replace(
        "        # 5) linking strict PBS -> BOM\n",
        "        PROG(72, 100, \"Linking PBS → BOM…\")\n"
        "        LOG(\"INFO\", \"[LINK] Avvio linking PBS→BOM…\")\n"
        "        # 5) linking strict PBS -> BOM\n",
        1,
    )

    text = text.replace(
        "        # 6) Esplosione BOM ricorsiva + confronto con PBS\n",
        "        PROG(80, 100, \"Explode BOM…\")\n"
        "        LOG(\"INFO\", \"[EXPLODE] Avvio esplosione…\")\n"
        "        # 6) Esplosione BOM ricorsiva + confronto con PBS\n",
        1,
    )

    # Prima del return finale del metodo run, aggiungi completamento (solo 1 volta)
    # Inseriamo vicino a "return result" dell'ultima riga del run.
    # (mettiamo la progress 100 subito prima dell'ultimo "return result" dentro run)
    text = re.sub(
        r"\n(\s+return result\n)\s*\n(if __name__ == \"__main__\":)",
        "\n        PROG(100, 100, \"Analisi completata\")\n        LOG(\"INFO\", \"Analisi completata\")\n\\1\n\\2",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if text == original:
        raise RuntimeError("Nessuna modifica applicata (probabilmente hai già patchato o patterns non matchano).")

    # Backup + write
    backup = TARGET.with_suffix(TARGET.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")

    print(f"OK ✅ Patch applicata.\nBackup: {backup}\nFile aggiornato: {TARGET}")


if __name__ == "__main__":
    main()
