from __future__ import annotations

import sys
from pathlib import Path

from core.parsers.discovery import discover_folder, choose_single_pbs
from core.parsers.bom_excel import parse_bom_excel_raw
from core.parsers.bom_pdf import parse_bom_pdf_raw
from core.services.bom_normalizer import build_bom_document

# opzionale: usa Excel singleton se disponibile (stesso approccio del tuo analyze_folder)
def _try_create_excel_app():
    try:
        import win32com.client  # type: ignore
    except Exception:
        return None

    excel = win32com.client.DispatchEx("Excel.Application")
    # safe flags
    for attr, val in [
        ("Visible", False),
        ("DisplayAlerts", False),
        ("ScreenUpdating", False),
        ("EnableEvents", False),
    ]:
        try:
            setattr(excel, attr, val)
        except Exception:
            pass
    try:
        excel.Calculation = -4135  # xlCalculationManual
    except Exception:
        pass
    try:
        excel.AskToUpdateLinks = False
    except Exception:
        pass
    return excel


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/find_code_in_boms.py <MDP_FOLDER> <CODE_TO_FIND>")
        print('Example: python tools/find_code_in_boms.py \\\\fs01\\Elthub\\JOB\\...\\LCTE MDP 231018117ASSY')
        sys.exit(2)

    base_dir = Path(sys.argv[1]).expanduser().resolve()
    needle = sys.argv[2].strip().upper()

    discovery = discover_folder(base_dir)
    all_bom_paths = list(discovery.bom_excel) + list(discovery.bom_pdf)

    print(f"Scanning BOM files: {len(all_bom_paths)}")
    print(f"Searching for: {needle}\n")

    excel = _try_create_excel_app()
    hits = 0

    try:
        for p in all_bom_paths:
            try:
                if p.suffix.lower() in (".xls", ".xlsx", ".xlsm"):
                    raw = parse_bom_excel_raw(p, excel=excel)  # works if you added excel= support
                elif p.suffix.lower() == ".pdf":
                    raw = parse_bom_pdf_raw(p)
                else:
                    continue

                bom = build_bom_document(
                    path=p,
                    header_code=raw.get("header", {}).get("code", ""),
                    header_rev=raw.get("header", {}).get("rev", ""),
                    header_title=raw.get("header", {}).get("title", ""),
                    doc_date_iso=raw.get("header", {}).get("date", ""),
                    raw_lines=raw.get("lines", []),
                )

                hdr = getattr(bom, "header", None)
                parent_code = getattr(hdr, "code", "") if hdr else ""
                parent_rev = getattr(hdr, "revision", "") if hdr else ""

                for line in getattr(bom, "lines", []) or []:
                    code = (getattr(line, "internal_code", "") or "").strip().upper()
                    if code == needle:
                        pos = getattr(line, "pos", "")
                        desc = getattr(line, "description", "")
                        print(f"HIT: {p.name}")
                        print(f"  BOM header: {parent_code} REV {parent_rev}")
                        print(f"  POS={pos}  CODE={code}")
                        if desc:
                            print(f"  DESC={desc}")
                        print()
                        hits += 1

            except TypeError:
                # se non hai ancora la firma excel= nel parser, ripiega
                raw = parse_bom_excel_raw(p)  # type: ignore
            except Exception as e:
                # non fermarti per un file problematico
                print(f"[WARN] Failed {p.name}: {e}")

    finally:
        if excel is not None:
            try:
                excel.Quit()
            except Exception:
                pass

    print(f"Done. Total hits: {hits}")


if __name__ == "__main__":
    main()
