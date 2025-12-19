import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter

import fitz


BOOKS_DIR = Path(".cache/books")
OUT_DIR = Path(".cache/data/books")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROMAN_RE = (
    r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)"
)

# Churchill: "LIBRO III" / "Capítulo I"
RE_CH_LIBRO = re.compile(rf"^\s*LIBRO\s+({ROMAN_RE})\s*$", re.IGNORECASE)
RE_CH_CAP = re.compile(rf"^\s*Cap[ií]tulo\s+({ROMAN_RE}|\d+)\s*$", re.IGNORECASE)

# Prieto: "1. TÍTULO..." (pero evitando "230. En esa época..." => exigimos que el "título" sea MAYÚSCULAS)
RE_PR_CAP = re.compile(r"^\s*(\d{1,3})\.\s+(.+?)\s*$")

# Intro
RE_INTRO = re.compile(r"^\s*Introducci[oó]n\s*$", re.IGNORECASE)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:24]


def clean_text_basic(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)  # notas [2]
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_mostly_upper(s: str, threshold: float = 0.75) -> bool:
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 6:
        return False
    upp = sum(1 for c in letters if c.isupper())
    return (upp / len(letters)) >= threshold


def guess_book_title(pdf_path: Path) -> str:
    return pdf_path.stem.replace("_", " ").replace("-", " ").strip()


def extract_pages_text(pdf_path: Path) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for p in range(len(doc)):
        txt = doc.load_page(p).get_text("text")
        pages.append(txt)
    doc.close()
    return pages


def strip_headers_footers(pages: List[str]) -> List[str]:
    """
    Quita líneas que se repiten en muchas páginas (cabeceras/pies típicos).
    Heurística: si una línea aparece en >= 30% de páginas, la eliminamos.
    """
    norm_pages_lines: List[List[str]] = []
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        norm_pages_lines.append(lines)

    cnt = Counter()
    for lines in norm_pages_lines:
        # contamos solo primeras 3 y últimas 3 líneas como candidatos a header/footer
        for ln in lines[:3] + lines[-3:]:
            cnt[ln] += 1

    min_occ = max(3, int(0.30 * len(pages)))
    bad = {ln for ln, c in cnt.items() if c >= min_occ}

    cleaned_pages = []
    for lines in norm_pages_lines:
        kept = [ln for ln in lines if ln not in bad]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def join_wrapped_title(lines: List[str], start_idx: int, max_lines: int = 3) -> str:
    """
    Une títulos que están en mayúsculas pero partidos por salto de línea.
    Toma hasta max_lines líneas si son "tipo título" (mayúsculas o casi).
    """
    parts = []
    for k in range(max_lines):
        i = start_idx + k
        if i >= len(lines):
            break
        ln = lines[i].strip()
        if not ln:
            break
        # acepta mayúsculas o muy mayúsculas
        if is_mostly_upper(ln, 0.70) or ln.isupper():
            parts.append(ln)
        else:
            break
    return " ".join(parts).strip()


def detect_chapter_starts(
    pages: List[str], book_kind: str
) -> List[Tuple[int, int, str, Optional[str]]]:
    """
    Devuelve lista de (page_idx, line_idx, chapter_label, book_part)
    book_kind: "churchill" | "prieto"
    """
    starts = []
    current_book_part = None

    for pi, ptxt in enumerate(pages):
        lines = [ln.strip() for ln in ptxt.splitlines() if ln.strip()]
        for li, ln in enumerate(lines):
            # Intro (común)
            if RE_INTRO.match(ln):
                starts.append((pi, li, "Introducción", current_book_part))
                continue

            if book_kind == "churchill":
                m = RE_CH_LIBRO.match(ln)
                if m:
                    current_book_part = f"LIBRO {m.group(1)}"
                    continue

                m = RE_CH_CAP.match(ln)
                if m:
                    chap_num = m.group(1)
                    # Intentamos capturar título real del capítulo en las siguientes líneas (si viene)
                    title = join_wrapped_title(lines, li + 1, max_lines=3)
                    chapter_label = f"Capítulo {chap_num}".strip()
                    if title:
                        chapter_label = f"{chapter_label} — {title}"
                    starts.append((pi, li, chapter_label, current_book_part))
                    continue

            elif book_kind == "prieto":
                m = RE_PR_CAP.match(ln)
                if m:
                    num = m.group(1)
                    possible_title = m.group(2).strip()

                    # Prieto: SOLO aceptamos como "capítulo" si el título es mayormente mayúsculas
                    # Esto elimina falsos positivos tipo "230. En esa época, ya estaban..."
                    if (
                        not is_mostly_upper(possible_title, 0.70)
                        and not possible_title.isupper()
                    ):
                        continue

                    # Además, si el título está cortado, podemos unir líneas siguientes en mayúsculas
                    # (si la línea siguiente continúa en mayúsculas)
                    cont = join_wrapped_title(lines, li + 1, max_lines=2)
                    full_title = possible_title
                    if cont:
                        full_title = f"{possible_title} {cont}".strip()

                    chapter_label = f"{num}. {full_title}"
                    starts.append((pi, li, chapter_label, None))
                    continue

    # Orden natural
    starts.sort(key=lambda x: (x[0], x[1]))
    return starts


def slice_text_between(
    pages: List[str], start: Tuple[int, int], end: Optional[Tuple[int, int]]
) -> str:
    """
    Recorta texto desde (start_page,start_line) hasta (end_page,end_line) exclusivo.
    """
    sp, sl = start
    if end is None:
        ep, el = len(pages) - 1, None
    else:
        ep, el = end

    out_lines = []
    for pi in range(sp, ep + 1):
        lines = [ln.rstrip() for ln in pages[pi].splitlines()]
        if pi == sp:
            lines = lines[sl:]  # desde start_line
        if end is not None and pi == ep:
            lines = lines[:el]  # hasta end_line (exclusivo)
        out_lines.extend(lines)

    return clean_text_basic("\n".join(out_lines))


def build_jsonl_for_pdf(pdf_path: Path, book_kind: str) -> List[Dict[str, Any]]:
    raw_pages = extract_pages_text(pdf_path)
    pages = strip_headers_footers(raw_pages)

    starts = detect_chapter_starts(pages, book_kind=book_kind)

    # Si no detecta nada, 1 registro con todo
    if not starts:
        full = clean_text_basic("\n\n".join(pages))
        return [
            {
                "id": stable_id(pdf_path.name, "FULL"),
                "source": "book",
                "lang": "es",
                "book_file": pdf_path.name,
                "book_title": guess_book_title(pdf_path),
                "book_part": None,
                "chapter": None,
                "page_start": 1,
                "page_end": len(pages),
                "ingested_at": now_utc_iso(),
                "content": full,
                "content_len": len(full),
            }
        ]

    records = []
    for i, (pi, li, chapter_label, book_part) in enumerate(starts):
        start = (pi, li)
        end = (starts[i + 1][0], starts[i + 1][1]) if i + 1 < len(starts) else None
        content = slice_text_between(pages, start, end)

        # Filtra cosas muy cortas (portadillas, etc.)
        if len(content) < 800:
            continue

        # páginas aproximadas: desde start_page hasta end_page-1
        page_start = pi + 1
        page_end = (starts[i + 1][0] if i + 1 < len(starts) else (len(pages) - 1)) + 1

        rec = {
            "id": stable_id(
                pdf_path.name, chapter_label, str(page_start), str(page_end)
            ),
            "source": "book",
            "lang": "es",
            "book_file": pdf_path.name,
            "book_title": guess_book_title(pdf_path),
            "book_part": book_part,
            "chapter": chapter_label,
            "page_start": page_start,
            "page_end": page_end,
            "ingested_at": now_utc_iso(),
            "content": content,
            "content_len": len(content),
        }
        records.append(rec)

    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    inputs = [
        (BOOKS_DIR / "La-Segunda-Guerra-Mundial-Winston-Churchill.pdf", "churchill"),
        (
            BOOKS_DIR / "Manuel-J.-Prieto.-Operaciones-especiales-de-la-SGM.pdf",
            "prieto",
        ),
    ]

    for pdf_path, kind in inputs:
        if not pdf_path.exists():
            print(f"[ERROR] Not found: {pdf_path}")
            continue

        records = build_jsonl_for_pdf(pdf_path, book_kind=kind)
        out_path = OUT_DIR / f"{pdf_path.stem}.chapters.es.jsonl"
        write_jsonl(out_path, records)

        print(f"\n[OK] {pdf_path.name}")
        print(f"   kind={kind}")
        print(f"   capítulos detectados (líneas JSONL): {len(records)}")
        print(f"   output: {out_path}\n")


if __name__ == "__main__":
    main()
