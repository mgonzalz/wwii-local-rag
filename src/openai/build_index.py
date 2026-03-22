import os, json, re, time
from pathlib import Path

import numpy as np
import faiss

from openai import OpenAI, RateLimitError, APIError, APITimeoutError

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

INPUT_JSONLS = [
    ".cache/data/wiki/wikipedia_ww2_seed_es.jsonl",
    ".cache/data/books/La-Segunda-Guerra-Mundial-Winston-Churchill.chapters.es.jsonl",
    ".cache/data/books/Manuel-J.-Prieto.-Operaciones-especiales-de-la-SGM.chapters.es.jsonl",
]

OUT_DIR = Path(".cache/rag_store")
OUT_DIR.mkdir(exist_ok=True, parents=True)

EMBED_MODEL = "text-embedding-3-small"

MAX_CHARS = 1800
OVERLAP = 200
BATCH = 16
SLEEP_BETWEEN = 0.25


def clean_text(t: str) -> str:
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk_text(text: str, max_chars=MAX_CHARS, overlap=OVERLAP):
    text = text.strip()
    if not text:
        return []
    out = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = max(0, j - overlap)
    return [c for c in out if c]


def load_records(paths):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def make_citation(rec):
    # Wikipedia: usa url si existe; Libros: referencia por páginas
    if rec.get("source") == "wikipedia" and rec.get("url"):
        return f"{rec.get('title','Wikipedia')} — {rec['url']}"
    if rec.get("source") == "book":
        return f"{rec.get('book_title','Libro')}, {rec.get('chapter','')}, pp. {rec.get('page_start')}-{rec.get('page_end')}"
    return "Fuente desconocida"


def make_key(meta: dict) -> str:
    # Clave estable para evitar duplicados
    base_id = meta.get("id") or f"{meta.get('source')}:{meta.get('requested_title') or meta.get('title')}"
    rev = meta.get("revision_id") or meta.get("revid") or ""
    return f"{base_id}::rev={rev}::chunk={meta.get('chunk_index')}"


def embed_texts(texts, max_retries=8) -> np.ndarray:
    """
    Embeddings con reintentos y backoff exponencial para evitar RateLimitError (429).
    """
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return vecs

        except RateLimitError:
            # Backoff exponencial con tope
            wait = min((2 ** attempt) * 0.5, 10.0)  # 0.5s,1s,2s,4s,... hasta 10s
            time.sleep(wait)

        except (APITimeoutError, APIError):
            # Errores transitorios
            wait = min((2 ** attempt) * 0.5, 10.0)
            time.sleep(wait)

    raise RuntimeError("No se pudo obtener embeddings tras varios reintentos.")


def main():
    index_path = OUT_DIR / "index.faiss"
    docs_path = OUT_DIR / "docs.json"

    # 1) Cargar existente (si existe)
    if index_path.exists() and docs_path.exists():
        index = faiss.read_index(str(index_path))
        with open(docs_path, "r", encoding="utf-8") as f:
            docs_prev = json.load(f)
        seen = set(make_key(d["meta"]) for d in docs_prev)
        print(f"[OK] Índice existente cargado. Vectores: {index.ntotal} | Docs: {len(docs_prev)}")
    else:
        index = None
        docs_prev = []
        seen = set()
        print("[OK] No existe índice previo. Se creará uno nuevo.")

    # 2) Preparar SOLO docs nuevos
    new_docs = []
    for rec in load_records(INPUT_JSONLS):
        base = {k: rec.get(k) for k in rec.keys() if k != "content"}
        content = clean_text(rec.get("content", ""))

        for idx, ch in enumerate(chunk_text(content)):
            meta = {
                **base,
                "chunk_index": idx,
                "citation": make_citation(rec),
            }
            key = make_key(meta)
            if key in seen:
                continue
            seen.add(key)
            new_docs.append({"text": ch, "meta": meta})

    if not new_docs:
        print("[OK] No hay nuevos chunks. Nada que añadir.")
        return

    # 3) Embeddings solo de lo nuevo (con throttling + retry)
    vectors_list = []
    total = len(new_docs)
    for i in range(0, total, BATCH):
        batch = new_docs[i : i + BATCH]
        vecs = embed_texts([d["text"] for d in batch])
        vectors_list.append(vecs)

        done = min(i + BATCH, total)
        print(f"Embedded NEW {done}/{total}")

        # Evita picos de tokens/minuto
        time.sleep(SLEEP_BETWEEN)

    vectors = np.vstack(vectors_list).astype(np.float32)

    # 4) Crear índice si no existía y añadir
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]

    if index is None:
        index = faiss.IndexFlatIP(dim)

    index.add(vectors)

    # 5) Guardar de vuelta (append)
    docs_all = docs_prev + new_docs
    faiss.write_index(index, str(index_path))
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs_all, f, ensure_ascii=False)

    print(f"[OK] Añadidos {len(new_docs)} chunks. Total vectores: {index.ntotal} | Total docs: {len(docs_all)}")


if __name__ == "__main__":
    main()
