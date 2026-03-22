import json, re, os
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INPUT_JSONLS = [
    ".cache/data/wiki/wikipedia_ww2_seed_es.jsonl",
    ".cache/data/books/La-Segunda-Guerra-Mundial-Winston-Churchill.chapters.es.jsonl",
    ".cache/data/books/Manuel-J.-Prieto.-Operaciones-especiales-de-la-SGM.chapters.es.jsonl",
]

OUT_DIR = Path(".cache/rag_store_local")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "intfloat/multilingual-e5-small"

MAX_CHARS = 1800
OVERLAP = 200
BATCH = 64


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
    if rec.get("source") == "wikipedia" and rec.get("url"):
        return f"{rec.get('title','Wikipedia')} — {rec['url']}"
    if rec.get("source") == "book":
        return f"{rec.get('book_title','Libro')}, {rec.get('chapter','')}, pp. {rec.get('page_start')}-{rec.get('page_end')}"
    return "Fuente desconocida"


def make_key(meta: dict) -> str:
    base_id = meta.get("id") or f"{meta.get('source')}:{meta.get('requested_title') or meta.get('title')}"
    rev = meta.get("revision_id") or meta.get("revid") or ""
    return f"{base_id}::rev={rev}::chunk={meta.get('chunk_index')}"

def main():
    model = SentenceTransformer(EMB_MODEL)

    index_path = OUT_DIR / "index.faiss"
    docs_path  = OUT_DIR / "docs.json"

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

    new_docs = []
    new_texts = []

    for rec in load_records(INPUT_JSONLS):
        base = {k: rec.get(k) for k in rec.keys() if k != "content"}
        content = clean_text(rec.get("content", ""))

        for idx, ch in enumerate(chunk_text(content)):
            meta = {**base, "chunk_index": idx, "citation": make_citation(rec)}
            key = make_key(meta)
            if key in seen:
                continue
            seen.add(key)
            new_docs.append({"text": ch, "meta": meta})
            new_texts.append(ch)

    if not new_docs:
        print("[OK] No hay nuevos chunks. Nada que añadir.")
        return

    passages = [f"passage: {t}" for t in new_texts]

    vectors = []
    for i in range(0, len(passages), BATCH):
        batch = passages[i : i + BATCH]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        vectors.append(emb.astype(np.float32))
        print(f"Embedded NEW {min(i+BATCH, len(passages))}/{len(passages)}")

    vectors = np.vstack(vectors)
    dim = vectors.shape[1]

    if index is None:
        index = faiss.IndexFlatIP(dim)

    index.add(vectors)

    docs_all = docs_prev + new_docs
    faiss.write_index(index, str(index_path))
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs_all, f, ensure_ascii=False)

    print(f"[OK] Añadidos {len(new_docs)} chunks. Total vectores: {index.ntotal} | Total docs: {len(docs_all)}")

if __name__ == "__main__":
    main()
