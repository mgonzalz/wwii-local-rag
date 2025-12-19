import os, json, re
from pathlib import Path
import numpy as np
import faiss

from openai import OpenAI

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
    # Wikipedia: usa url si existe; Libros: referencia por páginas
    if rec.get("source") == "wikipedia" and rec.get("url"):
        return f"{rec.get('title','Wikipedia')} — {rec['url']}"
    if rec.get("source") == "book":
        return f"{rec.get('book_title','Libro')}, {rec.get('chapter','')}, pp. {rec.get('page_start')}-{rec.get('page_end')}"
    return "Fuente desconocida"


def embed_texts(texts):
    # Embeddings API
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs


def main():
    docs = []
    for rec in load_records(INPUT_JSONLS):
        base = {k: rec.get(k) for k in rec.keys() if k != "content"}
        content = clean_text(rec.get("content", ""))

        # CHUNKING para recuperación
        for idx, ch in enumerate(chunk_text(content)):
            docs.append(
                {
                    "text": ch,
                    "meta": {
                        **base,
                        "chunk_index": idx,
                        "citation": make_citation(rec),
                    },
                }
            )

    # Embeddings por lotes
    vectors = []
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]
        vecs = embed_texts([d["text"] for d in batch])
        vectors.append(vecs)
        print(f"Embedded {min(i+BATCH, len(docs))}/{len(docs)}")
    vectors = np.vstack(vectors)

    # Crear índice FAISS (cosine via normalización + inner product)
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Guardar
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    with open(OUT_DIR / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print("[OK] Índice creado en .cache/rag_store/")


if __name__ == "__main__":
    main()
