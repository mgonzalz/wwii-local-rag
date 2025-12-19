import os, json
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

STORE = ".cache/rag_store"
TOP_K = 6


def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(v)
    return v


def load_store():
    index = faiss.read_index(f"{STORE}/index.faiss")
    with open(f"{STORE}/docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


def format_context(hits):
    blocks = []
    for h in hits:
        meta = h["meta"]
        blocks.append(
            f"[SCORE] {h.get('_score', 0):.4f}\n"
            f"[CITA] {meta.get('citation')}\n"
            f"[TEXTO]\n{h['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def format_sources(hits):
    lines = []
    for h in hits:
        meta = h["meta"]
        citation = meta.get("citation", "—")
        chunk = meta.get("chunk_index", "?")
        lines.append(f"- [SCORE: {h.get('_score', 0):.3f}] {citation} — chunk {chunk}")
    return "\n".join(lines)


def main():
    index, docs = load_store()

    while True:
        q = input("\nPregunta (enter para salir): ").strip()
        if not q:
            break

        qv = embed_query(q)
        scores, ids = index.search(qv, TOP_K)

        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            hit = docs[i]
            hit["_score"] = float(score)
            hits.append(hit)
        context = format_context(hits)
        sources_block = format_sources(hits)

        system = (
            "Eres un asistente experto en la Segunda Guerra Mundial. "
            "Responde SIEMPRE en español. "
            "Usa SOLO la información del CONTEXTO proporcionado. "
            "Si no está en el contexto, di que no aparece en las fuentes. "
            "Al final incluye una sección 'Fuentes' y pega EXACTAMENTE (sin cambiar nada) "
            "la lista de líneas que te doy en 'FUENTES_PREPARADAS'. "
            "No inventes fuentes nuevas ni modifiques scores."
        )

        user = (
            f"CONTEXTO:\n{context}\n\n"
            f"FUENTES_PREPARADAS:\n{sources_block}\n\n"
            f"PREGUNTA:\n{q}\n\n"
            "INSTRUCCIÓN: En la sección 'Fuentes' debes copiar exactamente FUENTES_PREPARADAS."
        )

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

        answer = resp.choices[0].message.content
        print("\n" + answer)


if __name__ == "__main__":
    main()
