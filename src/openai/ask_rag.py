import os, json
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

STORE = ".cache/rag_store"
TOP_K = 30
MIN_SCORE_TO_CITE = 0.50
MIN_GOOD_HITS = 2
NO_INFO_MSG = "No se dispone de información en las fuentes consultadas para responder a esta pregunta."


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
    # Contexto limpio: SOLO texto recuperado (sin scores ni etiquetas)
    blocks = []
    for h in hits:
        blocks.append(h["text"])
    return "\n\n---\n\n".join(blocks)


def format_sources(hits):
    # Fuentes con score + cita + chunk
    lines = []
    for h in hits:
        meta = h["meta"]
        citation = meta.get("citation", "—")
        chunk = meta.get("chunk_index", "?")
        lines.append(f"- [SCORE: {h.get('_score', 0):.3f}] {citation} — chunk {chunk}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""
Eres un asistente experto en la Segunda Guerra Mundial.

REGLAS ABSOLUTAS:
1) Responde SIEMPRE en español.
2) Responde ÚNICAMENTE usando información que aparezca EXPLÍCITAMENTE en el CONTEXTO.
3) NO uses conocimiento externo. NO consultes internet.
4) Si el CONTEXTO no contiene evidencia suficiente para responder, responde EXACTAMENTE:
   "{NO_INFO_MSG}"
5) Si hay evidencia suficiente, responde de forma clara y completa, pero NO incluyas una sección de fuentes
   (las fuentes se añadirán fuera del modelo).
""".strip()


def main():
    index, docs = load_store()

    while True:
        q = input("\nPregunta: ").strip()
        if not q:
            break

        qv = embed_query(q)
        scores, ids = index.search(qv, TOP_K)

        hits = []
        for score, i in zip(scores[0], ids[0]):
            if int(i) == -1:
                continue
            hit = dict(docs[int(i)])
            hit["_score"] = float(score)
            hits.append(hit)

        # Gating por score
        good_hits = [h for h in hits if h.get("_score", 0.0) >= MIN_SCORE_TO_CITE]
        if len(good_hits) < MIN_GOOD_HITS:
            print("\n" + NO_INFO_MSG)
            continue

        context = format_context(good_hits)
        sources_block = format_sources(good_hits)

        user = (
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA:\n{q}\n\n"
            f"INSTRUCCIÓN: Si no hay evidencia, responde exactamente: {NO_INFO_MSG}"
        )

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

        answer = resp.choices[0].message.content.strip()

        # Si el modelo dice que no hay info → NO mostrar fuentes
        if answer == NO_INFO_MSG:
            print("\n" + answer)
            continue

        # Si sí responde → añadimos fuentes nosotros
        print("\n" + answer)
        print("\nFUENTES:\n" + sources_block)

if __name__ == "__main__":
    main()
