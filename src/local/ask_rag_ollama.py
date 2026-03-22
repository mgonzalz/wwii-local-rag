import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

STORE = ".cache/rag_store_local"
TOP_K = 6

EMB_MODEL = "intfloat/multilingual-e5-small"

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


# LOAD STORE
def load_store():
    index = faiss.read_index(f"{STORE}/index.faiss")
    with open(f"{STORE}/docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


# CITATIONS (RICH + DEDUP)
def build_citation(meta: dict, score: float) -> str:
    """
    Construye una referencia rica con capítulo/sección/libro + páginas + chunk.
    Incluye link si existe (Wikipedia).
    """
    src = meta.get("source", "")
    chunk_idx = meta.get("chunk_index", None)

    if src == "wikipedia":
        title = meta.get("title") or meta.get("requested_title") or "Wikipedia"
        url = meta.get("url", "")
        extra = f" — chunk {chunk_idx}" if chunk_idx is not None else ""
        if url:
            return f"[SCORE: {score:.3f}] {title} — {url}{extra}".strip()
        return f"[SCORE: {score:.3f}] {title}{extra}".strip()

    if src == "book":
        book = meta.get("book_title", "Libro")
        part = meta.get("book_part")
        section = meta.get("section")
        chapter = meta.get("chapter")
        p1 = meta.get("page_start")
        p2 = meta.get("page_end")

        bits = [book]
        if part:
            bits.append(part)
        if section:
            bits.append(section)
        if chapter:
            bits.append(chapter)

        where = ", ".join(bits)
        pages = f"pp. {p1}-{p2}" if p1 and p2 else ""
        extra = f"chunk {chunk_idx}" if chunk_idx is not None else ""
        tail = ", ".join([x for x in [pages, extra] if x])
        if tail:
            where = f"{where}, {tail}"
        return f"[SCORE: {score:.3f}] {where}"

    base = meta.get("citation", "Fuente desconocida")
    extra = f" — chunk {chunk_idx}" if chunk_idx is not None else ""
    return f"[SCORE: {score:.3f}] {base}{extra}"


def unique_sources_from_hits(hits):
    """
    Dedup por (fuente base + capítulo/sección + páginas + chunk)
    para que no te repita Wikipedia 3 veces sin diferenciar.
    """
    seen = set()
    out = []
    for h in hits:
        m = h["meta"]
        key = (
            m.get("source"),
            m.get("book_file")
            or m.get("url")
            or m.get("requested_title")
            or m.get("title"),
            m.get("book_part"),
            m.get("section"),
            m.get("chapter"),
            m.get("page_start"),
            m.get("page_end"),
            m.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(build_citation(m, h["score"]))
    # ya vienen ordenadas por score porque hits lo está
    return out


# CONTEXT (WITH ANCHORS)
def format_context(hits):
    """
    Contexto con anclas HIT N para que el modelo enlace ideas sin inventar.
    """
    blocks = []
    for rank, h in enumerate(hits, start=1):
        cite = build_citation(h["meta"], h["score"])
        chunk_id = h["meta"].get("id") or h["meta"].get("chunk_index") or rank
        blocks.append(
            f"[HIT {rank}] [CHUNK_ID: {chunk_id}] {cite}\n" f"[TEXTO]\n{h['text']}"
        )
    return "\n\n---\n\n".join(blocks)


# OLLAMA
def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.18,
            "num_predict": 420,
            # Cortamos si intenta imprimir secciones de fuentes
            "stop": [
                "\nFuentes:",
                "\nFUENTES:",
                "\nReferencias:",
                "\nREFERENCIAS:",
                "\nBibliografía:",
                "\nBIBLIOGRAFÍA:",
                "\nNotas:",
                "\nNOTAS:",
            ],
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]

def evidence_strength(hits, min_score=0.80):
    strong = [h for h in hits if h["score"] >= min_score]
    if len(strong) >= 3:
        return "strong"
    if len(strong) >= 2:
        return "medium"
    return "weak"


def main():
    embedder = SentenceTransformer(EMB_MODEL)
    index, docs = load_store()

    # Prompt: “como ChatGPT”, sin apartados, sin “en las fuentes cargadas”
    system_prompt = """Eres un asistente RAG especializado en la Segunda Guerra Mundial.
    Reglas obligatorias:
    1) Usa ÚNICAMENTE la información del bloque CONTEXTO. No uses conocimiento externo. Nada de acceder a tu memoria.
    2) Si el CONTEXTO no contiene datos directos para responder, NO inventes.
    En ese caso:
    - di claramente que “no está en las fuentes actuales”,
    - explica qué sí aparece (si hay contexto relacionado),
    - y sugiere qué fuente faltaría añadir (p. ej. la página de Wikipedia correspondiente).
    3) Si el CONTEXTO contiene información parcial, responde SOLO con esa parte y márcalo como “parcial”.
    4) Incluye siempre una sección final “Fuentes” con las citas de los fragmentos utilizados.
    5) Si la pregunta es ambigua, haz UNA pregunta de aclaración breve, pero solo si el CONTEXTO no permite decidir.

    Formato de salida:
    - Respuesta (3–8 frases, concisa)
    - Fuentes: lista de citas usadas (sin inventar)
    - Cobertura: {completa | parcial | no en fuentes}"""

    while True:
        question = input("\nPregunta (enter para salir): ").strip()
        if not question:
            break

        # Embedding query (E5)
        q_emb = embedder.encode(
            [f"query: {question}"], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, ids = index.search(q_emb, TOP_K)

        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            d = docs[idx]
            hits.append(
                {
                    "text": d["text"],
                    "meta": d["meta"],
                    "score": float(score),
                }
            )

        hits.sort(key=lambda x: x["score"], reverse=True)
        strength = evidence_strength(hits, min_score=0.80)

        if strength == "weak":
            extra_rule = (
                "INSTRUCCIÓN EXTRA: La evidencia es limitada. "
                "No completes huecos. Responde en 2-4 frases máximo, "
                "limitándote a lo explícito."
            )
        elif strength == "medium":
            extra_rule = (
                "INSTRUCCIÓN EXTRA: Hay evidencia suficiente. "
                "Responde en 8-12 frases (120-180 palabras aprox.), "
                "en un único texto fluido, conectando 2-3 hechos del contexto. "
                "No inventes datos que no aparezcan."
            )
        else:  # strong
            extra_rule = (
                "INSTRUCCIÓN EXTRA: Hay evidencia fuerte. "
                "Responde en 10-14 frases (150-220 palabras aprox.), "
                "en un único texto fluido. "
                "Incluye: (1) quién fue, (2) cómo aparece en los eventos del contexto, "
                "(3) 2-4 acciones/decisiones mencionadas en los HITs. "
                "No inventes."
            )


        context = format_context(hits)

        # Instrucción adicional para que use HITs como anclas sin “justificación”
        prompt = (
            f"{system_prompt}\n"
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA:\n{question}\n\n"
            f"INSTRUCCIÓN: Responde en un único bloque de texto fluido. "
            f"{extra_rule} "
            f"Puedes conectar ideas apoyándote en los HITs, pero no los menciones explícitamente.\n"
        )

        answer = ollama_generate(prompt).strip()

        # Fuentes impresas por el sistema (no por el LLM)
        sources = unique_sources_from_hits(hits)

        print("\n" + answer)
        print("\nFuentes:")
        for s in sources:
            print(f"- {s}")


if __name__ == "__main__":
    main()
