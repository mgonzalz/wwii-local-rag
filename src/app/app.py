import json
import numpy as np
import faiss
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageOps
from pathlib import Path

EMBED_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

STORE = ".cache/rag_store_local"
TOP_K = 30
MIN_SCORE_TO_CITE = 0.50
MIN_GOOD_HITS = 2

NO_INFO_MSG = "No se dispone de información en las fuentes consultadas para responder a esta pregunta."


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def embed_query(q: str) -> np.ndarray:
    embedder = load_embedder()
    v = embedder.encode(
        [f"query: {q}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return v


def load_store():
    index = faiss.read_index(f"{STORE}/index.faiss")
    with open(f"{STORE}/docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


def format_context(hits):
    return "\n\n---\n\n".join(h["text"] for h in hits)


def format_sources(hits):
    lines = []
    for h in hits:
        meta = h["meta"]
        citation = meta.get("citation", "—")
        chunk = meta.get("chunk_index", "?")
        lines.append(f"- [SCORE: {h.get('_score', 0):.3f}] {citation} — chunk {chunk}")
    return "\n".join(lines)


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
    return r.json()["response"].strip()


SYSTEM_PROMPT = f"""
Eres “Churchill IA”, un asistente experto en la Segunda Guerra Mundial.

OBJETIVO:
Responder preguntas del usuario usando ÚNICAMENTE la evidencia incluida en el CONTEXTO recuperado por el sistema RAG.

REGLAS ABSOLUTAS:
1) Idioma: responde SIEMPRE en español.

2) Alcance:
   - Si la pregunta no está relacionada con la Segunda Guerra Mundial, responde EXACTAMENTE:
     "La pregunta está fuera del alcance del sistema."
   - No añadas ningún dato adicional en ese caso.

3) Evidencia obligatoria:
   - Solo puedes afirmar hechos que aparezcan EXPLÍCITAMENTE en el CONTEXTO.
   - No uses conocimiento externo, ni completes por intuición, ni hagas suposiciones.

4) Falta de evidencia:
   - Si el CONTEXTO no contiene evidencia suficiente para responder con certeza, responde EXACTAMENTE:
     "{NO_INFO_MSG}"
   - No añadas explicaciones, ni sugerencias, ni contexto extra.

5) Formato:
   - Si hay evidencia suficiente, responde con esta estructura exacta:
     ## Respuesta
     (explicación detallada y ordenada, con viñetas si conviene)

     ### Datos clave (extraídos del contexto)
     - (hechos concretos del contexto: fechas, nombres, lugares, eventos)

6) Prohibición de fuentes dentro del texto:
   - NO incluyas un apartado “Fuentes”, enlaces ni URLs.
   - Las fuentes las mostrará la aplicación fuera del modelo.
""".strip()

st.set_page_config(
    page_title="Churchill IA · WWII (RAG)",
    page_icon="🕯️",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background:#0b0f14; color:#e6edf3; }
      .block-container { max-width: 1100px; padding-top: 1.5rem; }
      .intro {
        background: rgba(230,237,243,0.04);
        border: 1px solid rgba(230,237,243,0.08);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
      }
      .stChatInput {
        position: sticky;
        bottom: 0;
        background: rgba(11,15,20,0.95);
        border-top: 1px solid rgba(230,237,243,0.08);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def _load():
    return load_store()


@st.cache_resource
def prepare_avatar_png(avatar_rel_path: str) -> str | None:
    base_dir = Path(__file__).resolve().parent
    src = (base_dir / avatar_rel_path).resolve()

    if not src.exists():
        return None

    out_dir = base_dir / ".cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "churchill_avatar.png"

    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGBA")
    img.save(out, format="PNG")

    return str(out)


CHURCHILL_IMG_REL = "images/churchill.jpg"
assistant_avatar = prepare_avatar_png(CHURCHILL_IMG_REL)

index, docs = _load()

st.markdown(
    """
    <div class="intro">
      <h2>Churchill IA · Asistente de la Segunda Guerra Mundial</h2>
      <p>
        Este asistente utiliza <b>RAG</b>. Antes de responder, recupera fragmentos relevantes de un índice vectorial
        construido a partir de <b>fuentes de Wikipedia</b> y <b>libros/documentos</b> del corpus del proyecto.
      </p>
      <p>
        Responde únicamente con evidencia recuperada. Si no existe evidencia suficiente, lo indicará explícitamente.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if assistant_avatar is None:
    st.error(
        "No puedo cargar la imagen de Churchill. Verifica que exista exactamente en: "
        f"`{CHURCHILL_IMG_REL}` (relativo a la carpeta donde está `app.py`)."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown("**Tú**")
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.markdown("**Churchill IA**")
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Fuentes"):
                    st.markdown(m["sources"])

q = st.chat_input("Escribe tu pregunta…")
if q and q.strip():
    st.session_state.messages.append({"role": "user", "content": q.strip()})
    with st.chat_message("user"):
        st.markdown("**Tú**")
        st.markdown(q.strip())

    qv = embed_query(q.strip())
    scores, ids = index.search(qv, TOP_K)

    hits = []
    for score, i in zip(scores[0], ids[0]):
        if int(i) == -1:
            continue
        hit = dict(docs[int(i)])
        hit["_score"] = float(score)
        hits.append(hit)

    good_hits = [h for h in hits if h["_score"] >= MIN_SCORE_TO_CITE]

    with st.chat_message("assistant", avatar=assistant_avatar):
        st.markdown("**Churchill IA**")

        if len(good_hits) < MIN_GOOD_HITS:
            st.markdown(NO_INFO_MSG)
            st.session_state.messages.append(
                {"role": "assistant", "content": NO_INFO_MSG, "sources": None}
            )
        else:
            context = format_context(good_hits)
            sources_block = format_sources(good_hits)

            prompt = f"""
{SYSTEM_PROMPT}

CONTEXTO (fragmentos recuperados):
{context}

PREGUNTA:
{q.strip()}

INSTRUCCIONES DE SALIDA (obligatorias):
- Si NO hay evidencia suficiente en el CONTEXTO, responde EXACTAMENTE: {NO_INFO_MSG}
- Si la pregunta NO trata sobre la Segunda Guerra Mundial, responde EXACTAMENTE: La pregunta está fuera del alcance del sistema.
- Si SÍ hay evidencia suficiente, responde con el siguiente formato:

## Respuesta
(Explicación detallada, ordenada, sin inventar datos)

### Datos clave (extraídos del contexto)
- Hecho 1
- Hecho 2
- Hecho 3
""".strip()

            answer = ollama_generate(prompt)
            st.markdown(answer)

            if answer != NO_INFO_MSG:
                with st.expander("Fuentes"):
                    st.markdown(sources_block)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources_block}
            )
