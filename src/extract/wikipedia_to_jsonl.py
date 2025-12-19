import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from urllib.parse import quote

import requests

WIKI_API_ES = "https://es.wikipedia.org/w/api.php"


def wiki_fetch_extract(
    title: str, session: requests.Session, timeout: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Descarga el 'extract' en texto plano de Wikipedia (ES) usando la API oficial.
    Usa EXACTAMENTE los params solicitados por el usuario.
    Devuelve un dict listo para guardarse como una línea en JSONL o None si falla.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }

    r = session.get(WIKI_API_ES, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    query = data.get("query", {})
    pages = query.get("pages", {})

    # La API devuelve un diccionario con key=pageid (o "-1" si no existe)
    if not pages:
        return None

    page = next(iter(pages.values()))
    pageid = page.get("pageid", None)

    # Si no existe, Wikipedia suele dar pageid=-1 y no hay extract
    if pageid is None or page.get("missing") is not None or pageid == -1:
        return {
            "source": "wikipedia",
            "lang": "es",
            "requested_title": title,
            "status": "missing",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    final_title = page.get("title", title)
    extract = (page.get("extract") or "").strip()
    revid = (
        page.get("revisions", [{}])[0].get("revid") if page.get("revisions") else None
    )

    # URL “humana”
    url = f"https://es.wikipedia.org/wiki/{quote(final_title.replace(' ', '_'))}"

    doc = {
        "id": f"wikipedia:es:{pageid}",
        "source": "wikipedia",
        "lang": "es",
        "requested_title": title,
        "title": final_title,
        "pageid": pageid,
        "revision_id": revid,
        "url": url,
        "content": extract,
        "content_len": len(extract),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        # Guarda el JSON original por trazabilidad (útil para auditoría/citas)
        "raw_api_response": data,
    }
    return doc


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    titles = [
        "Segunda Guerra Mundial",
        "Batalla de Stalingrado",
        "Winston Churchill",
    ]

    out_path = ".cache/data/wikipedia_ww2_seed_es.jsonl"

    docs: List[Dict[str, Any]] = []
    with requests.Session() as session:
        # User-Agent recomendado para APIs públicas
        session.headers.update(
            {"User-Agent": "WW2-RAG-StudentProject/1.0 (contact: local)"}
        )

        for i, t in enumerate(titles, start=1):
            try:
                doc = wiki_fetch_extract(t, session=session)
                if doc is not None:
                    docs.append(doc)
                    print(
                        f"[{i}/{len(titles)}] [OK] {doc.get('title')} ({doc.get('status')})"
                    )
                else:
                    print(f"[{i}/{len(titles)}] [EMPTY] {t}")
            except requests.HTTPError as e:
                docs.append(
                    {
                        "source": "wikipedia",
                        "lang": "es",
                        "requested_title": t,
                        "status": "http_error",
                        "error": str(e),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"[{i}/{len(titles)}] HTTP error -> {t}: {e}")
            except Exception as e:
                docs.append(
                    {
                        "source": "wikipedia",
                        "lang": "es",
                        "requested_title": t,
                        "status": "error",
                        "error": str(e),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"[{i}/{len(titles)}] Error -> {t}: {e}")

            # Pequeña pausa por educación con el endpoint
            time.sleep(0.6)

    write_jsonl(out_path, docs)
    print(f"\n [OK] Saved {len(docs)} records to {out_path}")


if __name__ == "__main__":
    main()
