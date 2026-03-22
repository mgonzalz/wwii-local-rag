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
    titles =[
    "Anexo:Participantes en la Segunda Guerra Mundial",
    "Aliados (Segunda Guerra Mundial)",
    "Potencias del Eje",
    "Holocausto",
    "Bombardeos estratégicos durante la Segunda Guerra Mundial",
    "Arma nuclear",
    "Anexo:Víctimas de la Segunda Guerra Mundial",
    "Invasión alemana de Polonia de 1939",
    "Invasión soviética de Polonia de 1939",
    "Ocupación de las repúblicas bálticas",
    "Unión Soviética",
    "Guerra de Invierno",
    "Guerra de broma",
    "Ocupación de Francia por las Fuerzas del Eje",
    "Ocupación de Noruega por la Alemania nazi",
    "Ocupación de Dinamarca por la Alemania nazi",
    "Ocupación alemana de Bélgica durante la Segunda Guerra Mundial",
    "Operación Dinamo",
    "Batalla de Dunkerque (1940)",
    "Batalla de los Alpes",
    "Guerra greco-italiana",
    "Batalla de Inglaterra",
    "Winston Churchill",
    "Franklin D. Roosevelt",
    "Carta del Atlántico",
    "Operación Barbarroja",
    "Batalla de Moscú",
    "Sitio de Leningrado",
    "Batalla de Stalingrado",
    "Imperio del Japón",
    "Acontecimientos que condujeron al ataque a Pearl Harbor",
    "Ataque a Pearl Harbor",
    "Batalla de Midway",
    "Guerra del Pacífico (1937-1945)",
    "Afrika Korps",
    "Invasión italiana de Egipto",
    "Batalla de Anzio",
    "Invasión aliada de Sicilia",
    "Batalla de Prójorovka",
    "Cohete V2",
    "Bombardeo de Dresde",
    "Batalla de Berlín",
    "Frente del Océano Pacífico (Segunda Guerra Mundial)",
    "Campaña de Guadalcanal",
    "Batalla del mar del Coral",
    "Batalla del golfo de Leyte",
    "Batalla de Iwo Jima",
    "Bombardeos atómicos de Hiroshima y Nagasaki",
    "Rendición de Japón",
    "Hirohito",
    "Capitulación alemana de mayo de 1945",
    "Conferencia de Teherán",
    "Conferencia de Yalta",
    "Conferencia de Potsdam",
    "Guerra Fría",
    "Descolonización",
    "Plan Marshall",
    "Campo de concentración de Dachau",
    "Leyes de Núremberg",
    "Campo de concentración de Sachsenhausen",
    "Crisis de los Sudetes",
    "Noche de los cristales rotos",
    "Gueorgui Zhúkov",
    "Viacheslav Mólotov",
    "Ocupación soviética de Letonia en 1940",
    "Anexo:Cronología de los Aliados de la Segunda Guerra Mundial",
    "Anexo:Enfrentamientos bélicos de la Segunda Guerra Mundial",
    "Anexo:Cronología de la Segunda Guerra Mundial (1939)",
    "Isoroku Yamamoto",
    "Erwin Rommel",
    "Benito Mussolini",
    "Anexo:Cronología de la Segunda Guerra Mundial (1940)",
    "Hermann Göring",
    "Operación Weserübung",
    "Campo de concentración de Auschwitz",
    "Anexo:Cronología de la Segunda Guerra Mundial (1941)",
    "Discurso de las cuatro libertades",
    "Real Fuerza Aérea (Reino Unido)",
    "Anexo:Cronología de la Segunda Guerra Mundial (1942)",
    "Declaración de las Naciones Unidas",
    "Chiang Kai-shek",
    "Campo de exterminio de Treblinka",
    "Campaña de las Islas Aleutianas",
    "Escuadrón 201",
    "Captura de Tobruk por el Eje",
    "Batalla del Edson Ridge",
    "Batalla del cabo Esperanza",
    "Enrico Fermi",
    "Anexo:Cronología de la Segunda Guerra Mundial (1943)",
    "Batalla del mar de Bismarck",
    "Operación Cartwheel",
    "Anexo:Cronología de la Segunda Guerra Mundial (1944)",
    "Batalla de Normandía",
    "Día D",
    "Masacre de Oradour-sur-Glane",
    "Campaña de las islas Marianas y Palaos",
    "Batalla del mar de Filipinas",
    "Operación Bagratión",
    "Ejército Rojo",
    "Operación Cobra",
    "Operación Dragoon",
    "Liberación de París",
    "Batalla del Bosque de Hürtgen",
    "Batalla del estuario del Escalda",
    "Batalla de Memel",
    "Atentado del 20 de julio de 1944",
    "Liberación de Belgrado",
    "Anexo:Cronología de la Segunda Guerra Mundial (1945)",
    "Segunda guerra sino-japonesa",
    "Pacto de Acero",
    "Tratado de Versalles (1919)",
    "Mi lucha",
    "Partido Nacionalsocialista Obrero Alemán",
    "Noche de los cuchillos largos",
    "Anschluss",
    "Acuerdos de Múnich",
    "Alemania nazi",
    "Pacto Ribbentrop-Mólotov",
    "Frente oriental (Segunda Guerra Mundial)",
    "Frente occidental (Segunda Guerra Mundial)",
    "Guerra del Pacífico (1937-1945)",
    "Heinrich Himmler",
    "Pueblo judío",
    "Tratado de Paz de Moscú",
    "Guerra de Continuación",
    "Ocupación de Polonia (1939-1945)",
    "Erich Raeder",
    "Blitz",
    "Guerra relámpago",
    "Fuerza Expedicionaria Británica (Segunda Guerra Mundial)",
    "Batalla de Francia",
    "Batalla de los Países Bajos",
    "Batalla de Bélgica",
    "Remilitarización de Renania",
    "Heinz Guderian",
    "Operación León Marino",
    "Frente del Mediterráneo en la Segunda Guerra Mundial",
    "Invasión italiana de Albania",
    "Batalla de Grecia",
    "Segunda batalla de El Alamein",
    "Ferdinand Schörner",
    "Bombardeo de Belgrado (1941)",
    "Campo de concentración de Banjica",
    "Batalla de Creta",
    "Batalla de Mazalquivir",
    "Operación Anton",
    "Campaña en África del Norte",
    "Bernard Law Montgomery",
    "Campaña del Desierto Occidental",
    "Operación Brevity",
    "Operación Crusader",
    "Primera batalla de El Alamein",
    "Italia fascista",
    "Francia de Vichy",
    "Solución final",
    "Invasión de Yugoslavia",
    "Campaña de Filipinas (1941-1942)",
    "Pacto Tripartito",
    "Incursión Doolittle",
    "Batalla de Tassafaronga",
    "Batalla de Buna-Gona",
    "Batalla de Singapur",
    "U-Boot#Segunda_Guerra_Mundial_(1939-1945)",
    "Enigma (máquina)",
    "Batalla del Atlántico",
    "Operación Azul",
    "Segunda batalla de Járkov",
    "Batalla del mar de Java",
    "Iósif Stalin",
    "Operación Chispa",
    "Ofensiva de Siniávino",
    "Batalla del Dniéper",
    "Campaña de las Islas Salomón",
    "Campaña de Filipinas (1944-1945)",
    "Bombardeo de Tokio",
    "Batalla de Okinawa",
    "Operación Market-Garden",
    "Invasión de los Aliados occidentales de Alemania",
    "Fin de la Segunda Guerra Mundial en Europa",
    "Batalla de Praga",
    "Bolsa del Ruhr",
    "Harry S. Truman",
    "Fat Man",
    "Día de la Victoria en Europa",
    "Bazuca",
    "Junkers Ju 87",
    "Boeing B-17 Flying Fortress",
    "Proyecto Manhattan",
    "Teatro americano en la Segunda Guerra Mundial",
    "Brasil en la Segunda Guerra Mundial",
    "México en la Segunda Guerra Mundial",
    "Chile durante la Segunda Guerra Mundial",
    "Colombia durante la Segunda Guerra Mundial",
    "España durante la Segunda Guerra Mundial",
    "Operación Félix",
    "Entrevista de Hendaya",
    "División Azul",
    "Wehrmacht",
    "Generalplan Ost",
    "Campo de exterminio",
    "Anexo:España en la Segunda Guerra Mundial",
    "Prostíbulos militares alemanes en la Segunda Guerra Mundial",
    "Experimentación nazi en seres humanos",
    "V-1 (bomba voladora)",
    "Posguerra de la Segunda Guerra Mundial",
    "Papel de la mujer en la Segunda Guerra Mundial",
    "Ana Frank",
    "Juicios de Núremberg",
    "Milagro económico alemán",
    "Campo de concentración de Majdanek",
    "Anexo:Cambios territoriales de la Segunda Guerra Mundial",
    "Historia diplomática de la Segunda Guerra Mundial",
    "Anexo:Líderes de las Potencias del Eje en la Segunda Guerra Mundial",
    "Causas de la Segunda Guerra Mundial",
    "Alemania en la Segunda Guerra Mundial",
    "Italia en la Segunda Guerra Mundial",
    "Japón durante la Segunda Guerra Mundial",
    "Reino Unido en la Segunda Guerra Mundial",
    "Historia militar de Estados Unidos en la Segunda Guerra Mundial",
    "Unión Soviética en la Segunda Guerra Mundial",
    "Frente de China en la Segunda Guerra Mundial",
    "Batalla de Kursk",
    "Batalla de las Ardenas",
    "Joseph Goebbels",
    "Violaciones durante la ocupación de Alemania",
    "Resistencia francesa",
    "Armia Krajowa",
    "Partisanos yugoslavos",
    "Resistencia italiana",
    "Crisis de Dánzig",
    "Política de apaciguamiento",
    "Dwight D. Eisenhower",
    "George S. Patton",
    "Douglas MacArthur",
    "Chester Nimitz",
    "Charles de Gaulle",
    "Jean Moulin",
    "Hideki Tōjō",
    "Tomoyuki Yamashita",
    "Chūichi Nagumo",
    "Karl Dönitz",
    "Albert Speer",
    "Wilhelm Keitel",
    "Alan Turing",
    "Reinhard Heydrich",
    "J. Robert Oppenheimer",
    "Batalla de Montecassino",
    "Batallas de Rzhev",
    "Sitio de Sebastopol (1941-1942)",
    "Batalla de Smolensk (1941)",
    "Batalla de Arras (1940)",
    "Batalla de Sedan",
    "Batallas de Narvik",
    "Batalla de Jaljin Gol",
    "Ocupación japonesa de Malasia",
    "Campaña de Birmania",
    "Marcha de la Muerte de Bataán",
    "Propaganda en la Unión Soviética",
    "Escuadrón 731",
    "Mujeres de consuelo",
    "T-34",
    "Panzer IV",
    "Panzer VI Tiger",
    "Yamato (1941)",
    "Bismarck (1940)",
    "Messerschmitt Bf 109",
    "Batalla de Luzón",
    "Propaganda nazi",
    "Fascismo",
    "Militarismo japonés",
    "Estalinismo",
    "Campaña de Italia (Segunda Guerra Mundial)",
    "Campaña soviética de los Balcanes",
    "Segunda Guerra Mundial",
    "Conferencia de Casablanca",
    "Operación Torch",
    "Campo de exterminio de Sobibor",
    "Campo de exterminio de Bełżec",
    "Operación Reinhard",
    "Masacre de Katyn",
    "Masacre de Nankín",
    "Eva Braun",
    "Categoría:Militares de la Segunda Guerra Mundial",
    "Categoría:Operaciones militares de la Segunda Guerra Mundial",
    "Adolf Hitler"
]

    out_path = ".cache/data/wiki/wikipedia_ww2_seed_es.jsonl"

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
