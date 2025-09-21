# preprocess_embeddings.py
import unicodedata
import os
import re
import io
import csv
import numpy as np

def _norm(s: str) -> str:
    if s is None: return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()

# ---------------------------
# 1) Carga de documentos
# ---------------------------
def ingest_documents(data_folder):
    documentos = []
    for root, _, files in os.walk(data_folder):
        for filename in files:
            if filename.endswith(".csv"):
                ruta = os.path.join(root, filename)
                try:
                    with open(ruta, encoding="utf-8", errors="replace") as f:
                        contenido = f.read()
                    año = next((part for part in root.split(os.sep) if part.isdigit()), None)
                    institucion = os.path.basename(root)
                    documentos.append({
                        "nombre": filename,
                        "ruta": ruta,
                        "contenido": contenido,
                        "año": año,
                        "institucion": institucion
                    })
                except Exception as e:
                    print(f"❌ Error al leer {ruta}: {e}")
    print(f"📂 Total documentos cargados: {len(documentos)}")
    return documentos

# ---------------------------
# 2) Embeddings
# ---------------------------
def get_embedding(client, text, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
    return float(np.dot(v1, v2) / denom)

# ---------------------------
# 3) Parsing numérico/columnas robusto (fallback semántico)
# ---------------------------
MESES = ["enero","febrero","marzo","abril","mayo","junio",
         "julio","agosto","septiembre","octubre","noviembre","diciembre"]
TRIM_CIERRE = {1:"marzo", 2:"junio", 3:"septiembre", 4:"diciembre"}

def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
        return dialect.delimiter
    except Exception:
        counts = {d: sample.count(d) for d in [";",";",",","\t","|"]}
        return max(counts, key=counts.get)

def _to_float(x):
    if x is None: return 0.0
    s = str(x).strip().replace("\u00a0","")
    s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except: return 0.0

def _infer_period_from_name_path_headers(doc_name: str, doc_path: str, headers):
    def _n(x): return _norm(x).replace("-", " ").replace("_"," ")
    full = _n((doc_path or "") + " " + (doc_name or ""))
    # q1..q4
    m = re.search(r"\bq\s*([1-4])\b", full, flags=re.I)
    if m:
        q = int(m.group(1)); return {"period_type":"quarter", "quarter": q, "mes_cierre": TRIM_CIERRE[q]}
    # mes explícito
    for mes in MESES:
        if re.search(rf"\b{mes}\b", full):
            return {"period_type":"month", "month": mes, "mes_cierre": mes}
    # headers
    if headers:
        hs = [_norm(h) for h in headers]
        for q, word in {1:"primer", 2:"segundo", 3:"tercer", 4:"cuarto"}.items():
            for h in hs:
                if "ejec" in h and "acumul" in h and "trimestre" in h and word in h:
                    return {"period_type":"quarter", "quarter": q, "mes_cierre": TRIM_CIERRE[q]}
        for mes in MESES:
            for h in hs:
                if "ejec" in h and "acumul" in h and mes in h:
                    return {"period_type":"month", "month": mes, "mes_cierre": mes}
    return {"period_type": None, "mes_cierre": None}

def _score_header_for_exec(h: str, period_hint: dict):
    h0 = _norm(h)
    score = 0
    if "ejec" in h0: score += 2
    if "acumul" in h0: score += 2
    if "ejecutado" in h0: score += 2
    if "devengado" in h0: score += 1
    if "monto" in h0 and ("ejec" in h0 or "ejecut" in h0): score += 1
    if period_hint:
        if period_hint.get("period_type") == "quarter":
            q = period_hint.get("quarter")
            if q and f"q{q}" in h0: score += 2
            word = {1:"primer", 2:"segundo", 3:"tercer", 4:"cuarto"}.get(q)
            if word and "trimestre" in h0 and word in h0: score += 2
        mes = period_hint.get("mes_cierre")
        if mes:
            if mes in h0: score += 2
            if mes[:3] in h0: score += 1
    if "vigente" in h0 or "presupuesto" in h0: score -= 2
    return score

def _choose_exec_col(headers, period_hint):
    if not headers: return None
    best, best_score = None, -999
    for h in headers:
        sc = _score_header_for_exec(h, period_hint)
        if sc > best_score:
            best, best_score = h, sc
    if best is None:
        # último recurso
        for h in headers:
            h0 = _norm(h)
            if "total" in h0 or "vigente" in h0 or "presupuesto" in h0:
                return h
    return best

def sum_csv_doc(doc, annual_hint=False):
    """Suma ejecución del documento → usa mejor columna; filtra a GASTO 21–34 si hay subtítulo."""
    # prepara lector
    sample = doc["contenido"][:5000]
    delim = _sniff_delimiter(sample)
    f = io.StringIO(doc["contenido"])
    reader = csv.DictReader(f, delimiter=delim)
    headers = reader.fieldnames or []
    # detectar período (archivo + ruta + headers)
    period_hint = _infer_period_from_name_path_headers(doc.get("nombre"), doc.get("ruta"), headers)
    if annual_hint and not period_hint.get("mes_cierre"):
        # si es anual y no hay periodo → asumir diciembre (total)
        period_hint = {"period_type":"month", "month":"diciembre", "mes_cierre":"diciembre"}

    exec_col = _choose_exec_col(headers, period_hint)

    # columna subtítulo para filtrar GASTO
    sub_col = None
    if headers:
        for h in headers:
            h0 = _norm(h)
            if "subt" in h0: sub_col = h; break

    total = 0.0
    if not headers:
        return 0.0

    for row in reader:
        # filtra gasto por subtítulo
        if sub_col:
            try:
                sub = int(str(row.get(sub_col, "")).strip())
            except ValueError:
                sub = None
            if sub is not None and not (21 <= sub <= 34):
                continue
        val = _to_float(row.get(exec_col)) if exec_col else 0.0
        total += val

    print(f"🧭 {doc['nombre']} → delim='{delim}' | col='{exec_col}' | period='{period_hint.get('mes_cierre')}'")
    return total

# ---------------------------
# 4) Heurísticas alcance
# ---------------------------
ALIAS = {
    "partida": ["partida 24", "partida24", "partida_24"],
    "capitulos": {
        "subsecretaria": ["subsecretaría", "subsecretaria", "sse", "subsec"],
        "cne": ["comisión nacional de energía", "cne"],
        "cchen": ["comisión chilena de energía nuclear", "cchen"],
        "sec": ["superintendencia de electricidad y combustibles", "sec"],
    },
}
PRIORITY_FILES = [
    "ejecucion_partida24_",
    "ejecucion_capitulo_subsecretaria_",
    "ejecucion_capitulo_cne_",
    "ejecucion_capitulo_cchen_",
    "ejecucion_capitulo_sec_",
]

def _mentions_any(text, tokens):
    t = text.lower()
    return any(tok in t for tok in tokens)

def _is_partida24(nombre):
    n = nombre.lower().replace(" ", "").replace("-", "_")
    return ("partida24" in n) or ("partida_24" in n) or ("partida24_" in n)

def _cap_of(nombre):
    n = nombre.lower()
    for cap_key in ALIAS["capitulos"].keys():
        if cap_key in n:
            return cap_key
    return None

def _match_priority(nombre: str) -> bool:
    n = nombre.lower()
    return any(pat in n for pat in PRIORITY_FILES)

# ---------------------------
# 5) Búsqueda + agregación en Python (fallback)
# ---------------------------
def search_semantic(client, docs, query, model="gpt-4-turbo"):
    years = sorted(set(re.findall(r"\b(20[0-9]{2})\b", query)))
    q = query.lower()
    annual_hint = any(w in q for w in ["total", "anual", "año", "compar", "ejecución total", "ejecucion total"])
    quarterly_hint = ("trimestr" in q) or bool(re.search(r"\bq\s*[1-4]\b", q, flags=re.I))
    print(f"🧭 quarterly_hint={quarterly_hint} | annual_hint={annual_hint}")

    docs_year = [d for d in docs if (not years) or (d.get("año") in years)]

    wants_partida = _mentions_any(q, ALIAS["partida"])
    wanted_capitulos = [k for k,a in ALIAS["capitulos"].items() if _mentions_any(q, a)]
    docs_partida = [d for d in docs_year if _is_partida24(d["nombre"])]
    docs_cap = [d for d in docs_year if _cap_of(d["nombre"]) is not None and not _is_partida24(d["nombre"])]

    must = []
    if wants_partida: must += docs_partida
    if wanted_capitulos: must += [d for d in docs_cap if _cap_of(d["nombre"]) in wanted_capitulos]
    if not wants_partida and not wanted_capitulos:
        must = docs_partida + docs_cap
    must += [d for d in docs_year if _match_priority(d["nombre"])]
    # incluir Q4 si es anual (no trimestral) para asegurar “cierre”
    if years and annual_hint and not quarterly_hint:
        for y in years:
            q4s = [d for d in docs_year if d.get("año")==y and ("q4" in d["nombre"].lower() or "diciembre" in d["nombre"].lower())]
            for d in q4s:
                if d not in must: must.append(d)
    must = list({id(d): d for d in must}.values())

    query_emb = get_embedding(client, query)
    resto = [d for d in docs_year if d not in must and d.get("embedding")]
    ranked_resto = sorted(resto, key=lambda d: cosine_similarity(d["embedding"], query_emb), reverse=True)

    adicionales = ranked_resto[:max(0, 40 - len(must))]
    usados = must + adicionales

    if annual_hint and not quarterly_hint:
        def es_q4_dic(d):
            n = d["nombre"].lower()
            return ("q4_" in n) or ("diciembre" in n)
        usados = [d for d in usados if es_q4_dic(d)]

    print(f"🔹 Must: {len(must)} | 🔹 Adicionales: {len(adicionales)}")
    print(f"🔎 Archivos usados: {[d['nombre'] for d in usados][:15]}{'...' if len(usados)>15 else ''}")

    target_years = years or sorted({d.get("año") for d in usados if d.get("año")})

    # Trimestral: acumula a marzo/junio/sep/dic y saca totales por diferencia
    if quarterly_hint:
        def _accum(docs_year, year, scope_caps):
            acc = {1:0.0, 2:0.0, 3:0.0, 4:0.0}
            for d in docs_year:
                if d.get("año") != year: 
                    continue
                if scope_caps and all(cap not in d["nombre"].lower() for cap in scope_caps):
                    continue
                # sumar (sin forzar anual_hint)
                tot = sum_csv_doc(d, annual_hint=False)
                # inferir periodo para mapear a trimestre de cierre
                sample = d["contenido"][:5000]
                delim = _sniff_delimiter(sample)
                r = csv.DictReader(io.StringIO(d["contenido"]), delimiter=delim)
                headers = r.fieldnames or []
                ph = _infer_period_from_name_path_headers(d.get("nombre"), d.get("ruta"), headers)
                mes = ph.get("mes_cierre")
                if not mes: 
                    continue
                for q, m in TRIM_CIERRE.items():
                    if m == mes:
                        acc[q] += tot
                        break
            return acc

        scope_caps = wanted_capitulos[:]  # ej. ["sec"]
        lines = []
        for y in target_years:
            acc = _accum(docs_year, y, scope_caps)
            q1 = acc[1]
            q2 = max(acc[2]-acc[1], 0.0)
            q3 = max(acc[3]-acc[2], 0.0)
            q4 = max(acc[4]-acc[3], 0.0)
            lines.append(f"AÑO {y}: Q1={q1:,.2f} | Q2={q2:,.2f} | Q3={q3:,.2f} | Q4={q4:,.2f}")
        summary_text = "\n".join(lines)
        system_prompt = f"""Eres un analista presupuestario.
Usa EXCLUSIVAMENTE los totales trimestrales calculados en Python (Q1=marzo, Q2=junio, Q3=septiembre, Q4=diciembre).
No inventes ni reestimes.

TABLA
{summary_text}

Pregunta:
{query}

Instrucciones:
- Describe la variación entre trimestres con cifras y %.
- Si un trimestre está ausente (0.0), indícalo como falta de datos.
"""
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}]
        )
        return completion.choices[0].message.content.strip()

    # Anual: suma directa (con filtro a Q4 si tocaba)
    per_year = {y: 0.0 for y in target_years}
    per_year_details = {y: [] for y in target_years}
    for d in usados:
        y = d.get("año")
        if not y or y not in per_year: 
            continue
        tot = sum_csv_doc(d, annual_hint=annual_hint)
        per_year[y] += tot
        per_year_details[y].append({"archivo": d["nombre"], "total_doc": tot})

    resumen = []
    for y in target_years:
        resumen.append(f"AÑO {y}: TOTAL = {per_year[y]:,.2f}")
        dets = per_year_details[y]
        for it in dets[:8]:
            resumen.append(f"  - {it['archivo']}: {it['total_doc']:,.2f}")
        if len(dets) > 8:
            resumen.append(f"  - (+{len(dets)-8} archivos más)")
    summary_text = "\n".join(resumen)

    system_prompt = f"""Eres un analista presupuestario.
Usa EXCLUSIVAMENTE los totales anuales calculados en Python (cierre en diciembre para evitar doble conteo).
No inventes ni reestimes.

TABLA
{summary_text}

Pregunta:
{query}

Instrucciones:
- Si hay 2 años, da diferencia absoluta y %.
- Si faltan cierres de año, adviértelo.
"""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}]
    )
    return completion.choices[0].message.content.strip()
