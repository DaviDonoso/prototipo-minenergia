# etl_normalize.py
import os, re, io, csv, unicodedata
import pandas as pd

def _norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip()

CANDIDATE_DELIMS = [";", ",", "\t", "|"]

def _sniff_delim(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
    except Exception:
        counts = {d: sample.count(d) for d in CANDIDATE_DELIMS}
        return max(counts, key=counts.get)

MESES = ["enero","febrero","marzo","abril","mayo","junio",
         "julio","agosto","septiembre","octubre","noviembre","diciembre"]
TRIM_CIERRE = {1:"marzo", 2:"junio", 3:"septiembre", 4:"diciembre"}

# ---------- Detección de período (archivo, ruta y headers) ----------
def _infer_period_from_filename_and_path(name: str, path: str):
    def _n(x): return _norm(x).lower().replace("-", " ").replace("_", " ")
    full = _n(path + " " + name)
    # Q1..Q4 (con o sin espacios)
    m = re.search(r"\bq\s*([1-4])\b", full, flags=re.I)
    if m:
        q = int(m.group(1))
        return {"period_type":"quarter", "quarter": q, "mes_cierre": TRIM_CIERRE[q]}
    # Mes explícito
    for mes in MESES:
        if re.search(rf"\b{mes}\b", full):
            return {"period_type":"month", "month": mes, "mes_cierre": mes}
    return {"period_type": None, "mes_cierre": None}

def _infer_period_from_headers(headers):
    """Si no hubo match por nombre/ruta, intenta deducir de headers como
       'Ejecución Acumulada a Primer Trimestre' o '... a Diciembre'."""
    if not headers: 
        return None
    hs = [_norm(h).lower() for h in headers]
    # Trimestres explícitos
    for q, word in {1:"primer", 2:"segundo", 3:"tercer", 4:"cuarto"}.items():
        for h in hs:
            if "ejec" in h and "acumul" in h and "trimestre" in h and word in h:
                return {"period_type":"quarter", "quarter": q, "mes_cierre": TRIM_CIERRE[q]}
    # Mes explícito en encabezado
    for mes in MESES:
        for h in hs:
            if "ejec" in h and "acumul" in h and mes in h:
                return {"period_type":"month", "month": mes, "mes_cierre": mes}
    return None

# ---------- Column mapping ----------
COL_ALIASES = {
    "subtitulo": ["subtitulo", "subtítulo", "sub t", "subt"],
    "item": ["item", "ítem"],
    "asignacion": ["asignacion", "asignación"],
    "sub_asignacion": ["subasignacion", "sub asignacion", "sub-asignacion", "sub asignación"],
    "denominacion": ["denominacion", "denominación", "glosa", "descripcion", "descripción"],
    "capitulo": ["capitulo", "capítulo"],
    "programa": ["programa"],
    "partida": ["partida"],
}

def _find_col(headers, wants):
    hs = [_norm(h).lower() for h in headers]
    for want in wants:
        w = want.lower()
        for i, h in enumerate(hs):
            if w in h:
                return headers[i]
    return None

# ---------- Exec column detection (flexible) ----------
def _score_header_for_exec(h: str, period_hint: dict):
    h0 = _norm(h).lower()
    score = 0
    # señales fuertes
    if "ejec" in h0: score += 2
    if "acumul" in h0: score += 2
    if "ejecutado" in h0: score += 2
    if "devengado" in h0: score += 1
    if "monto" in h0 and ("ejec" in h0 or "ejecut" in h0): score += 1
    # match de periodo
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
    # castigar vigentes/presupuesto/total (menos preferente)
    if "vigente" in h0 or "presupuesto" in h0: score -= 2
    return score

def _choose_exec_col(headers, period_hint):
    if not headers: return None
    best = None
    best_score = -999
    for h in headers:
        sc = _score_header_for_exec(h, period_hint)
        if sc > best_score:
            best_score = sc
            best = h
    # si lo mejor es muy bajo, acepta "total"/"vigente" como último recurso
    if best is None:
        for h in headers:
            h0 = _norm(h).lower()
            if "total" in h0 or "vigente" in h0 or "presupuesto" in h0:
                return h
    return best

def _to_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return 0.0
    s = str(x).replace("\u00a0","").replace(".", "").replace(",", ".").strip()
    try: return float(s)
    except: return 0.0

def normalize_csvs(input_paths_or_dir, default_partida="24"):
    """
    Lee uno o varios CSV (ruta o carpeta) y devuelve DataFrame canónico (todas las denominaciones).
    """
    # recolecta archivos
    if isinstance(input_paths_or_dir, str) and os.path.isdir(input_paths_or_dir):
        files = []
        for root, _, filenames in os.walk(input_paths_or_dir):
            for fn in filenames:
                if fn.lower().endswith(".csv"):
                    files.append(os.path.join(root, fn))
    else:
        files = input_paths_or_dir if isinstance(input_paths_or_dir, (list, tuple)) else [input_paths_or_dir]

    if not files:
        print(f"⚠️  normalize_csvs: no se encontraron .csv en {input_paths_or_dir}")
        return pd.DataFrame()

    rows = []
    for path in files:
        name = os.path.basename(path)

        # año por ruta o nombre
        anio = None
        parts = re.findall(r"(20[0-9]{2})", path)
        if parts: anio = parts[-1]

        # leer contenido
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        sample = text[:5000]
        delim = _sniff_delim(sample)
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        headers = reader.fieldnames or []

        # período: intenta por archivo/ruta, si no, por headers
        period_hint = _infer_period_from_filename_and_path(name, path)
        if not period_hint or not period_hint.get("mes_cierre"):
            h_hint = _infer_period_from_headers(headers)
            if h_hint: period_hint = h_hint

        # columna de ejecución (flexible)
        exec_col = _choose_exec_col(headers, period_hint)

        # columnas canon
        col_subt = _find_col(headers, COL_ALIASES["subtitulo"])
        col_item = _find_col(headers, COL_ALIASES["item"])
        col_asig = _find_col(headers, COL_ALIASES["asignacion"])
        col_subasig = _find_col(headers, COL_ALIASES["sub_asignacion"])
        col_deno = _find_col(headers, COL_ALIASES["denominacion"])
        col_cap = _find_col(headers, COL_ALIASES["capitulo"])
        col_prog = _find_col(headers, COL_ALIASES["programa"])
        col_part = _find_col(headers, COL_ALIASES["partida"])

        # --- detectar tipo de archivo por nombre ---
        nlow = _norm(name).lower()
        is_cap = "ejecucion_capitulo_" in nlow
        is_prog = "ejecucion_programa_" in nlow

        infer_cap = None
        infer_prog = None

        # si es archivo de CAPÍTULO -> inferimos capitulo
        if is_cap:
            for tag in ["subsecretaria", "cne", "cchen", "sec"]:
                if f"capitulo_{tag}" in nlow or f"capitulo {tag}" in nlow or f"_{tag}_" in nlow:
                    infer_cap = tag
                    break

        # si es archivo de PROGRAMA -> inferimos programa, NO capitulo
        if is_prog:
            m = re.search(r"ejecucion_programa_([a-z0-9_]+)", nlow)
            if m:
                infer_prog = m.group(1)

        for row in reader:
            monto = _to_float(row.get(exec_col)) if exec_col else 0.0

            # subtítulo -> clasifica ingreso/gasto
            subt = row.get(col_subt) if col_subt else None
            try:
                subt_num = int(str(subt).strip()) if subt is not None and str(subt).strip().isdigit() else None
            except:
                subt_num = None
            tipo_mov = None
            if subt_num is not None:
                if 5 <= subt_num <= 15:
                    tipo_mov = "INGRESO"
                elif 21 <= subt_num <= 34:
                    tipo_mov = "GASTO"

            rows.append({
                "anio": int(anio) if anio else None,
                "period_type": period_hint.get("period_type"),
                "quarter": period_hint.get("quarter"),
                "mes_cierre": period_hint.get("mes_cierre"),
                "partida": row.get(col_part) or default_partida,
                "capitulo": row.get(col_cap) or infer_cap,
                "programa": row.get(col_prog) or infer_prog,
                "subtitulo": subt_num,
                "item": row.get(col_item),
                "asignacion": row.get(col_asig),
                "sub_asignacion": row.get(col_subasig),
                "denominacion": row.get(col_deno),
                "tipo_mov": tipo_mov,
                "monto": monto,
                "fuente": name,
            })

    df = pd.DataFrame(rows)

    # limpia nulos de strings
    for c in ["partida","capitulo","programa","item","asignacion","sub_asignacion","denominacion","mes_cierre","period_type","fuente"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("")
    return df
