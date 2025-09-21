from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

import os
import re
import traceback
from dotenv import load_dotenv
from openai import OpenAI

from etl_normalize import normalize_csvs

# === NUEVO: ETL + Analytics determinístico ===
from etl_normalize import normalize_csvs
from analytics import (
    totales_trimestrales,
    totales_anuales,
    serie_mensual,
    desglose_por_denominacion,
)

# === EXISTENTE: embeddings (fallback) ===
from loader import load_embeddings
from preprocess_embeddings import search_semantic  # fallback semántico

app = FastAPI()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".")

# === GLOBALS ===
documentos_global = []
df_canonico = None  # DataFrame normalizado de todos los CSV

# ---------- helpers intención/scope ----------
def _detect_intents(q: str):
    qn = q.lower()
    return {
        "quarterly": ("trimestr" in qn) or bool(re.search(r"\bq\s*[1-4]\b", qn, re.I)),
        "annual": any(w in qn for w in ["total", "anual", "año", "compar"]),
        "monthly": any(w in qn for w in ["mensual", "mes a mes", "evolución mensual", "evolucion mensual"]),
        "breakdown": any(w in qn for w in ["denominacion", "denominación", "glosa", "detalle", "desglose"]),
    }

def _build_scope(q: str):
    qn = q.lower()
    scope = {"incluir_ingresos": False}  # por defecto, excluye ingresos (GASTO 21–34)

    # años
    years = sorted(set(re.findall(r"\b(20[0-9]{2})\b", q)))
    if years:
        scope["anio"] = [int(y) for y in years]

    # capítulo / programa
    # capítulos conocidos
    caps = []
    if "sec" in qn or "superintendencia de electricidad y combustibles" in qn:
        caps.append("sec")
    if "subsecretaria" in qn or "subsecretaría" in qn or "sse" in qn or "subsec" in qn:
        caps.append("subsecretaria")
    if "cne" in qn or "comision nacional de energia" in qn or "comisión nacional de energía" in qn:
        caps.append("cne")
    if "cchen" in qn or "comision chilena de energia nuclear" in qn or "comisión chilena de energía nuclear" in qn:
        caps.append("cchen")
    if caps:
        scope["capitulo"] = list(set(caps))

    # programas (palabras clave simples)
    progs = []
    if "aderc" in qn:
        progs.append("aderc")
    if "ers" in qn:
        progs.append("ers")
    if "paee" in qn:
        progs.append("paee")
    if "transicion justa" in qn or "transición justa" in qn:
        progs.append("transicion_justa")
    if "subsecretaria" in qn or "subsecretaría" in qn:
        progs.append("subsecretaria")
    if progs:
        scope["programa"] = list(set(progs))

    # (Opcional) rango subtítulos si se pide ingresos también
    if "incluye ingresos" in qn or "con ingresos" in qn:
        scope["incluir_ingresos"] = True

    return scope

def _summarize_df_for_prompt(df):
    # tabla compacta para GPT (máximo ~30 filas para no inflar prompt)
    try:
        if len(df) > 30:
            head = df.head(15).to_string(index=False)
            tail = df.tail(15).to_string(index=False)
            return f"{head}\n...\n{tail}\n(filas: {len(df)})"
        return df.to_string(index=False)
    except Exception:
        return str(df)

def _answer_with_gpt(model, summary_text, question):
    system_prompt = f"""Eres un analista presupuestario del Gobierno de Chile.
Tienes que responder usando EXCLUSIVAMENTE la tabla resumida (números ya calculados en Python).
No recalcules, no inventes cifras, no asumas datos faltantes.

TABLA
{summary_text}

Pregunta:
{question}

Instrucciones:
- Explica brevemente los resultados (variaciones, tendencias).
- Si corresponde, entrega diferencia absoluta y variación % entre periodos.
- Si la tabla no permite responder algo, dilo explícitamente.
"""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}],
    )
    return completion.choices[0].message.content.strip()

# ---------- enrutador principal ----------
def route_and_answer(question: str) -> str:
    global df_canonico, documentos_global

    intents = _detect_intents(question)
    scope = _build_scope(question)

    print(f"🧭 intents={intents} | scope={scope}")

    # 1) Trimestral
    if intents["quarterly"] and df_canonico is not None:
        df_res = totales_trimestrales(df_canonico, scope)
        print("🔢 trimestral\n", df_res)
        if df_res.empty:
            return "No se encontraron datos trimestrales para ese alcance/periodo."
        return _answer_with_gpt("gpt-5", _summarize_df_for_prompt(df_res), question)

    # 2) Anual (totales con Q4/diciembre)
    if intents["annual"] and df_canonico is not None:
        df_res = totales_anuales(df_canonico, scope)
        print("🔢 anual\n", df_res)
        if df_res.empty:
            return "No se encontraron datos anuales para ese alcance/periodo."
        return _answer_with_gpt("gpt-5", _summarize_df_for_prompt(df_res), question)

    # 3) Mensual (acumulado a cada mes)
    if intents["monthly"] and df_canonico is not None:
        df_res = serie_mensual(df_canonico, scope)
        print("🔢 mensual\n", df_res)
        if df_res.empty:
            return "No se encontraron datos mensuales para ese alcance/periodo."
        return _answer_with_gpt("gpt-5", _summarize_df_for_prompt(df_res), question)

    # 4) Desglose por denominación (top)
    if intents["breakdown"] and df_canonico is not None:
        df_res = desglose_por_denominacion(df_canonico, scope, top=20, periodo="anual")
        print("🔢 desglose\n", df_res.head(5))
        if df_res.empty:
            return "No se encontraron denominaciones para ese alcance/periodo."
        return _answer_with_gpt("gpt-5", _summarize_df_for_prompt(df_res), question)

    # 5) Fallback semántico (lo que ya tenías, con embeddings)
    #    -> útil para preguntas abiertas, comparativas texto, etc.
    return search_semantic(client, documentos_global, question)

# ---------- FastAPI ----------
@app.on_event("startup")
async def startup_event():
    global documentos_global, df_canonico
    print("Cargando embeddings (para fallback) y normalizando CSV...")

    # ✅ pass the client
    documentos_global = load_embeddings(client, force_recalculate=False)

    try:
        df_canonico = normalize_csvs("data")  # lee TODOS los CSV disponibles
        print(f"✅ DF canónico cargado: {len(df_canonico)} filas")
    except Exception as e:
        print(f"⚠️ No se pudo normalizar CSV (se usará solo el flujo semántico): {e}")
        df_canonico = None

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": ""})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    try:
        if not question.strip():
            return templates.TemplateResponse("index.html", {"request": request, "response": "La pregunta no puede estar vacía."})

        print(f"➡️ Pregunta recibida: {question}")
        response = route_and_answer(question)
        print(f"✅ Respuesta generada: {response[:200]}...")
        return templates.TemplateResponse("index.html", {"request": request, "response": response})
    except Exception as e:
        print(f"❌ Error procesando la pregunta: {str(e)}")
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {"request": request, "response": f"Error procesando la pregunta: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
