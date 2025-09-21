# Prototipo MEM – Análisis Presupuestario (Partida 24)

API en **FastAPI** + pipeline **ETL** (normalización CSV) + módulo **analytics** (cálculos determinísticos) + fallback semántico con **embeddings**.

## Estructura
- `main.py` – Servidor FastAPI y ruteo/intents.
- `etl_normalize.py` – Normalización y consolidación de CSV (DF canónico).
- `analytics.py` – Totales anuales/trimestrales, series mensuales, desgloses.
- `preprocess_embeddings.py` – Ingesta y vectorización (text-embedding-3-small).
- `loader.py` – Carga de embeddings persistidos (`embeddings.pkl`).
- `index.html` – Formulario simple para consultas.
- `requirements.txt` – Dependencias.

## Requisitos
```bash
python >= 3.10
pip install -r requirements.txt
