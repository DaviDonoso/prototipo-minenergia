import os
import pickle
from preprocess_embeddings import ingest_documents, get_embedding

EMBEDDINGS_FILE = "embeddings.pkl"

def load_embeddings(client, force_recalculate=False):
    # Si existe archivo y no forzamos recalcular → cargar directo
    if os.path.exists(EMBEDDINGS_FILE) and not force_recalculate:
        print(f"📂 Cargando embeddings desde {EMBEDDINGS_FILE}...")
        with open(EMBEDDINGS_FILE, "rb") as f:
            documentos = pickle.load(f)
        print(f"✅ Embeddings cargados: {len(documentos)} documentos.")
        return documentos

    # Si no existe o forzamos recalcular → generar
    documentos = ingest_documents("data")
    for i, doc in enumerate(documentos, start=1):
        try:
            doc["embedding"] = get_embedding(client, doc["contenido"][:3000])
            if i % 50 == 0:
                print(f"🔹 Progreso: {i}/{len(documentos)} documentos procesados")
        except Exception as e:
            print(f"❌ Error generando embedding para {doc['nombre']}: {e}")
            doc["embedding"] = []

    # Guardar en disco
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(documentos, f)
    print(f"💾 Embeddings guardados en {EMBEDDINGS_FILE}")

    return documentos
