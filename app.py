import os
import io
import traceback
import logging
import tempfile
from flask import Flask, request, jsonify
from pypdf import PdfReader
from typing import Dict, Any

# LangChain y Google
import vertexai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
clients = {}

# --- CONFIGURACIÓN ---
# Al ser una cuenta nueva, podemos usar un nombre limpio para la colección.
COLLECTION_NAME = "pdf_embeded_documents" 

def get_clients():
    global clients
    if 'firestore' not in clients:
        logger.info("--- Inicializando clientes de Google Cloud... ---")
        try:
            PROJECT_ID = os.environ.get("PROJECT_ID")
            VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
            
            # Inicializar Vertex AI
            vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)
            
            clients['firestore'] = firestore.Client()
            clients['storage'] = storage.Client()
            
            # CAMBIO IMPORTANTE: Usando el modelo Gemini Embedding 001
            clients['embedding'] = VertexAIEmbeddings(model_name="gemini-embedding-001")
            
            # Modelo de Chat (puedes cambiarlo a gemini-1.5-pro si prefieres)
            clients['llm'] = ChatVertexAI(model_name="gemini-2.0-flash") 
            
            logger.info("--- Clientes inicializados correctamente. ---")
        except Exception as e:
            logger.error(f"--- !!! ERROR CRÍTICO inicializando clientes: {e} ---", exc_info=True)
            clients = {}
    return clients

def _process_and_embed_pdf_file(file_path: str, filename: str) -> Dict[str, Any]:
    try:
        logger.info(f"Iniciando procesamiento para el archivo: {filename}")
        clients_local = get_clients()
        firestore_client = clients_local.get('firestore')
        embedding_model = clients_local.get('embedding')
        
        if not firestore_client or not embedding_model:
            raise Exception("Clientes de GCP no disponibles.")
        
        # Verificar duplicados
        docs_ref = firestore_client.collection(COLLECTION_NAME)
        existing_docs = docs_ref.where(filter=FieldFilter("metadata.source", "==", filename)).limit(1).stream()
        
        if len(list(existing_docs)) > 0:
            logger.info(f"El archivo '{filename}' ya existe. Saltando...")
            return {"status": "skipped", "message": "Archivo ya procesado."}

        # Leer PDF desde archivo temporal
        reader = PdfReader(file_path)
        text_content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        if not text_content:
            return {"status": "error", "reason": "No se pudo extraer texto del PDF."}

        # Metadatos
        pdf_meta = reader.metadata
        book_title = pdf_meta.title if pdf_meta and pdf_meta.title else os.path.splitext(filename)[0].replace("_", " ").title()
        book_author = pdf_meta.author if pdf_meta and pdf_meta.author else "Autor Desconocido"
        
        # Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200) # Ajustado para Gemini
        chunks = text_splitter.split_text(text_content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={ 
                    "source": filename, 
                    "chunk_index": i, 
                    "title": book_title, 
                    "author": book_author,
                    "model": "gemini-embedding-001" # Marca de versión
                }
            )
            documents.append(doc)
        
        # Guardar en Firestore
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, embedding_service=embedding_model, client=firestore_client
        )
        
        batch_size = 50 # Reducido ligeramente para evitar timeouts de escritura
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Lote {i//batch_size + 1} insertado.")
        
        return {"status": "ok", "message": f"Archivo {filename} procesado con éxito."}
    except Exception as e:
        logger.error(f"Error procesando PDF: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

@app.route("/", methods=["POST"])
def handle_gcs_event():
    """Maneja eventos de Cloud Storage (Eventarc)."""
    try:
        clients_local = get_clients()
        storage_client = clients_local.get('storage')
        if not storage_client:
            return "Error interno", 500

        event = request.get_json(silent=True)
        if not event:
            return "Sin body", 400

        # Soporte para formato directo de GCS y formato Eventarc
        bucket_name = event.get("bucket")
        file_id = event.get("name")
        
        if not bucket_name or not file_id:
            return "Evento ignorado (no es un archivo)", 200

        logger.info(f"Evento recibido para bucket: {bucket_name}, archivo: {file_id}")
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        
        if not blob.exists():
            return "Archivo no encontrado", 200 # 200 para que no reintente infinitamente
            
        if blob.size == 0:
            return "Archivo vacío", 200

        # MEJORA: Descarga a archivo temporal para ahorrar RAM
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            logger.info(f"Descargando {file_id} a {temp_pdf.name}...")
            blob.download_to_filename(temp_pdf.name)
            temp_pdf.close() # Cerrar para liberar el file handle
            
            try:
                # Procesar
                result = _process_and_embed_pdf_file(temp_pdf.name, file_id)
            finally:
                # Limpieza obligatoria
                if os.path.exists(temp_pdf.name):
                    os.unlink(temp_pdf.name)

        if result.get("status") == "ok":
            return jsonify(result), 200
        else:
            return jsonify(result), 200 # Retornamos 200 incluso en error lógico para evitar reintentos de PubSub si el PDF está corrupto

    except Exception as e:
        logger.error(f"Error inesperado en handler: {e}", exc_info=True)
        return f"Error: {str(e)}", 500

@app.route("/query", methods=["POST"])
def query_rag_handler():
    try:
        request_data = request.get_json()
        if not request_data or "query" not in request_data:
             return jsonify({"error": "Falta el campo 'query'"}), 400
             
        user_query = request_data["query"]
        
        clients_local = get_clients()
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, 
            embedding_service=clients_local.get('embedding'), 
            client=clients_local.get('firestore')
        )
        
        # Búsqueda de similitud
        found_docs = vector_store.similarity_search(query=user_query, k=5)

        results = []
        for doc in found_docs:
            # CORRECCIÓN: Acceso directo a metadatos sin .get("metadata") extra
            meta = doc.metadata
            
            result_item = {
                "source": meta.get("source", "Desconocido"),
                "title": meta.get("title", "Sin título"),
                "author": meta.get("author", "Desconocido"),
                "content": doc.page_content
            }
            results.append(result_item)
        
        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"Error en query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
