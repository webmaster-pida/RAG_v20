import os
import io
import traceback
import logging
import tempfile
import warnings

# Filtramos la advertencia de storage para limpiar logs
warnings.filterwarnings("ignore", "Support for google-cloud-storage", category=FutureWarning)

from flask import Flask, request, jsonify
from pypdf import PdfReader
from typing import Dict, Any, List

# LangChain y Google
import vertexai
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # <--- Importante para la clase custom
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import ChatVertexAI
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# SDK Nativo de Vertex (Bypass para el error de Pydantic)
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
clients = {}

COLLECTION_NAME = "pdf_embeded_documents"

# --- CLASE CUSTOM PARA SOLUCIONAR EL ERROR DE DEPENDENCIAS ---
class CustomGeminiEmbeddings(Embeddings):
    """
    Wrapper personalizado para Gemini Embeddings que soporta output_dimensionality
    sin depender de la versión de langchain-google-vertexai.
    """
    def __init__(self, model_name="gemini-embedding-001", dimensionality=2048):
        self.model_name = model_name
        self.dimensionality = dimensionality
        # Cargamos el modelo usando el SDK nativo, que sí soporta la función
        self.client = TextEmbeddingModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Vertex AI procesa mejor en lotes pequeños (ej. 100), pero aquí hacemos
        # una implementación simple. El SDK maneja la paginación interna a veces.
        embeddings = []
        # Procesamos en mini-baches de 20 para evitar límites de la API nativa
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in batch]
            try:
                # AQUÍ está la clave: pasamos output_dimensionality directamente al SDK
                results = self.client.get_embeddings(inputs, output_dimensionality=self.dimensionality)
                embeddings.extend([embedding.values for embedding in results])
            except Exception as e:
                logger.error(f"Error generando embeddings nativos: {e}")
                raise e
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_QUERY")]
        results = self.client.get_embeddings(inputs, output_dimensionality=self.dimensionality)
        return results[0].values

# -----------------------------------------------------------

def get_clients():
    global clients
    if 'firestore' not in clients:
        logger.info("--- Inicializando clientes de Google Cloud... ---")
        try:
            PROJECT_ID = os.environ.get("PROJECT_ID")
            VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
            
            # Inicializar SDK
            vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)
            
            clients['firestore'] = firestore.Client()
            clients['storage'] = storage.Client()
            
            # CAMBIO CRÍTICO: Usamos nuestra clase custom en lugar de VertexAIEmbeddings
            # Esto nos garantiza 2048 dimensiones sin errores de validación
            logger.info("Inicializando CustomGeminiEmbeddings (2048 dim)...")
            clients['embedding'] = CustomGeminiEmbeddings(
                model_name="gemini-embedding-001",
                dimensionality=2048
            )
            
            MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            logger.info(f"Usando modelo LLM: {MODEL_NAME}")
            clients['llm'] = ChatVertexAI(model_name=MODEL_NAME) 
            
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

        reader = PdfReader(file_path)
        text_content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        if not text_content:
            return {"status": "error", "reason": "No se pudo extraer texto del PDF."}

        pdf_meta = reader.metadata
        book_title = pdf_meta.title if pdf_meta and pdf_meta.title else os.path.splitext(filename)[0].replace("_", " ").title()
        book_author = pdf_meta.author if pdf_meta and pdf_meta.author else "Autor Desconocido"
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text_content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={ 
                    "source": filename, 
                    "chunk_index": i, 
                    "title": book_title, 
                    "author": book_author
                }
            )
            documents.append(doc)
        
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, embedding_service=embedding_model, client=firestore_client
        )
        
        # Loteamos la subida a Firestore
        batch_size = 50 
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Lote {i//batch_size + 1} insertado en Firestore.")
        
        return {"status": "ok", "message": f"Archivo {filename} procesado con éxito."}
    except Exception as e:
        logger.error(f"Error procesando PDF: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

@app.route("/", methods=["POST"])
def handle_gcs_event():
    try:
        clients_local = get_clients()
        storage_client = clients_local.get('storage')
        if not storage_client: return "Error interno", 500

        event = request.get_json(silent=True)
        if not event: return "Sin body", 400

        bucket_name = event.get("bucket")
        file_id = event.get("name")
        
        if not bucket_name or not file_id: return "Evento ignorado", 200

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        
        if not blob.exists(): return "Archivo no encontrado", 200
        if blob.size == 0: return "Archivo vacío", 200

        # Archivo temporal para evitar OOM
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            blob.download_to_filename(temp_pdf.name)
            temp_pdf.close()
            try:
                result = _process_and_embed_pdf_file(temp_pdf.name, file_id)
            finally:
                if os.path.exists(temp_pdf.name): os.unlink(temp_pdf.name)

        return jsonify(result), 200

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
        
        found_docs = vector_store.similarity_search(query=user_query, k=5)
        
        llm = clients_local.get('llm')
        context_text = "\n\n".join([doc.page_content for doc in found_docs])
        
        prompt = f"""Eres un asistente útil y preciso. Responde a la pregunta basándote SOLO en el siguiente contexto:
        
        CONTEXTO:
        {context_text}
        
        PREGUNTA:
        {user_query}
        """
        
        ai_response = llm.invoke(prompt)

        results = []
        for doc in found_docs:
            meta = doc.metadata
            result_item = {
                "source": meta.get("source"),
                "title": meta.get("title"),
                "content": doc.page_content
            }
            results.append(result_item)
        
        return jsonify({
            "answer": ai_response.content,
            "sources": results
        }), 200

    except Exception as e:
        logger.error(f"Error en query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
