import os
import io
import json
import traceback
import logging
import tempfile
import warnings

# Filtramos advertencias de librerías
warnings.filterwarnings("ignore", "Support for google-cloud-storage", category=FutureWarning)

from flask import Flask, request, jsonify
from typing import Dict, Any, List

# LangChain y Google (Nuevas importaciones)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# SDK GenAI unificado y LangChain GenAI
from google import genai
from google.genai.types import EmbedContentConfig
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
clients = {}

COLLECTION_NAME = "pida_kb_genai-v20" # Nombre nuevo sugerido para la nueva estructura

# --- CLASE CUSTOM MIGRADA AL NUEVO SDK ---
class CustomGeminiEmbeddings(Embeddings):
    def __init__(self, model_name="gemini-embedding-001", dimensionality=2048, project=None, location=None):
        self.model_name = model_name
        self.dimensionality = dimensionality
        # Inicializamos el nuevo cliente unificado en modo VERTEX AI (Seguridad Enterprise)
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # El nuevo SDK acepta listas directamente y es más limpio
        config = EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.dimensionality
        )
        try:
            # Mandamos el lote (batch) de textos
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=config
            )
            # El objeto response.embeddings contiene la lista de vectores
            return [embedding.values for embedding in response.embeddings]
        except Exception as e:
            logger.error(f"Error generando embeddings con nuevo SDK: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        config = EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.dimensionality
        )
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text, # Aquí se manda un solo string
            config=config
        )
        return response.embeddings[0].values

# -----------------------------------------------------------

def get_clients():
    global clients
    if 'firestore' not in clients:
        logger.info("--- Inicializando clientes (Nuevo SDK google-genai)... ---")
        try:
            PROJECT_ID = os.environ.get("PROJECT_ID")
            VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
            
            clients['firestore'] = firestore.Client()
            clients['storage'] = storage.Client()
            
            # Usamos el wrapper custom actualizado
            clients['embedding'] = CustomGeminiEmbeddings(
                model_name="gemini-embedding-001",
                dimensionality=2048,
                project=PROJECT_ID,
                location=VERTEX_AI_LOCATION
            )
            
            # Nuevo modelo de Chat de LangChain compatible con el entorno Vertex
            MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            logger.info(f"Usando modelo LLM nativo: {MODEL_NAME}")
            # Usamos el cliente nativo unificado en modo Vertex para extracción
            clients['genai_client'] = genai.Client(vertexai=True, project=PROJECT_ID, location=VERTEX_AI_LOCATION)
            clients['llm_model_name'] = MODEL_NAME 
            
            logger.info("--- Clientes inicializados. ---")
        except Exception as e:
            logger.error(f"ERROR CRÍTICO inicializando: {e}", exc_info=True)
            clients = {}
    return clients

def _process_and_embed_text_file(file_path: str, filename: str) -> Dict[str, Any]:
    try:
        logger.info(f"Procesando archivo de texto: {filename}")
        clients_local = get_clients()
        firestore_client = clients_local.get('firestore')
        embedding_model = clients_local.get('embedding')
        genai_client = clients_local.get('genai_client')
        llm_model_name = clients_local.get('llm_model_name')
        
        if not firestore_client or not embedding_model:
            raise Exception("Clientes GCP no disponibles.")
        
        # Verificar si ya existe (Opcional: podrías querer sobrescribir)
        docs_ref = firestore_client.collection(COLLECTION_NAME)
        existing_docs = docs_ref.where(filter=FieldFilter("metadata.source", "==", filename)).limit(1).stream()
        if len(list(existing_docs)) > 0:
            logger.warning(f"El archivo {filename} ya existe. Saltando o podrías borrarlo aquí para re-indexar.")
            return {"status": "skipped", "message": "Archivo ya existe en la base de datos."}

        # 1. LEER TEXTO PLANO
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text_content = f.read()
        
        if not text_content:
            return {"status": "error", "reason": "El archivo está vacío."}

        # --- NUEVO: EXTRACCIÓN INTELIGENTE DE METADATOS ---
        doc_title = filename
        doc_author = "Desconocido"
        
        try:
            # Tomamos una muestra del inicio donde suele estar el título/autor
            sample_text = text_content[:3000]
            
            prompt_meta = f"""Eres un bibliotecario experto. Analiza el siguiente fragmento de texto y extrae el Título y el Autor.
            
            Reglas:
            1. Si no encuentras el autor explícitamente, pon "Autor Desconocido".
            2. Si no encuentras el título claro, usa: "{filename}".
            3. Responde ÚNICAMENTE un JSON válido con este formato: {{"title": "...", "author": "..."}}
            
            TEXTO:
            {sample_text}
            """
            
            # Invocamos al modelo nativo de GenAI
            meta_response = genai_client.models.generate_content(
                model=llm_model_name,
                contents=prompt_meta
            )
            
            # Limpiamos la respuesta nativa para obtener solo el JSON
            json_str = meta_response.text.replace("```json", "").replace("```", "").strip()
            metadata_extracted = json.loads(json_str)
            
            doc_title = metadata_extracted.get("title", filename)
            doc_author = metadata_extracted.get("author", "Autor Desconocido")
            
            logger.info(f"METADATOS EXTRAÍDOS: Título='{doc_title}', Autor='{doc_author}'")
            
        except Exception as e:
            logger.warning(f"No se pudieron extraer metadatos con IA, usando defaults: {e}")
        # ----------------------------------------------------

        # 2. PROCESAMIENTO (SPLITTING)
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(md_header_splits)
        
        # Preparar documentos para Firestore CON LOS NUEVOS METADATOS
        documents = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata.copy()
            meta.update({
                "source": filename,
                "title": doc_title,   # <--- AQUI GUARDAMOS EL TÍTULO
                "author": doc_author, # <--- AQUI GUARDAMOS EL AUTOR
                "chunk_index": i,
                "model": "gemini-embedding-001"
            })
            
            doc = Document(page_content=chunk.page_content, metadata=meta)
            documents.append(doc)
        
        # 3. GUARDAR
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, embedding_service=embedding_model, client=firestore_client
        )
        
        batch_size = 50 
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Lote {i//batch_size + 1} guardado.")
        
        return {"status": "ok", "message": f"Archivo procesado: {doc_title} por {doc_author}"}
        
    except Exception as e:
        logger.error(f"Error procesando Texto/MD: {e}", exc_info=True)
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
        file_id = event.get("name") # ej: "carpeta/documento.md"
        
        if not bucket_name or not file_id: return "Evento ignorado", 200

        # FILTRO: Solo procesar .txt o .md (opcional)
        if not (file_id.endswith(".txt") or file_id.endswith(".md")):
            logger.info(f"Archivo {file_id} ignorado (no es txt/md).")
            return "Formato no soportado", 200

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        
        if not blob.exists() or blob.size == 0: return "Archivo inválido", 200

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_file.close()
            try:
                # Llamamos a la nueva función de texto
                result = _process_and_embed_text_file(temp_file.name, file_id)
            finally:
                if os.path.exists(temp_file.name): os.unlink(temp_file.name)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error handler: {e}", exc_info=True)
        return f"Error: {str(e)}", 500

@app.route("/query", methods=["POST"])
def query_rag_handler():
    try:
        request_data = request.get_json()
        if not request_data or "query" not in request_data:
             return jsonify({"error": "Falta query"}), 400
             
        user_query = request_data["query"]
        clients_local = get_clients()
        
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, 
            embedding_service=clients_local.get('embedding'), 
            client=clients_local.get('firestore')
        )
        
        logger.info(f"Buscando documentos para: '{user_query}'")
        found_docs = vector_store.similarity_search(query=user_query, k=5)
        
        results = []
        for i, doc in enumerate(found_docs):
            # Obtenemos la metadata cruda
            raw_meta = doc.metadata
            
            # --- CORRECCIÓN DE ANIDAMIENTO ---
            # A veces Firestore devuelve los campos directamente, a veces dentro de una llave 'metadata'.
            # Verificamos si existe un diccionario interno llamado 'metadata' y lo usamos.
            inner_meta = raw_meta.get("metadata", {})
            if isinstance(inner_meta, dict) and inner_meta:
                # Si hay datos anidados, damos prioridad a esos
                data_source = inner_meta
            else:
                # Si no, usamos el nivel superior
                data_source = raw_meta
            
            # Ahora extraemos los datos de la fuente correcta
            doc_source = data_source.get("source", "Desconocido")
            doc_title = data_source.get("title", data_source.get("Title", doc_source))
            doc_author = data_source.get("author", data_source.get("Author", "Autor Desconocido"))
            # ----------------------------------

            results.append({
                "source": doc_source,
                "content": doc.page_content,
                "title": doc_title,
                "author": doc_author
            })
        
        return jsonify({
            "results": results, 
            "count": len(results)
        }), 200

    except Exception as e:
        logger.error(f"Error query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
