import os
import io
import json
import traceback
import logging
import tempfile
import warnings
import re 

# Filtramos advertencias de librerías
warnings.filterwarnings("ignore", "Support for google-cloud-storage", category=FutureWarning)

from flask import Flask, request, jsonify
from typing import Dict, Any, List

# LangChain y Google
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_firestore import FirestoreVectorStore
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# SDK GenAI unificado
from google import genai
from google.genai.types import EmbedContentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
clients = {}

COLLECTION_NAME = "pida_kb_genai-v20" 

# --- CLASE CUSTOM MIGRADA AL NUEVO SDK ---
class CustomGeminiEmbeddings(Embeddings):
    def __init__(self, model_name="gemini-embedding-001", dimensionality=2048, project=None, location=None):
        self.model_name = model_name
        self.dimensionality = dimensionality
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        config = EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.dimensionality
        )
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=config
            )
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
            contents=text, 
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
            
            # Usamos tu configuración superior comprobada
            clients['embedding'] = CustomGeminiEmbeddings(
                model_name="gemini-embedding-001",
                dimensionality=2048,
                project=PROJECT_ID,
                location=VERTEX_AI_LOCATION
            )
            
            MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            logger.info(f"Usando modelo LLM nativo: {MODEL_NAME}")
            
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
        
        docs_ref = firestore_client.collection(COLLECTION_NAME)
        existing_docs = docs_ref.where(filter=FieldFilter("metadata.source", "==", filename)).limit(1).stream()
        if len(list(existing_docs)) > 0:
            logger.warning(f"El archivo {filename} ya existe. Saltando...")
            return {"status": "skipped", "message": "Archivo ya existe en la base de datos."}

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text_content = f.read()
        
        if not text_content:
            return {"status": "error", "reason": "El archivo está vacío."}

        # 1. EXTRACCIÓN DE METADATOS 
        doc_title = filename
        doc_author = "Desconocido"
        
        try:
            sample_text = text_content[:3000]
            prompt_meta = f"""Eres un bibliotecario experto. Analiza el siguiente fragmento de texto y extrae el Título y el Autor.
            Reglas:
            1. Si no encuentras el autor explícitamente, pon "Autor Desconocido".
            2. Si no encuentras el título claro, usa: "{filename}".
            3. Responde ÚNICAMENTE un JSON válido con este formato: {{"title": "...", "author": "..."}}
            TEXTO:
            {sample_text}
            """
            
            meta_response = genai_client.models.generate_content(
                model=llm_model_name,
                contents=prompt_meta
            )
            
            json_str = meta_response.text.replace("```json", "").replace("```", "").strip()
            metadata_extracted = json.loads(json_str)
            
            doc_title = metadata_extracted.get("title", filename)
            doc_author = metadata_extracted.get("author", "Autor Desconocido")
            logger.info(f"METADATOS EXTRAÍDOS: Título='{doc_title}', Autor='{doc_author}'")
            
        except Exception as e:
            logger.warning(f"No se pudieron extraer metadatos con IA, usando defaults: {e}")

        # 2. PROCESAMIENTO INTELIGENTE (REGEX SPLITTING CORTE IDH)
        logger.info("Aplicando Regex Chunking para sentencias legales...")
        
        patron_parrafo = r'\n(?=\d{1,4}\.\s)'
        fragmentos_crudos = re.split(patron_parrafo, text_content)
        
        documents = []
        contexto_actual_h1 = "Sin Título Principal"
        contexto_actual_h2 = "Sin Capítulo"
        contexto_actual_h3 = "Sin Subsección"
        chunk_index = 0
        
        for fragmento in fragmentos_crudos:
            fragmento = fragmento.strip()
            if not fragmento: 
                continue
                
            lineas = fragmento.split('\n')
            for linea in lineas:
                linea_limpia = linea.strip()
                if linea_limpia.startswith('# '):
                    contexto_actual_h1 = linea_limpia.replace('# ', '').strip()
                elif linea_limpia.startswith('## '):
                    contexto_actual_h2 = linea_limpia.replace('## ', '').strip()
                    contexto_actual_h3 = "" 
                elif linea_limpia.startswith('### '):
                    contexto_actual_h3 = linea_limpia.replace('### ', '').strip()
                    
            if len(fragmento) < 40 and not re.match(r'^\d{1,4}\.', fragmento):
                continue
                
            match_num = re.match(r'^(\d{1,4})\.', fragmento)
            num_parrafo = int(match_num.group(1)) if match_num else None
            
            meta = {
                "source": filename,
                "title": doc_title,   
                "author": doc_author, 
                "seccion_h1": contexto_actual_h1,
                "seccion_h2": contexto_actual_h2,
                "subseccion_h3": contexto_actual_h3,
                "numero_parrafo": num_parrafo,
                "chunk_index": chunk_index,
                "model": "gemini-embedding-001"
            }
            
            doc = Document(page_content=fragmento, metadata=meta)
            documents.append(doc)
            chunk_index += 1
        
        # 3. GUARDAR VECTORES EN FIRESTORE
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, embedding_service=embedding_model, client=firestore_client
        )
        
        batch_size = 50 
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Lote {i//batch_size + 1} guardado.")
            
        # 4. GUARDAR EL REGISTRO EN EL CATÁLOGO GLOBAL
        try:
            safe_id = re.sub(r'[^a-zA-Z0-9]', '_', doc_title)[:150]
            catalog_ref = firestore_client.collection("library_registry").document(safe_id)
            catalog_ref.set({
                "title": doc_title,
                "author": doc_author,
                "total_chunks": len(documents)
            }, merge=True)
            logger.info(f"¡Éxito! Libro registrado en library_registry con ID: {safe_id}")
        except Exception as cat_err:
            logger.error(f"Error guardando el registro en library_registry: {cat_err}")
        
        return {"status": "ok", "message": f"Archivo procesado: {doc_title} por {doc_author} ({len(documents)} vectores)"}
        
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
        file_id = event.get("name") 
        
        if not bucket_name or not file_id: return "Evento ignorado", 200

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
            raw_meta = doc.metadata
            
            inner_meta = raw_meta.get("metadata", {})
            if isinstance(inner_meta, dict) and inner_meta:
                data_source = inner_meta
            else:
                data_source = raw_meta
            
            doc_source = data_source.get("source", "Desconocido")
            doc_title = data_source.get("title", data_source.get("Title", doc_source))
            doc_author = data_source.get("author", data_source.get("Author", "Autor Desconocido"))

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
