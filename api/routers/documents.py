"""
Router pour la gestion des documents
"""
import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.ingestion import load_document
from app.chunker import chunk_documents
from app.embedder import embed_document
from app.retriever import load_vectorstore
from config import VECTORSTORE_PATH, COLLECTION_NAME
from api.schemas.document import DocumentUpload, DocumentInfo, DocumentList, DocumentDelete

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=List[DocumentUpload])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload et indexe un ou plusieurs documents.

    Formats supportés : PDF, CSV, DOCX, TXT, HTML
    """
    results = []

    for file in files:
        try:
            # Vérifier l'extension
            ext = Path(file.filename).suffix.lower()
            valid_extensions = {".pdf", ".csv", ".docx", ".txt", ".html"}

            if ext not in valid_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Format non supporté : {ext}. Formats acceptés : {valid_extensions}"
                )

            # Sauvegarder temporairement
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Charger le document
                docs = load_document(tmp_path)

                # Chunking
                chunks = chunk_documents(docs, strategy="recursive")

                # Embedding et stockage
                embed_document(chunks)

                results.append(DocumentUpload(
                    filename=file.filename,
                    chunks_count=len(chunks),
                    status="indexed"
                ))

                logger.info(f"Document '{file.filename}' indexé : {len(chunks)} chunks")

            finally:
                # Nettoyer le fichier temporaire
                Path(tmp_path).unlink(missing_ok=True)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'upload de '{file.filename}': {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {str(e)}")

    return results


@router.get("/", response_model=DocumentList)
async def list_documents():
    """
    Liste tous les documents indexés dans le vectorstore.
    """
    try:
        vectorstore = load_vectorstore()
        collection = vectorstore._collection

        # Récupérer les métadonnées uniques
        docs = collection.get(include=["metadatas"])

        if not docs or not docs.get("metadatas"):
            return DocumentList(documents=[], total=0)

        # Compter les chunks par fichier
        file_chunks = {}
        for metadata in docs["metadatas"]:
            filename = metadata.get("filename", "unknown")
            if filename not in file_chunks:
                file_chunks[filename] = 0
            file_chunks[filename] += 1

        documents = [
            DocumentInfo(filename=name, chunks_count=count)
            for name, count in file_chunks.items()
        ]

        return DocumentList(documents=documents, total=len(documents))

    except Exception as e:
        logger.error(f"Erreur lors de la liste des documents : {e}")
        return DocumentList(documents=[], total=0)


@router.delete("/{filename}", response_model=DocumentDelete)
async def delete_document(filename: str):
    """
    Supprime un document et ses chunks du vectorstore.

    Attention : Cette opération est irréversible.
    """
    try:
        vectorstore = load_vectorstore()
        collection = vectorstore._collection

        # Récupérer tous les documents
        docs = collection.get(include=["metadatas"])

        if not docs or not docs.get("ids"):
            raise HTTPException(status_code=404, detail="Aucun document dans le vectorstore")

        # Trouver les IDs à supprimer
        ids_to_delete = []
        deleted_count = 0

        for i, metadata in enumerate(docs["metadatas"]):
            if metadata.get("filename") == filename:
                ids_to_delete.append(docs["ids"][i])
                deleted_count += 1

        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun chunk trouvé pour le fichier '{filename}'"
            )

        # Supprimer les chunks
        collection.delete(ids=ids_to_delete)

        logger.info(f"Document '{filename}' supprimé : {deleted_count} chunks retirés")

        return DocumentDelete(
            filename=filename,
            status="deleted",
            message=f"{deleted_count} chunks supprimés"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de '{filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression : {str(e)}")
