"""Router pour la gestion des documents (upload, liste, suppression)."""

import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.schemas.document import (
    DocumentDelete,
    DocumentInfo,
    DocumentList,
    DocumentUpload,
)
from app.chunker import chunk_documents
from app.embedder import embed_document
from app.ingestion import SUPPORTED_EXTENSIONS, load_document
from app.retriever import load_vectorstore, remove_by_filename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=List[DocumentUpload])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload et indexe un ou plusieurs documents.

    Formats supportés : PDF, CSV, DOCX, TXT, Markdown, HTML.

    Si un fichier portant le même nom est déjà indexé, ses anciens chunks
    sont remplacés (pas de doublons).
    """
    results = []

    for file in files:
        ext = Path(file.filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté : {ext}. Formats acceptés : {sorted(SUPPORTED_EXTENSIONS)}",
            )

        # Sauvegarde temporaire du fichier uploadé.
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_document(tmp_path)
            chunks = chunk_documents(docs, strategy="recursive")

            # Remplace les chunks existants portant le même nom de fichier.
            removed = remove_by_filename(file.filename)
            if removed:
                logger.info(f"Document '{file.filename}' : {removed} ancien(s) chunk(s) remplacé(s)")

            embed_document(chunks)

            results.append(
                DocumentUpload(
                    filename=file.filename,
                    chunks_count=len(chunks),
                    status="indexed",
                )
            )

            logger.info(f"Document '{file.filename}' indexé : {len(chunks)} chunks")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'upload de '{file.filename}': {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {str(e)}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return results


@router.get("/", response_model=DocumentList)
async def list_documents():
    """Liste tous les documents indexés dans le vector store."""
    try:
        vectorstore = load_vectorstore()
        collection = vectorstore._collection

        docs = collection.get(include=["metadatas"])

        if not docs or not docs.get("metadatas"):
            return DocumentList(documents=[], total=0)

        file_chunks: dict = {}
        for metadata in docs["metadatas"]:
            filename = metadata.get("filename", "unknown")
            file_chunks[filename] = file_chunks.get(filename, 0) + 1

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
    """Supprime un document et tous ses chunks du vector store.

    Attention : cette opération est irréversible.
    """
    try:
        deleted_count = remove_by_filename(filename)

        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun chunk trouvé pour le fichier '{filename}'",
            )

        logger.info(f"Document '{filename}' supprimé : {deleted_count} chunks retirés")

        return DocumentDelete(
            filename=filename,
            status="deleted",
            message=f"{deleted_count} chunks supprimés",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de '{filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression : {str(e)}")
