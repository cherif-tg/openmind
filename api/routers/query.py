"""Router pour les requêtes RAG."""

import logging
import os

from fastapi import APIRouter, HTTPException

from api.schemas.query import QueryRequest, QueryResponse, Source
from app.rag_chain import build_rag_chain
from config import LLM_MODE, TOP_K

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])

VALID_MODES = {"groq", "ollama", "huggingface"}


def _build_sources(documents) -> list:
    """Transforme les documents récupérés en sources avec métadonnées."""
    sources = []
    for doc in documents:
        sources.append(
            Source(
                content=doc.page_content[:500],
                metadata={
                    "source": doc.metadata.get("filename", "unknown"),
                    "page": doc.metadata.get("page"),
                    "chunk": doc.metadata.get("index", 0),
                    "chunk_size": doc.metadata.get("chunk_size", len(doc.page_content)),
                    "rerank_score": doc.metadata.get("rerank_score"),
                },
            )
        )
    return sources


@router.post("/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Pose une question sur les documents indexés.

    Retourne la réponse générée par le LLM (avec citations [n]) ainsi que
    les sources correspondantes, dans l'ordre des numéros de citation.
    """
    llm_mode = request.llm_mode or LLM_MODE or os.getenv("LLM_MODE", "groq")

    if llm_mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Mode LLM invalide : {llm_mode}. Modes valides : {sorted(VALID_MODES)}",
        )

    top_k = request.top_k or TOP_K

    logger.info(
        f"Requête RAG : '{request.question}' (mode: {llm_mode}, top_k: {top_k}, rerank: {request.rerank})"
    )

    try:
        answer, documents = build_rag_chain(
            request.question,
            top_k=top_k,
            llm_mode=llm_mode,
            rerank=request.rerank,
        )
    except Exception as e:
        logger.error(f"Erreur lors de la requête RAG : {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse : {str(e)}",
        )

    sources = _build_sources(documents)

    logger.info(f"Réponse générée avec {len(sources)} source(s)")

    return QueryResponse(answer=answer, sources=sources, llm_mode=llm_mode)


@router.get("/health")
async def health_check():
    """Vérifie que l'endpoint de query est opérationnel."""
    return {"status": "healthy", "service": "query"}
