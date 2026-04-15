"""
Router pour les requêtes RAG
"""
import logging
import os

from fastapi import APIRouter, HTTPException
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.retriever import get_retriever
from app.llm_factory import get_llm
from config import LLM_MODE, TOP_K
from api.schemas.query import QueryRequest, QueryResponse, Source

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Pose une question sur les documents indexés.

    Retourne la réponse générée par le LLM ainsi que les sources utilisées.
    """
    try:
        # Déterminer le mode LLM
        llm_mode = request.llm_mode or LLM_MODE or os.getenv("LLM_MODE", "groq")

        # Valider le mode
        valid_modes = {"groq", "ollama", "huggingface"}
        if llm_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Mode LLM invalide : {llm_mode}. Modes valides : {valid_modes}"
            )

        logger.info(f"Requête RAG : '{request.question}' (mode: {llm_mode}, top_k: {request.top_k})")

        # Initialiser le retriever avec le top_k demandé
        retriever = get_retriever()
        # Note: pour changer dynamiquement le top_k, il faudrait modifier get_retriever()
        # Pour l'instant, on utilise le TOP_K de config

        # Récupérer les documents pertinents
        relevant_docs = retriever.invoke(request.question)

        if not relevant_docs:
            logger.warning("Aucun document pertinent trouvé")
            return QueryResponse(
                answer="Je n'ai pas trouvé de documents pertinents dans la base de connaissances pour répondre à cette question.",
                sources=[],
                llm_mode=llm_mode
            )

        # Formater le contexte
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Créer le prompt template
        rag_prompt = PromptTemplate(
            template="""Tu es un assistant de recherche expert. Utilise UNIQUEMENT les informations suivantes pour répondre à la question de manière précise et concise. Si les informations ne sont pas suffisantes, dis-le clairement.

Contexte:
{context}

Question: {question}

Réponse:""",
            input_variables=["context", "question"]
        )

        # Initialiser le LLM
        llm = get_llm(mode=llm_mode)

        # Créer et exécuter la chaîne RAG
        chain = rag_prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": request.question})

        # Formater les sources
        sources = []
        seen = set()

        for doc in relevant_docs:
            source_id = (
                doc.metadata.get("filename", "unknown"),
                doc.metadata.get("index", 0)
            )

            if source_id not in seen:
                seen.add(source_id)
                sources.append(Source(
                    content=doc.page_content[:500],  # Limiter la longueur
                    metadata={
                        "source": doc.metadata.get("filename", "unknown"),
                        "chunk": doc.metadata.get("index", 0),
                        "chunk_size": doc.metadata.get("chunk_size", len(doc.page_content))
                    }
                ))

        logger.info(f"Réponse générée avec {len(sources)} sources")

        return QueryResponse(
            answer=response,
            sources=sources,
            llm_mode=llm_mode
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la requête RAG : {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse : {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Vérifie que l'endpoint de query est opérationnel.
    """
    return {"status": "healthy", "service": "query"}
