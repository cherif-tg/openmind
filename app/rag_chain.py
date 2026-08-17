"""Pipeline RAG complet : retrieval -> re-ranking -> génération avec citations.

Ce module est la source de vérité du prompt RAG : l'API, l'interface et la
démo passent par ici, ce qui évite toute duplication.
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.llm_factory import get_llm
from app.retriever import get_retriever
from config import RERANK_ENABLED, RETRIEVAL_K, TOP_K

RAG_PROMPT = PromptTemplate(
    template=(
        "Tu es un assistant de recherche expert. Réponds à la question en te basant "
        "UNIQUEMENT sur le contexte fourni ci-dessous. Chaque passage est numéroté "
        "entre crochets, par exemple [1], [2], etc. Cite tes sources en insérant les "
        "numéros correspondants à la fin des phrases concernées. Si le contexte ne "
        "contient pas la réponse, dis-le clairement sans rien inventer.\n\n"
        "Contexte:\n{context}\n\n"
        "Question: {question}\n\n"
        "Réponse (avec citations [n]):"
    ),
    input_variables=["context", "question"],
)


def build_context(documents: List[Document]) -> str:
    """Formate les documents en contexte numéroté pour le prompt."""
    parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, start=1)]
    return "\n\n".join(parts)


def _rerank(query: str, documents: List[Document], top_k: int) -> List[Document]:
    """Re-ranking avec repli silencieux sur la troncature simple."""
    try:
        from app.reranker import rerank_documents

        return rerank_documents(query, documents, top_k=top_k)
    except Exception:  # noqa: BLE001 — dégradation gracieuse si modèle absent
        return documents[:top_k]


def build_rag_chain(
    question: str,
    top_k: int = TOP_K,
    llm_mode: Optional[str] = None,
    rerank: Optional[bool] = None,
    retriever=None,
) -> Tuple[str, List[Document]]:
    """Exécute le pipeline RAG complet.

    Args:
        question: question utilisateur.
        top_k: nombre de chunks finaux utilisés pour la génération.
        llm_mode: mode LLM ("groq", "ollama", "huggingface"). Défaut : config.
        rerank: active/désactive le re-ranking. Défaut : valeur de config.
        retriever: retriever optionnel (défaut : construit depuis le vector store).

    Returns:
        Tuple ``(réponse, documents utilisés)``. Les documents sont numérotés
        dans le même ordre que les citations [n] de la réponse.
    """
    if rerank is None:
        rerank = RERANK_ENABLED

    # On récupère plus de candidats quand le re-ranking est actif, pour
    # laisser de la marge au cross-encoder avant de tronquer à top_k.
    fetch_k = RETRIEVAL_K if rerank else top_k

    if retriever is None:
        retriever = get_retriever(top_k=fetch_k)

    documents = retriever.invoke(question)

    if not documents:
        return (
            "Je n'ai pas trouvé de documents pertinents pour répondre à cette question.",
            [],
        )

    if rerank and len(documents) > 1:
        documents = _rerank(question, documents, top_k)
    else:
        documents = documents[:top_k]

    context = build_context(documents)
    llm = get_llm(mode=llm_mode)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, documents
