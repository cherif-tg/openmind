"""Re-ranking des documents récupérés avec un cross-encoder local."""

from typing import List, Optional

from langchain_core.documents import Document

from config import RERANK_MODEL

_reranker = None


def get_reranker():
    """Retourne un cross-encoder partagé (chargé une seule fois).

    Le modèle est chargé paresseusement pour ne pas pénaliser le démarrage.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """Réordonne les documents par pertinence vis-à-vis de la requête.

    Args:
        query: question utilisateur.
        documents: candidats issus de la recherche vectorielle.
        top_k: nombre de documents à conserver (défaut : tous).

    Returns:
        Documents triés par score décroissant, enrichis du champ
        ``rerank_score`` dans leurs métadonnées.
    """
    if not documents:
        return []

    model = get_reranker()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored = list(zip(scores, documents))
    scored.sort(key=lambda item: item[0], reverse=True)

    ranked: List[Document] = []
    for score, doc in scored:
        doc.metadata["rerank_score"] = float(score)
        ranked.append(doc)

    if top_k:
        ranked = ranked[:top_k]

    return ranked
