"""Découpage des documents en chunks selon plusieurs stratégies."""

from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from config import CHUNK_OVERLAP, CHUNK_SIZE


def _build_splitters(chunk_size: int, chunk_overlap: int) -> Dict[str, object]:
    """Construit les différents splitters disponibles."""
    return {
        "recursive": RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        ),
        "character": CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        "token": TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    }


def chunk_documents(
    docs: List[Document],
    strategy: str = "recursive",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Découpe une liste de documents LangChain en chunks.

    Args:
        docs: documents LangChain à découper.
        strategy: stratégie de découpage ("recursive", "character" ou "token").
        chunk_size: taille maximale d'un chunk.
        chunk_overlap: chevauchement entre chunks consécutifs.

    Raises:
        ValueError: si la stratégie est inconnue ou si aucun chunk n'est produit.

    Returns:
        Les chunks (documents LangChain) enrichis des métadonnées ``index``,
        ``strategy`` et ``chunk_size``.
    """
    splitters = _build_splitters(chunk_size, chunk_overlap)

    if strategy not in splitters:
        raise ValueError(f"Stratégie de chunking non supportée : {strategy}")

    chunks = splitters[strategy].split_documents(docs)

    if not chunks:
        raise ValueError("Le document fourni est vide.")

    for i, chunk in enumerate(chunks):
        chunk.metadata["index"] = i
        chunk.metadata["strategy"] = strategy
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"{len(docs)} document(s) transformé(s) en {len(chunks)} chunk(s).")
    return chunks
