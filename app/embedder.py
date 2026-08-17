"""Vectorisation des chunks et stockage dans ChromaDB."""

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Retourne le modèle d'embeddings HuggingFace configuré."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )


def embed_document(chunks: List[Document]) -> Chroma:
    """Vectorise une liste de chunks et les stocke dans le vector store.

    Args:
        chunks: documents LangChain à indexer.

    Raises:
        ValueError: si ``chunks`` est vide.

    Returns:
        Le vector store Chroma persistant.
    """
    if not chunks:
        raise ValueError("Aucun chunk à indexer.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_PATH,
    )
    return vector_store
