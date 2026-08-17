"""Recherche sémantique dans le vector store ChromaDB."""

from langchain_chroma import Chroma

from app.embedder import get_embeddings
from config import COLLECTION_NAME, TOP_K, VECTORSTORE_PATH


def get_retriever(top_k: int = None):
    """Retourne le retriever courant avec un ``top_k`` configurable.

    Args:
        top_k: nombre de documents à récupérer. Défaut : ``TOP_K`` de config.
    """
    if top_k is None:
        top_k = TOP_K

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_PATH,
        embedding_function=get_embeddings(),
    )
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def load_vectorstore() -> Chroma:
    """Charge le vector store persistant depuis le disque."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_PATH,
        embedding_function=get_embeddings(),
    )


def remove_by_filename(filename: str) -> int:
    """Supprime tous les chunks associés à un fichier.

    Utilisé pour éviter les doublons lors d'un ré-upload.

    Args:
        filename: nom du fichier (métadonnée ``filename`` des chunks).

    Returns:
        Nombre de chunks supprimés.
    """
    vector_store = load_vectorstore()
    collection = vector_store._collection

    existing = collection.get(where={"filename": filename}, include=[])
    ids = existing.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return len(ids)
