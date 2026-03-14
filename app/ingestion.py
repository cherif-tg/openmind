import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
)

# ── Extension → Loader mapping ────────────────────────────
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".csv":  CSVLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".html": UnstructuredHTMLLoader,
}

def load_document(file_path: str) -> List[Document]:
    """
    Charge un seul fichier selon son extension.
    Retourne une liste de Documents LangChain.
    """
    ext = Path(file_path).suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(f"Format non supporté : {ext}")

    loader_class = LOADER_MAP[ext]
    loader = loader_class(file_path)
    docs = loader.load()

    # Enrichir les métadonnées
    for doc in docs:
        doc.metadata["source"]    = file_path
        doc.metadata["file_type"] = ext
        doc.metadata["filename"]  = Path(file_path).name

    return docs


def load_folder(folder_path: str) -> List[Document]:
    """
    Charge tous les documents d'un dossier récursivement.
    """
    all_docs = []
    folder   = Path(folder_path)

    files = [f for f in folder.rglob("*") if f.suffix.lower() in LOADER_MAP]

    print(f"[OpenMind RAG] {len(files)} fichier(s) trouvé(s) dans '{folder_path}'")

    for file in files:
        try:
            docs = load_document(str(file))
            all_docs.extend(docs)
            print(f"  {file.name} — {len(docs)} page(s)/section(s)")
        except Exception as e:
            print(f"   {file.name} — Erreur : {e}")

    print(f"[OpenMind RAG] Total : {len(all_docs)} document(s) chargé(s)")
    return all_docs


def load_from_url(url: str) -> List[Document]:
    """
    Charge le contenu d'une page web.
    """
    loader = WebBaseLoader(url)
    docs   = loader.load()

    for doc in docs:
        doc.metadata["source"]    = url
        doc.metadata["file_type"] = "web"

    return docs