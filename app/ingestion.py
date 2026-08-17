"""Chargement de documents depuis fichiers et URL."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

# Extension -> Loader LangChain.
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".html": UnstructuredHTMLLoader,
}

# Extensions acceptées pour l'upload via l'API et l'interface.
SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys())

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


def _text_loader(file_path: str) -> TextLoader:
    """TextLoader forcé en UTF-8 pour gérer correctement les accents."""
    return TextLoader(file_path, encoding="utf-8")


def load_document(file_path: str) -> List[Document]:
    """Charge un fichier selon son extension.

    Args:
        file_path: chemin du fichier à charger.

    Raises:
        ValueError: si l'extension n'est pas supportée.

    Returns:
        Liste de documents LangChain, enrichis de métadonnées (source,
        file_type, filename).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    ext = path.suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(f"Format non supporté : {ext}")

    loader_class = LOADER_MAP[ext]
    if ext in _TEXT_EXTENSIONS:
        loader = _text_loader(file_path)
    else:
        loader = loader_class(file_path)

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file_path
        doc.metadata["file_type"] = ext
        doc.metadata["filename"] = Path(file_path).name

    return docs


def load_folder(folder_path: str) -> List[Document]:
    """Charge récursivement tous les documents supportés d'un dossier."""
    all_docs: List[Document] = []
    folder = Path(folder_path)

    files = [f for f in folder.rglob("*") if f.suffix.lower() in LOADER_MAP]

    print(f"[OpenMind RAG] {len(files)} fichier(s) trouvé(s) dans '{folder_path}'")

    for file in files:
        try:
            docs = load_document(str(file))
            all_docs.extend(docs)
            print(f"  {file.name} - {len(docs)} page(s)/section(s)")
        except Exception as e:  # noqa: BLE001 — on log et on continue
            print(f"  {file.name} - Erreur : {e}")

    print(f"[OpenMind RAG] Total : {len(all_docs)} document(s) chargé(s)")
    return all_docs


def load_from_url(url: str) -> List[Document]:
    """Charge le contenu d'une page web."""
    loader = WebBaseLoader(url)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["file_type"] = "web"

    return docs
