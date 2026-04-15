"""
Fixtures pytest partagées pour tous les tests
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


@pytest.fixture(scope="session")
def sample_documents():
    """
    Fixture: Liste de documents de test
    """
    return [
        Document(
            page_content="Le RAG (Retrieval-Augmented Generation) est une technique "
                        "qui combine la récupération d'informations et la génération de texte.",
            metadata={"source": "test_rag.pdf", "page": 1}
        ),
        Document(
            page_content="Les embeddings sont des représentations vectorielles de texte. "
                        "Ils permettent de comparer la similarité sémantique entre documents.",
            metadata={"source": "test_embeddings.pdf", "page": 1}
        ),
        Document(
            page_content="ChromaDB est une base de données vectorielle open-source. "
                        "Elle stocke les embeddings et permet la recherche de similarité.",
            metadata={"source": "test_chroma.pdf", "page": 1}
        ),
    ]


@pytest.fixture(scope="session")
def sample_chunks(sample_documents):
    """
    Fixture: Chunks de test dérivés des documents
    """
    chunks = []
    for i, doc in enumerate(sample_documents):
        chunk = Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "index": i,
                "chunk_size": len(doc.page_content)
            }
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def temp_vectorstore_dir():
    """
    Fixture: Répertoire temporaire pour ChromaDB
    Nettoie automatiquement après chaque test
    """
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def embedding_model():
    """
    Fixture: Modèle d'embedding pour les tests
    Utilise un modèle léger pour des tests rapides
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


@pytest.fixture
def vectorstore(temp_vectorstore_dir, embedding_model, sample_chunks):
    """
    Fixture: Vectorstore ChromaDB initialisé avec des chunks de test
    """
    vs = Chroma.from_documents(
        documents=sample_chunks,
        embedding=embedding_model,
        persist_directory=temp_vectorstore_dir,
        collection_name="test_collection"
    )
    return vs


@pytest.fixture
def test_file_txt(tmp_path):
    """
    Fixture: Fichier texte de test
    """
    content = """
    Le RAG est une technique puissante.
    Il permet de répondre à des questions sur des documents.
    La retrieval est suivie de la génération.
    """
    file_path = tmp_path / "test.txt"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def test_file_csv(tmp_path):
    """
    Fixture: Fichier CSV de test
    """
    content = """name,description
    RAG,Retrieval-Augmented Generation
    LLM,Large Language Model
    embedding,représentation vectorielle
    """
    file_path = tmp_path / "test.csv"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)
