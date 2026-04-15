"""
Tests unitaires pour l'API FastAPI
Nécessite pytest-asyncio pour les tests asynchrones
"""
import pytest
import tempfile
import shutil
from pathlib import Path

# Configuration pour les tests async
pytestmark = pytest.mark.asyncio


class TestAPIHealth:
    """Tests pour les endpoints de santé"""

    async def test_root_endpoint(self, client):
        """Test l'endpoint racine"""
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "name" in data
        assert "endpoints" in data

    async def test_health_endpoint(self, client):
        """Test l'endpoint health"""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDocumentsAPI:
    """Tests pour les endpoints de documents"""

    async def test_upload_documents(self, client, test_file_txt):
        """Test l'upload d'un document"""
        with open(test_file_txt, "rb") as f:
            files = {"files": ("test.txt", f, "text/plain")}

            response = await client.post("/api/documents/upload", files=files)

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["filename"] == "test.txt"
        assert data[0]["status"] == "indexed"
        assert "chunks_count" in data[0]

    async def test_upload_multiple_documents(self, client, test_file_txt, test_file_csv):
        """Test l'upload de plusieurs documents"""
        with open(test_file_txt, "rb") as f_txt, \
             open(test_file_csv, "rb") as f_csv:

            files = [
                ("files", ("test.txt", f_txt, "text/plain")),
                ("files", ("test.csv", f_csv, "text/csv"))
            ]

            response = await client.post("/api/documents/upload", files=files)

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 2
        filenames = [doc["filename"] for doc in data]
        assert "test.txt" in filenames
        assert "test.csv" in filenames

    async def test_upload_unsupported_format(self, client, tmp_path):
        """Test l'upload d'un format non supporté"""
        # Créer un fichier avec extension non supportée
        file_path = tmp_path / "test.xyz"
        file_path.write_text("contenu")

        with open(file_path, "rb") as f:
            files = {"files": ("test.xyz", f, "application/octet-stream")}

            response = await client.post("/api/documents/upload", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "non supporté" in data["detail"]

    async def test_list_documents(self, client, test_file_txt):
        """Test la liste des documents"""
        # D'abord, uploader un document
        with open(test_file_txt, "rb") as f:
            await client.post("/api/documents/upload", files={"files": ("test.txt", f, "text/plain")})

        # Puis lister
        response = await client.get("/api/documents/")

        assert response.status_code == 200
        data = response.json()

        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    async def test_list_documents_empty(self, client):
        """Test la liste des documents quand vide"""
        response = await client.get("/api/documents/")

        assert response.status_code == 200
        data = response.json()

        assert data["documents"] == []
        assert data["total"] == 0

    async def test_delete_document(self, client, test_file_txt):
        """Test la suppression d'un document"""
        # D'abord, uploader un document
        with open(test_file_txt, "rb") as f:
            await client.post("/api/documents/upload", files={"files": ("test.txt", f, "text/plain")})

        # Puis supprimer
        response = await client.delete("/api/documents/test.txt")

        assert response.status_code == 200
        data = response.json()

        assert data["filename"] == "test.txt"
        assert data["status"] == "deleted"
        assert "message" in data

    async def test_delete_nonexistent_document(self, client):
        """Test la suppression d'un document inexistant"""
        response = await client.delete("/api/documents/inexistant.txt")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestQueryAPI:
    """Tests pour les endpoints de query"""

    async def test_query_simple(self, client, test_file_txt):
        """Test une requête simple"""
        # D'abord, indexer un document
        with open(test_file_txt, "rb") as f:
            await client.post("/api/documents/upload", files={"files": ("test.txt", f, "text/plain")})

        # Puis poser une question
        response = await client.post(
            "/api/query/",
            json={"question": "De quoi parle le document?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "llm_mode" in data
        assert isinstance(data["sources"], list)

    async def test_query_with_custom_top_k(self, client, test_file_txt):
        """Test une requête avec top_k personnalisé"""
        # Indexer un document
        with open(test_file_txt, "rb") as f:
            await client.post("/api/documents/upload", files={"files": ("test.txt", f, "text/plain")})

        # Question avec top_k=3
        response = await client.post(
            "/api/query/",
            json={"question": "Qu'est-ce que le RAG?", "top_k": 3}
        )

        assert response.status_code == 200

    async def test_query_with_llm_mode(self, client, test_file_txt):
        """Test une requête avec mode LLM spécifié"""
        # Indexer un document
        with open(test_file_txt, "rb") as f:
            await client.post("/api/documents/upload", files={"files": ("test.txt", f, "text/plain")})

        # Question avec mode ollama
        response = await client.post(
            "/api/query/",
            json={"question": "Qu'est-ce que le RAG?", "llm_mode": "ollama"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["llm_mode"] == "ollama"

    async def test_query_invalid_llm_mode(self, client):
        """Test une requête avec mode LLM invalide"""
        response = await client.post(
            "/api/query/",
            json={"question": "Test", "llm_mode": "invalid_mode"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    async def test_query_empty_question(self, client):
        """Test une requête avec question vide"""
        response = await client.post(
            "/api/query/",
            json={"question": ""}
        )

        # Pydantic devrait rejeter la question vide
        assert response.status_code == 422  # Validation error

    async def test_query_no_documents(self, client):
        """Test une requête sans documents indexés"""
        response = await client.post(
            "/api/query/",
            json={"question": "Quelle est la capitale de la France?"}
        )

        # Devrait retourner une réponse même sans documents
        assert response.status_code == 200
        data = response.json()

        # La réponse devrait indiquer qu'aucun document n'a été trouvé
        assert "answer" in data

    async def test_query_health(self, client):
        """Test l'endpoint de santé du query"""
        response = await client.get("/api/query/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "query"


# ── Fixtures spécifiques à l'API ──────────────────────────────

@pytest.fixture(scope="session")
def test_vectorstore_dir():
    """Fixture: Répertoire pour le vectorstore de test"""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def client(test_vectorstore_dir, mocker):
    """
    Fixture: Client HTTP pour tester l'API

    Mock les composants externes pour des tests rapides
    """
    from fastapi.testclient import TestClient
    from api.main import app

    # Mock de get_embeddings pour éviter de charger le modèle
    mock_embeddings = mocker.patch("app.embedder.get_embeddings")
    mock_embeddings.return_value = mocker.MagicMock()

    # Configurer le vectorstore pour utiliser le répertoire de test
    mocker.patch("app.embedder.VECTORSTORE_PATH", test_vectorstore_dir)
    mocker.patch("app.retriever.VECTORSTORE_PATH", test_vectorstore_dir)

    with TestClient(app) as test_client:
        yield test_client
