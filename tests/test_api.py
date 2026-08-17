"""Tests unitaires pour l'API FastAPI (hermétiques, sans modèle ni réseau)."""

import pytest
from langchain_core.documents import Document


class TestAPIHealth:
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "name" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestDocumentsAPI:
    def test_upload_documents(self, client, test_file_txt):
        with open(test_file_txt, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"files": ("test.txt", f, "text/plain")},
            )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert data[0]["filename"] == "test.txt"
        assert data[0]["status"] == "indexed"
        assert data[0]["chunks_count"] >= 1

    def test_upload_multiple_documents(self, client, test_file_txt, test_file_csv):
        with open(test_file_txt, "rb") as f_txt, open(test_file_csv, "rb") as f_csv:
            files = [
                ("files", ("test.txt", f_txt, "text/plain")),
                ("files", ("test.csv", f_csv, "text/csv")),
            ]
            response = client.post("/api/documents/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        filenames = [doc["filename"] for doc in data]
        assert "test.txt" in filenames
        assert "test.csv" in filenames

    def test_upload_unsupported_format(self, client, tmp_path):
        file_path = tmp_path / "test.xyz"
        file_path.write_text("contenu", encoding="utf-8")
        with open(file_path, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"files": ("test.xyz", f, "application/octet-stream")},
            )
        assert response.status_code == 400
        assert "non supporté" in response.json()["detail"]

    def test_list_documents(self, client):
        response = client.get("/api/documents/")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    def test_delete_document(self, client):
        response = client.delete("/api/documents/test.txt")
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["status"] == "deleted"

    def test_delete_nonexistent_document(self, client):
        response = client.delete("/api/documents/inexistant.txt")
        assert response.status_code == 404


class TestQueryAPI:
    def test_query_simple(self, client):
        response = client.post("/api/query/", json={"question": "De quoi parle le document ?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "llm_mode" in data
        assert isinstance(data["sources"], list)

    def test_query_sources_include_page(self, client):
        response = client.post("/api/query/", json={"question": "test"})
        data = response.json()
        assert data["sources"][0]["metadata"]["page"] == 1

    def test_query_with_custom_top_k(self, client):
        response = client.post("/api/query/", json={"question": "test", "top_k": 3})
        assert response.status_code == 200

    def test_query_with_llm_mode(self, client):
        response = client.post("/api/query/", json={"question": "test", "llm_mode": "ollama"})
        assert response.status_code == 200
        assert response.json()["llm_mode"] == "ollama"

    def test_query_invalid_llm_mode(self, client):
        response = client.post("/api/query/", json={"question": "test", "llm_mode": "invalid"})
        assert response.status_code == 400

    def test_query_empty_question(self, client):
        response = client.post("/api/query/", json={"question": ""})
        assert response.status_code == 422

    def test_query_health(self, client):
        response = client.get("/api/query/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.fixture
def client(mocker):
    """Client HTTP hermétique : mocks le vector store et le pipeline RAG."""
    from fastapi.testclient import TestClient

    from api.main import app

    fake_collection = mocker.MagicMock()
    fake_collection.get.return_value = {"ids": [], "metadatas": []}
    fake_vs = mocker.MagicMock()
    fake_vs._collection = fake_collection

    mocker.patch("api.routers.documents.embed_document")
    mocker.patch("api.routers.documents.load_vectorstore", return_value=fake_vs)

    def fake_remove(filename):
        return 1 if filename == "test.txt" else 0

    mocker.patch("api.routers.documents.remove_by_filename", side_effect=fake_remove)

    mocker.patch(
        "api.routers.query.build_rag_chain",
        return_value=(
            "Réponse test [1]",
            [
                Document(
                    page_content="Le RAG est une technique.",
                    metadata={"filename": "test.txt", "index": 0, "page": 1, "chunk_size": 25},
                )
            ],
        ),
    )

    with TestClient(app) as test_client:
        yield test_client
