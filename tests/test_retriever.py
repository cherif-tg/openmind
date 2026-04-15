"""
Tests unitaires pour le module de retrieval (app/retriever.py)
"""
import pytest
from unittest.mock import patch, MagicMock

from app.retriever import get_retriever, load_vectorstore


class TestGetRetriever:
    """Tests pour la fonction get_retriever"""

    def test_get_retriever_returns_retriever(self, mocker):
        """Test que get_retriever retourne un retriever valide"""
        # Mock de get_embeddings pour éviter de charger le modèle
        mock_embeddings = mocker.patch("app.embedder.get_embeddings")
        mock_embeddings.return_value = MagicMock()

        # Mock de Chroma
        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        retriever = get_retriever()

        assert retriever is not None
        assert mock_vectorstore.as_retriever.called

    def test_get_retriever_uses_config(self, mocker):
        """Test que get_retriever utilise la configuration"""
        mock_embeddings = mocker.patch("app.embedder.get_embeddings")
        mock_embeddings.return_value = MagicMock()

        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        from config import COLLECTION_NAME, VECTORSTORE_PATH, TOP_K

        get_retriever()

        # Vérifier que Chroma est appelé avec les bons paramètres
        mock_chroma.assert_called_once()
        call_kwargs = mock_chroma.call_args[1]

        assert call_kwargs["collection_name"] == COLLECTION_NAME
        assert call_kwargs["persist_directory"] == VECTORSTORE_PATH

    def test_get_retriever_search_kwargs(self, mocker):
        """Test que le retriever utilise TOP_K pour la recherche"""
        from config import TOP_K

        mock_embeddings = mocker.patch("app.embedder.get_embeddings")
        mock_embeddings.return_value = MagicMock()

        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        get_retriever()

        # Vérifier que search_kwargs contient k=TOP_K
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_kwargs={"k": TOP_K}
        )


class TestLoadVectorstore:
    """Tests pour la fonction load_vectorstore"""

    def test_load_vectorstore_returns_chroma(self, mocker):
        """Test que load_vectorstore retourne un objet Chroma"""
        mock_embeddings = mocker.patch("app.embedder.get_embeddings")
        mock_embeddings.return_value = MagicMock()

        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore

        result = load_vectorstore()

        assert result is not None
        assert mock_chroma.called

    def test_load_vectorstore_uses_config(self, mocker):
        """Test que load_vectorstore utilise la configuration"""
        from config import COLLECTION_NAME, VECTORSTORE_PATH

        mock_embeddings = mocker.patch("app.embedder.get_embeddings")
        mock_embeddings.return_value = MagicMock()

        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore

        load_vectorstore()

        mock_chroma.assert_called_once()
        call_kwargs = mock_chroma.call_args[1]

        assert call_kwargs["collection_name"] == COLLECTION_NAME
        assert call_kwargs["persist_directory"] == VECTORSTORE_PATH


class TestRetrieverIntegration:
    """Tests d'intégration pour le retriever (avec vrai vectorstore)"""

    def test_retriever_search_with_real_vectorstore(self, vectorstore, mocker):
        """Test de recherche avec un vrai vectorstore"""
        # Mock pour utiliser notre vectorstore de test
        mock_embeddings = mocker.patch("app.embedder.get_embeddings")
        mock_embeddings.return_value = vectorstore._embedding_function

        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_chroma.return_value = vectorstore

        retriever = get_retriever()

        # Tester la recherche
        results = retriever.invoke("Quelle est la technique RAG?")

        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0
        assert hasattr(results[0], "page_content")
        assert hasattr(results[0], "metadata")

    def test_retriever_relevant_results(self, vectorstore, mocker):
        """Test que le retriever retourne des résultats pertinents"""
        mock_embeddings = mocker.patch("app.embedder.get_embeddings")
        mock_embeddings.return_value = vectorstore._embedding_function

        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_chroma.return_value = vectorstore

        retriever = get_retriever()

        # Recherche sur le sujet des embeddings
        results = retriever.invoke("Comment fonctionnent les embeddings?")

        assert len(results) > 0

        # Au moins un résultat devrait contenir le mot "embedding"
        contents = [r.page_content.lower() for r in results]
        assert any("embedding" in content for content in contents)
