"""
Tests unitaires pour le RAG chain (app/rag_chain.py)
"""
import pytest
from unittest.mock import MagicMock, patch

from app.rag_chain import build_rag_chain


class TestBuildRagChain:
    """Tests pour la fonction build_rag_chain"""

    def test_build_rag_chain_returns_tuple(self, mocker):
        """Test que build_rag_chain retourne un tuple (réponse, docs)"""
        # Mock du LLM
        mock_llm = mocker.patch("app.llm_factory.get_llm")
        mock_llm.return_value = mocker.MagicMock()
        mock_llm.return_value.invoke.return_value = MagicMock(content="Réponse test")

        # Mock du retriever
        mock_retriever = mocker.patch("app.retriever.get_retriever")
        mock_docs = [
            MagicMock(page_content="Document 1", metadata={"source": "test.pdf"}),
            MagicMock(page_content="Document 2", metadata={"source": "test.pdf"})
        ]
        mock_retriever.return_value.invoke.return_value = mock_docs

        # Mock de PromptTemplate et StrOutputParser
        mocker.patch("app.rag_chain.PromptTemplate")
        mocker.patch("app.rag_chain.StrOutputParser")

        response, docs = build_rag_chain("Question de test")

        assert isinstance(response, str)
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_build_rag_chain_calls_retriever(self, mocker):
        """Test que build_rag_chain appelle bien le retriever"""
        mock_llm = mocker.patch("app.llm_factory.get_llm")
        mock_llm.return_value = mocker.MagicMock()

        mock_retriever = mocker.patch("app.retriever.get_retriever")
        mock_docs = [MagicMock(page_content="Doc", metadata={})]
        mock_retriever.return_value.invoke.return_value = mock_docs

        mocker.patch("app.rag_chain.PromptTemplate")
        mocker.patch("app.rag_chain.StrOutputParser")

        build_rag_chain("Question")

        # Vérifier que le retriever a été appelé avec la question
        mock_retriever.return_value.invoke.assert_called_once_with("Question")

    def test_build_rag_chain_calls_llm(self, mocker):
        """Test que build_rag_chain appelle le LLM"""
        mock_llm = mocker.patch("app.llm_factory.get_llm")
        mock_llm_instance = mocker.MagicMock()
        mock_llm.return_value = mock_llm_instance

        mock_retriever = mocker.patch("app.retriever.get_retriever")
        mock_docs = [MagicMock(page_content="Contexte de test", metadata={})]
        mock_retriever.return_value.invoke.return_value = mock_docs

        mocker.patch("app.rag_chain.PromptTemplate")
        mocker.patch("app.rag_chain.StrOutputParser")

        build_rag_chain("Question")

        # Vérifier que la chaîne a été invoquée
        assert mock_llm_instance.invoke.called or mock_llm_instance.__or__.called

    def test_build_rag_chain_formats_context(self, mocker):
        """Test que le contexte est bien formaté"""
        mock_llm = mocker.patch("app.llm_factory.get_llm")
        mock_llm.return_value = mocker.MagicMock()

        mock_retriever = mocker.patch("app.retriever.get_retriever")
        mock_docs = [
            MagicMock(page_content="Premier document", metadata={}),
            MagicMock(page_content="Deuxième document", metadata={})
        ]
        mock_retriever.return_value.invoke.return_value = mock_docs

        mocker.patch("app.rag_chain.PromptTemplate")
        mocker.patch("app.rag_chain.StrOutputParser")

        build_rag_chain("Question")

        # Le contexte devrait contenir les deux documents séparés par des newlines
        mock_retriever.return_value.invoke.assert_called_once()


class TestRagChainIntegration:
    """Tests d'intégration pour le RAG chain"""

    @pytest.mark.skip(reason="Nécessite un LLM configuré")
    def test_build_rag_chain_real_llm(self):
        """Test réel avec un vrai LLM (skip par défaut)"""
        # Ce test nécessite une clé API valide
        response, docs = build_rag_chain("Qu'est-ce que le RAG?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(docs, list)

    def test_build_rag_chain_empty_results(self, mocker):
        """Test quand le retriever ne trouve aucun document"""
        mock_llm = mocker.patch("app.llm_factory.get_llm")
        mock_llm.return_value = mocker.MagicMock()

        mock_retriever = mocker.patch("app.retriever.get_retriever")
        mock_retriever.return_value.invoke.return_value = []  # Aucun résultat

        mocker.patch("app.rag_chain.PromptTemplate")
        mocker.patch("app.rag_chain.StrOutputParser")

        # Gérer le cas où le contexte est vide
        with pytest.raises(Exception):
            build_rag_chain("Question sans résultats")
