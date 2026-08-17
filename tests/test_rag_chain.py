"""Tests unitaires pour le pipeline RAG (app/rag_chain.py)."""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app import rag_chain


def _docs():
    return [
        Document(
            page_content="Premier passage",
            metadata={"filename": "a.txt", "index": 0, "page": 1},
        ),
        Document(
            page_content="Deuxième passage",
            metadata={"filename": "b.txt", "index": 1, "page": 2},
        ),
    ]


class TestBuildContext:
    def test_numbers_chunks(self):
        context = rag_chain.build_context(_docs())
        assert "[1] Premier passage" in context
        assert "[2] Deuxième passage" in context

    def test_empty(self):
        assert rag_chain.build_context([]) == ""


class TestBuildRagChain:
    def test_no_documents(self, mocker):
        mocker.patch("app.rag_chain.get_retriever").return_value.invoke.return_value = []

        answer, docs = rag_chain.build_rag_chain("question", rerank=False)

        assert "pas trouvé" in answer
        assert docs == []

    def test_generates_answer_with_citations(self, mocker):
        docs = _docs()
        mocker.patch("app.rag_chain.get_retriever").return_value.invoke.return_value = docs
        mocker.patch(
            "app.rag_chain.get_llm",
            return_value=RunnableLambda(lambda inputs: "Réponse [1] et [2]"),
        )

        answer, used_docs = rag_chain.build_rag_chain("question", rerank=False)

        assert answer == "Réponse [1] et [2]"
        assert used_docs == docs

    def test_rerank_path(self, mocker):
        docs = _docs()
        mocker.patch("app.rag_chain.get_retriever").return_value.invoke.return_value = docs
        mocker.patch("app.rag_chain.get_llm", return_value=RunnableLambda(lambda inputs: "ok"))
        rerank_mock = mocker.patch("app.reranker.rerank_documents", return_value=[docs[1], docs[0]])

        rag_chain.build_rag_chain("question", top_k=1, rerank=True)

        rerank_mock.assert_called_once()

    def test_rerank_falls_back_on_error(self, mocker):
        docs = _docs()
        mocker.patch("app.rag_chain.get_retriever").return_value.invoke.return_value = docs
        mocker.patch("app.rag_chain.get_llm", return_value=RunnableLambda(lambda inputs: "ok"))
        mocker.patch("app.reranker.rerank_documents", side_effect=RuntimeError("model absent"))

        answer, used_docs = rag_chain.build_rag_chain("question", top_k=1, rerank=True)

        assert len(used_docs) == 1

    def test_accepts_custom_retriever(self, mocker):
        docs = _docs()
        mock_retriever = mocker.MagicMock()
        mock_retriever.invoke.return_value = docs
        mocker.patch("app.rag_chain.get_llm", return_value=RunnableLambda(lambda inputs: "ok"))
        get_retriever_mock = mocker.patch("app.rag_chain.get_retriever")

        answer, used_docs = rag_chain.build_rag_chain(
            "question", rerank=False, retriever=mock_retriever
        )

        assert used_docs == docs
        get_retriever_mock.assert_not_called()
