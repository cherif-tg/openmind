"""Tests unitaires pour le re-ranking (app/reranker.py)."""

from langchain_core.documents import Document

from app import reranker


class TestRerankDocuments:
    def test_orders_by_score(self, mocker):
        docs = [
            Document(page_content="A", metadata={}),
            Document(page_content="B", metadata={}),
            Document(page_content="C", metadata={}),
        ]
        mock_model = mocker.MagicMock()
        mock_model.predict.return_value = [1.0, 3.0, 2.0]
        mocker.patch("app.reranker.get_reranker", return_value=mock_model)

        ranked = reranker.rerank_documents("question", docs, top_k=2)

        assert [d.page_content for d in ranked] == ["B", "C"]
        assert ranked[0].metadata["rerank_score"] == 3.0

    def test_empty(self):
        assert reranker.rerank_documents("q", []) == []

    def test_preserves_metadata(self, mocker):
        docs = [Document(page_content="A", metadata={"source": "x"})]
        mock_model = mocker.MagicMock()
        mock_model.predict.return_value = [1.0]
        mocker.patch("app.reranker.get_reranker", return_value=mock_model)

        ranked = reranker.rerank_documents("q", docs)

        assert ranked[0].metadata["source"] == "x"
        assert ranked[0].metadata["rerank_score"] == 1.0
