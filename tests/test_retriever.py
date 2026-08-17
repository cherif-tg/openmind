"""Tests unitaires pour le module de retrieval (app/retriever.py)."""

from app import retriever


class TestGetRetriever:
    def test_returns_retriever(self, mocker):
        mocker.patch("app.retriever.get_embeddings")
        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vs = mocker.MagicMock()
        mock_vs.as_retriever.return_value = "retriever"
        mock_chroma.return_value = mock_vs

        result = retriever.get_retriever()

        assert result == "retriever"
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": retriever.TOP_K})

    def test_custom_top_k(self, mocker):
        mocker.patch("app.retriever.get_embeddings")
        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vs = mocker.MagicMock()
        mock_chroma.return_value = mock_vs

        retriever.get_retriever(top_k=10)

        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


class TestLoadVectorstore:
    def test_returns_chroma(self, mocker):
        mocker.patch("app.retriever.get_embeddings")
        mock_chroma = mocker.patch("app.retriever.Chroma")
        mock_vs = mocker.MagicMock()
        mock_chroma.return_value = mock_vs

        result = retriever.load_vectorstore()

        assert result is mock_vs


class TestRemoveByFilename:
    def test_deletes_existing_chunks(self, mocker):
        mock_vs = mocker.MagicMock()
        mock_vs._collection.get.return_value = {"ids": ["a", "b"]}
        mocker.patch("app.retriever.load_vectorstore", return_value=mock_vs)

        count = retriever.remove_by_filename("doc.txt")

        assert count == 2
        mock_vs._collection.delete.assert_called_once_with(ids=["a", "b"])

    def test_no_chunks(self, mocker):
        mock_vs = mocker.MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}
        mocker.patch("app.retriever.load_vectorstore", return_value=mock_vs)

        count = retriever.remove_by_filename("doc.txt")

        assert count == 0
        mock_vs._collection.delete.assert_not_called()
