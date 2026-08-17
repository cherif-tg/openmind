"""Tests unitaires pour le module de chunking (app/chunker.py)."""

import pytest
from langchain_core.documents import Document

from app.chunker import chunk_documents
from config import CHUNK_OVERLAP, CHUNK_SIZE


class TestChunkDocuments:
    """Tests pour la fonction chunk_documents."""

    def test_chunk_recursive_strategy(self, sample_documents):
        chunks = chunk_documents(sample_documents, strategy="recursive")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(hasattr(chunk, "page_content") for chunk in chunks)

    def test_chunk_character_strategy(self, sample_documents):
        chunks = chunk_documents(sample_documents, strategy="character")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_token_strategy(self, sample_documents):
        chunks = chunk_documents(sample_documents, strategy="token")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_invalid_strategy(self, sample_documents):
        with pytest.raises(ValueError, match="Stratégie de chunking non supportée"):
            chunk_documents(sample_documents, strategy="invalid_strategy")

    def test_chunk_metadata_enrichment(self, sample_documents):
        chunks = chunk_documents(sample_documents, strategy="recursive")
        assert len(chunks) > 0
        chunk = chunks[0]
        assert "index" in chunk.metadata
        assert "strategy" in chunk.metadata
        assert "chunk_size" in chunk.metadata
        assert chunk.metadata["strategy"] == "recursive"

    def test_chunk_custom_size(self, sample_documents):
        chunks = chunk_documents(
            sample_documents,
            strategy="recursive",
            chunk_size=100,
            chunk_overlap=10,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.page_content) <= 100 + 20  # marge de tolérance

    def test_chunk_empty_document(self):
        empty_doc = Document(page_content="", metadata={"source": "empty.txt"})
        with pytest.raises(ValueError, match="Le document fourni est vide"):
            chunk_documents([empty_doc], strategy="recursive")

    def test_chunk_sequential_indexing(self, sample_documents):
        chunks = chunk_documents(sample_documents, strategy="recursive")
        indices = [chunk.metadata["index"] for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_preserves_source_metadata(self, sample_documents):
        chunks = chunk_documents(sample_documents, strategy="recursive")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == sample_documents[i].metadata["source"]


class TestChunkSizes:
    """Tests spécifiques aux tailles de chunks."""

    def test_default_config_values(self):
        doc = Document(page_content="A" * 1000, metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], strategy="recursive")
        assert len(chunks) > 0
        assert chunks[0].metadata["chunk_size"] <= CHUNK_SIZE + 10

    def test_large_document(self):
        large_doc = Document(
            page_content=" ".join(["Mot"] * 1000),
            metadata={"source": "large.txt"},
        )
        chunks = chunk_documents([large_doc], strategy="recursive")
        assert len(chunks) > 1
        assert all(len(chunk.page_content) > 0 for chunk in chunks)
