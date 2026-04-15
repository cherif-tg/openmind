"""
Tests unitaires pour le module d'embedding (app/embedder.py)
"""
import pytest
from langchain_core.documents import Document

from app.embedder import get_embeddings, embed_document
from config import EMBEDDING_MODEL


class TestGetEmbeddings:
    """Tests pour la fonction get_embeddings"""

    def test_get_embeddings_model_type(self):
        """Test que le modèle retourné est du bon type"""
        model = get_embeddings()

        # Vérifier que c'est un objet HuggingFaceEmbeddings
        assert model is not None
        assert hasattr(model, "embed_query")
        assert hasattr(model, "embed_documents")

    def test_get_embeddings_model_name(self):
        """Test que le modèle utilise bien celui de la config"""
        model = get_embeddings()

        assert EMBEDDING_MODEL in model.model_name


class TestEmbedDocument:
    """Tests pour la fonction embed_document"""

    def test_embed_document_creates_vectorstore(self, sample_chunks, temp_vectorstore_dir):
        """Test que embed_document crée un vectorstore"""
        # Modifier temporairement la config pour utiliser le répertoire de test
        import app.embedder as embedder_module
        original_path = embedder_module.VECTORSTORE_PATH
        embedder_module.VECTORSTORE_PATH = temp_vectorstore_dir

        try:
            vs = embed_document(sample_chunks)

            assert vs is not None
            assert hasattr(vs, "_collection")

            # Vérifier que les documents sont bien stockés
            count = vs._collection.count()
            assert count == len(sample_chunks)
        finally:
            embedder_module.VECTORSTORE_PATH = original_path

    def test_embed_document_with_single_chunk(self, temp_vectorstore_dir):
        """Test l'embedding d'un seul chunk"""
        single_chunk = [
            Document(
                page_content="Ceci est un test d'embedding.",
                metadata={"source": "test.txt", "index": 0}
            )
        ]

        import app.embedder as embedder_module
        original_path = embedder_module.VECTORSTORE_PATH
        embedder_module.VECTORSTORE_PATH = temp_vectorstore_dir

        try:
            vs = embed_document(single_chunk)

            assert vs._collection.count() == 1
        finally:
            embedder_module.VECTORSTORE_PATH = original_path

    def test_embed_document_empty_list(self, temp_vectorstore_dir):
        """Test l'embedding d'une liste vide"""
        import app.embedder as embedder_module
        original_path = embedder_module.VECTORSTORE_PATH
        embedder_module.VECTORSTORE_PATH = temp_vectorstore_dir

        try:
            # ChromaDB devrait gérer une liste vide
            vs = embed_document([])

            assert vs is not None
        finally:
            embedder_module.VECTORSTORE_PATH = original_path

    def test_embedding_dimension(self, embedding_model):
        """Test que les embeddings ont la bonne dimension (384 pour all-MiniLM-L6-v2)"""
        # all-MiniLM-L6-v2 produit des embeddings de dimension 384
        test_text = "Test de dimension d'embedding"
        embedding = embedding_model.embed_query(test_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_embedding_consistency(self, embedding_model):
        """Test que le même texte produit le même embedding"""
        test_text = "Texte de test pour la consistance"

        embedding1 = embedding_model.embed_query(test_text)
        embedding2 = embedding_model.embed_query(test_text)

        assert embedding1 == embedding2

    def test_embedding_similarity(self, embedding_model):
        """Test que des textes similaires ont des embeddings similaires"""
        import numpy as np

        text1 = "Le chat est sur le tapis"
        text2 = "Le chat est sur le tapis"  # Identique
        text3 = "La politique économique du gouvernement"  # Différent

        emb1 = np.array(embedding_model.embed_query(text1))
        emb2 = np.array(embedding_model.embed_query(text2))
        emb3 = np.array(embedding_model.embed_query(text3))

        # Similarité cosinus entre textes identiques
        similarity_same = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Similarité cosinus entre textes différents
        similarity_diff = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        # Les textes identiques doivent avoir une similarité de 1.0
        assert abs(similarity_same - 1.0) < 0.01

        # Les textes différents doivent avoir une similarité plus faible
        assert similarity_same > similarity_diff
