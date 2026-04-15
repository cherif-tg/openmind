"""
Tests unitaires pour le module d'ingestion (app/ingestion.py)
"""
import pytest
from pathlib import Path

from app.ingestion import load_document, load_folder, load_from_url


class TestLoadDocument:
    """Tests pour la fonction load_document"""

    def test_load_txt_document(self, test_file_txt):
        """Test le chargement d'un fichier TXT"""
        docs = load_document(test_file_txt)

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(hasattr(doc, "page_content") for doc in docs)
        assert all(hasattr(doc, "metadata") for doc in docs)

    def test_load_csv_document(self, test_file_csv):
        """Test le chargement d'un fichier CSV"""
        docs = load_document(test_file_csv)

        assert isinstance(docs, list)
        assert len(docs) > 0
        # CSVLoader retourne une ligne par document
        assert len(docs) >= 1

    def test_load_document_metadata(self, test_file_txt):
        """Test que les métadonnées sont correctement ajoutées"""
        docs = load_document(test_file_txt)

        assert len(docs) > 0
        doc = docs[0]

        assert "source" in doc.metadata
        assert "file_type" in doc.metadata
        assert "filename" in doc.metadata
        assert doc.metadata["file_type"] == ".txt"
        assert doc.metadata["filename"] == "test.txt"

    def test_load_unsupported_format(self, tmp_path):
        """Test qu'une erreur est levée pour un format non supporté"""
        # Créer un fichier avec une extension non supportée
        file_path = tmp_path / "test.xyz"
        file_path.write_text("contenu", encoding="utf-8")

        with pytest.raises(ValueError, match="Format non supporté"):
            load_document(str(file_path))

    def test_load_nonexistent_file(self):
        """Test qu'une erreur est levée pour un fichier inexistant"""
        with pytest.raises(FileNotFoundError):
            load_document("/chemin/inexistant/fichier.txt")


class TestLoadFolder:
    """Tests pour la fonction load_folder"""

    def test_load_folder_with_mixed_files(self, tmp_path):
        """Test le chargement d'un dossier avec plusieurs fichiers"""
        # Créer plusieurs fichiers de test
        txt_file = tmp_path / "doc1.txt"
        txt_file.write_text("Document 1", encoding="utf-8")

        txt_file2 = tmp_path / "doc2.txt"
        txt_file2.write_text("Document 2", encoding="utf-8")

        docs = load_folder(str(tmp_path))

        assert isinstance(docs, list)
        assert len(docs) >= 2  # Au moins 2 documents chargés

    def test_load_folder_recursive(self, tmp_path):
        """Test que le chargement est récursif"""
        # Créer un sous-dossier
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        txt_file = subdir / "nested.txt"
        txt_file.write_text("Document imbriqué", encoding="utf-8")

        docs = load_folder(str(tmp_path))

        assert len(docs) >= 1
        assert any("nested.txt" in doc.metadata.get("filename", "") for doc in docs)

    def test_load_empty_folder(self, tmp_path):
        """Test le chargement d'un dossier vide"""
        docs = load_folder(str(tmp_path))

        assert isinstance(docs, list)
        assert len(docs) == 0


class TestLoadFromUrl:
    """Tests pour la fonction load_from_url"""

    @pytest.mark.skip(reason="Nécessite une connexion internet")
    def test_load_from_url_real(self):
        """Test réel de chargement d'URL (skip par défaut)"""
        docs = load_from_url("https://example.com")
        assert len(docs) > 0

    def test_load_from_url_structure(self, mocker):
        """Test que load_from_url retourne la bonne structure"""
        # Mock du loader web
        mock_loader = mocker.patch("app.ingestion.WebBaseLoader")
        mock_loader.return_value.load.return_value = [
            Document(page_content="Contenu web", metadata={"source": "https://example.com"})
        ]

        from langchain_core.documents import Document
        docs = load_from_url("https://example.com")

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert docs[0].metadata["file_type"] == "web"
