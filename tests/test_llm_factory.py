"""
Tests unitaires pour le llm_factory (app/llm_factory.py)
"""
import pytest
from unittest.mock import patch, MagicMock

from app.llm_factory import get_llm


class TestGetLLM:
    """Tests pour la fonction get_llm"""

    def test_get_llm_groq_mode(self, mocker):
        """Test le mode Groq"""
        mock_chat_groq = mocker.patch("app.llm_factory.ChatGroq")
        mock_chat_groq.return_value = MagicMock()

        llm = get_llm(mode="groq")

        assert llm is not None
        mock_chat_groq.assert_called_once()

    def test_get_llm_ollama_mode(self, mocker):
        """Test le mode Ollama"""
        mock_ollama = mocker.patch("app.llm_factory.Ollama")
        mock_ollama.return_value = MagicMock()

        llm = get_llm(mode="ollama")

        assert llm is not None
        mock_ollama.assert_called_once()

    def test_get_llm_huggingface_mode(self, mocker):
        """Test le mode HuggingFace"""
        # Mock des composants HuggingFace
        mock_tokenizer = mocker.patch("app.llm_factory.AutoTokenizer")
        mock_model = mocker.patch("app.llm_factory.AutoModelForCausalLM")
        mock_pipeline = mocker.patch("app.llm_factory.pipeline")
        mock_hf_pipeline = mocker.patch("app.llm_factory.HuggingFacePipeline")
        mock_hf_pipeline.return_value = MagicMock()

        llm = get_llm(mode="huggingface")

        assert llm is not None
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()

    def test_get_llm_invalid_mode(self):
        """Test qu'une erreur est levée pour un mode invalide"""
        with pytest.raises(ValueError, match="Mode LLM inconnu"):
            get_llm(mode="invalid_mode")

    def test_get_llm_default_mode(self, mocker):
        """Test le mode par défaut (utilise LLM_MODE de config)"""
        from config import LLM_MODE

        # Selon le mode par défaut, mocker le bon import
        if LLM_MODE == "groq":
            mock_llm = mocker.patch("app.llm_factory.ChatGroq")
        elif LLM_MODE == "ollama":
            mock_llm = mocker.patch("app.llm_factory.Ollama")
        else:
            mock_llm = mocker.patch("app.llm_factory.HuggingFacePipeline")

        mock_llm.return_value = MagicMock()

        llm = get_llm()  # Sans argument, utilise le mode par défaut

        assert llm is not None
        mock_llm.assert_called_once()

    def test_get_llm_temperature_parameter(self, mocker):
        """Test que le paramètre temperature est passé au LLM"""
        mock_chat_groq = mocker.patch("app.llm_factory.ChatGroq")
        mock_chat_groq.return_value = MagicMock()

        get_llm(mode="groq", temperature=0.7)

        # Vérifier que temperature=0.7 a été passé
        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_get_llm_custom_temperature(self, mocker):
        """Test avec une température personnalisée"""
        mock_ollama = mocker.patch("app.llm_factory.Ollama")
        mock_ollama.return_value = MagicMock()

        get_llm(mode="ollama", temperature=0.5)

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs["temperature"] == 0.5


class TestLLMFactoryConfig:
    """Tests liés à la configuration du llm_factory"""

    def test_groq_uses_config(self, mocker):
        """Test que le mode Groq utilise la config"""
        from config import GROQ_API_KEY, GROQ_MODEL

        mock_chat_groq = mocker.patch("app.llm_factory.ChatGroq")
        mock_chat_groq.return_value = MagicMock()

        get_llm(mode="groq")

        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["api_key"] == GROQ_API_KEY
        assert call_kwargs["model_name"] == GROQ_MODEL

    def test_ollama_uses_config(self, mocker):
        """Test que le mode Ollama utilise la config"""
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        mock_ollama = mocker.patch("app.llm_factory.Ollama")
        mock_ollama.return_value = MagicMock()

        get_llm(mode="ollama")

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs["base_url"] == OLLAMA_BASE_URL
        assert call_kwargs["model"] == OLLAMA_MODEL
