"""Tests unitaires pour llm_factory (app/llm_factory.py)."""

import pytest

from app.llm_factory import get_llm


class TestGetLLM:
    def test_groq_mode(self, mocker):
        mock_cls = mocker.patch("langchain_groq.ChatGroq")
        mock_cls.return_value = "llm"
        assert get_llm(mode="groq") == "llm"
        mock_cls.assert_called_once()

    def test_ollama_mode(self, mocker):
        mock_cls = mocker.patch("langchain_community.llms.Ollama")
        mock_cls.return_value = "llm"
        assert get_llm(mode="ollama") == "llm"
        mock_cls.assert_called_once()

    def test_huggingface_mode(self, mocker):
        mocker.patch("transformers.AutoTokenizer")
        mocker.patch("transformers.AutoModelForCausalLM")
        mocker.patch("transformers.pipeline")
        mock_hf = mocker.patch("langchain_huggingface.HuggingFacePipeline")
        mock_hf.return_value = "llm"
        assert get_llm(mode="huggingface") == "llm"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Mode LLM inconnu"):
            get_llm(mode="invalid_mode")


class TestLLMFactoryConfig:
    def test_groq_uses_config(self, mocker):
        from config import GROQ_API_KEY, GROQ_MODEL

        mock_cls = mocker.patch("langchain_groq.ChatGroq")
        get_llm(mode="groq")
        kwargs = mock_cls.call_args[1]
        assert kwargs["api_key"] == GROQ_API_KEY
        assert kwargs["model_name"] == GROQ_MODEL

    def test_ollama_uses_config(self, mocker):
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        mock_cls = mocker.patch("langchain_community.llms.Ollama")
        get_llm(mode="ollama")
        kwargs = mock_cls.call_args[1]
        assert kwargs["base_url"] == OLLAMA_BASE_URL
        assert kwargs["model"] == OLLAMA_MODEL

    def test_temperature_passed(self, mocker):
        mock_cls = mocker.patch("langchain_groq.ChatGroq")
        get_llm(mode="groq", temperature=0.7)
        assert mock_cls.call_args[1]["temperature"] == 0.7
