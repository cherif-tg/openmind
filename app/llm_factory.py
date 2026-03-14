from config import (
    LLM_MODE,
    GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    HF_MODEL_ID, HF_TOKEN
)

def get_llm(mode: str = None, temperature: float = 0.1):
    """
    Retourne le LLM selon le mode choisi.
    mode: 'groq' | 'ollama' | 'huggingface'
    """
    selected = mode or LLM_MODE

    if selected == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=temperature,
        )

    elif selected == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=temperature,
        )

    elif selected == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=temperature,
        )
        return HuggingFacePipeline(pipeline=pipe)

    else:
        raise ValueError(f"Mode LLM inconnu : {selected}. Choisis 'groq', 'ollama' ou 'huggingface'.")