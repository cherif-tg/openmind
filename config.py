import os
from dotenv import load_dotenv

load_dotenv()

# ── Mode LLM ──────────────────────────────────────────────
# "groq" | "ollama" | "huggingface"
LLM_MODE = os.getenv("LLM_MODE", "groq")

# ── Groq (production) ─────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.1-70b-versatile"   # rapide + gratuit
GROQ_MODEL_LIGHT  = "llama-3.1-8b-instant"      # encore plus rapide

# ── Ollama (local) ────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = "gemma3b"                  # ou mistral, phi3...

# ── HuggingFace local (fallback) ──────────────────────────
HF_MODEL_ID       = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN          = os.getenv("HUGGINGFACE_TOKEN", "")

# ── Embedding (toujours local, gratuit) ───────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE  = "cpu"    # "cuda" si GPU disponible

# ── Vector store ──────────────────────────────────────────
VECTORSTORE_PATH  = "./vectorstore"
COLLECTION_NAME   = "openmind_docs"

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50

# ── Retrieval ─────────────────────────────────────────────
TOP_K             = 5

# ── API Backend ───────────────────────────────────────────
API_HOST          = os.getenv("API_HOST", "localhost")
API_PORT          = int(os.getenv("API_PORT", "8000"))
STREAMLIT_API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")