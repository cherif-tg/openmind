"""
Schémas Pydantic pour les requêtes RAG
"""
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Requête pour poser une question"""
    question: str = Field(..., min_length=1, description="La question à poser")
    top_k: int = Field(default=5, ge=1, le=20, description="Nombre de chunks à récupérer")
    llm_mode: Optional[str] = Field(default=None, description="Mode LLM (groq|ollama|huggingface)")


class Source(BaseModel):
    """Source d'un chunk récupéré"""
    content: str
    metadata: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Le RAG est une technique...",
                "metadata": {
                    "source": "document.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            }
        }


class QueryResponse(BaseModel):
    """Réponse à une question RAG"""
    answer: str = Field(..., description="La réponse générée par le LLM")
    sources: list[Source] = Field(default_factory=list, description="Les sources utilisées")
    llm_mode: str = Field(..., description="Le mode LLM utilisé")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Le RAG (Retrieval-Augmented Generation) est...",
                "sources": [
                    {
                        "content": "Le RAG combine retrieval et génération...",
                        "metadata": {"source": "doc.pdf", "page": 1}
                    }
                ],
                "llm_mode": "groq"
            }
        }
