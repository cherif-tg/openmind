"""
Schémas Pydantic pour la gestion des documents
"""
from pydantic import BaseModel, Field
from typing import Optional


class DocumentUpload(BaseModel):
    """Réponse après upload d'un document"""
    filename: str
    chunks_count: int
    status: str = "indexed"


class DocumentInfo(BaseModel):
    """Informations sur un document indexé"""
    filename: str
    chunks_count: int
    indexed_at: Optional[str] = None


class DocumentList(BaseModel):
    """Liste des documents indexés"""
    documents: list[DocumentInfo]
    total: int


class DocumentDelete(BaseModel):
    """Réponse après suppression d'un document"""
    filename: str
    status: str
    message: str
