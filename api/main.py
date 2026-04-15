"""
OpenMind RAG API - Point d'entrée FastAPI
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import documents, query

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="OpenMind RAG API",
    description="API REST pour interroger vos documents en langage naturel",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
# Permet à Streamlit (généralement sur localhost:8501) d'appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://localhost:8502",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8502",
        "*",  # En développement, permettre tout
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Toutes les méthodes HTTP
    allow_headers=["*"],  # Tous les headers
)

# Enregistrement des routers
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/")
async def root():
    """
    Endpoint racine - Informations sur l'API
    """
    return {
        "name": "OpenMind RAG API",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "documents": "/api/documents",
            "query": "/api/query"
        }
    }


@app.get("/health")
async def health():
    """
    Health check - Vérifie que l'API est opérationnelle
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
