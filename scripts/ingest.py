"""Ingestion d'un dossier en ligne de commande.

Usage:
    python scripts/ingest.py <dossier> [--strategy recursive] [--clear]

Exemple:
    python scripts/ingest.py documents/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunker import chunk_documents  # noqa: E402
from app.embedder import embed_document  # noqa: E402
from app.ingestion import load_folder  # noqa: E402
from app.retriever import load_vectorstore  # noqa: E402
from config import COLLECTION_NAME  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexe un dossier de documents dans le vector store.")
    parser.add_argument("folder", help="Dossier contenant les documents à indexer")
    parser.add_argument("--strategy", default="recursive", choices=["recursive", "character", "token"])
    parser.add_argument("--clear", action="store_true", help="Vide la collection avant ingestion")
    args = parser.parse_args()

    if args.clear:
        vectorstore = load_vectorstore()
        vectorstore._collection.delete(where={})
        print(f"[OpenMind RAG] Collection '{COLLECTION_NAME}' vidée.")

    docs = load_folder(args.folder)

    if not docs:
        print("[OpenMind RAG] Aucun document à indexer.")
        return 1

    chunks = chunk_documents(docs, strategy=args.strategy)
    embed_document(chunks)
    print(f"[OpenMind RAG] {len(chunks)} chunk(s) indexé(s) dans '{COLLECTION_NAME}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
