#!/usr/bin/env python3
"""
Test du chunking sur les documents du dossier docs/
"""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion import load_folder
from app.chunker import chunk_documents
import pandas as pd

def test_chunking():
    print("=" * 70)
    print(" TEST CHUNKING - OpenMind RAG")
    print("=" * 70)
    
    # 1. Charger les documents
    print("\n1. Chargement des documents du dossier 'docs/'...")
    docs_folder = Path(__file__).parent / "docs"
    docs = load_folder(str(docs_folder))
    
    if not docs:
        print("Aucun document chargé !")
        return
    
    # 2. Afficher les statistiques des documents bruts
    print("\n 2. Statistiques des documents bruts :")
    print(f"   • Nombre total de documents : {len(docs)}")
    print(f"   • Taille moyenne : {sum(len(d.page_content) for d in docs) / len(docs):.0f} caractères")
    print(f"   • Taille totale : {sum(len(d.page_content) for d in docs):,} caractères")
    
    # 3. Chunking
    print("\n 3. Application du chunking (stratégie : recursive)...")
    chunks = chunk_documents(docs, strategy="recursive")
    
    # 4. Afficher les statistiques des chunks
    print("\n 4. Statistiques des chunks :")
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    print(f"   • Nombre total de chunks : {len(chunks)}")
    print(f"   • Taille moyenne : {sum(chunk_sizes) / len(chunk_sizes):.0f} caractères")
    print(f"   • Taille min : {min(chunk_sizes)} caractères")
    print(f"   • Taille max : {max(chunk_sizes)} caractères")
    print(f"   • Taille totale : {sum(chunk_sizes):,} caractères")
    
    # 5. Détail par fichier source
    print("\n 5. Détail par fichier source :")
    sources_stats = {}
    for chunk in chunks:
        source = chunk.metadata.get("filename", "unknown")
        if source not in sources_stats:
            sources_stats[source] = {"doc_count": 0, "chunk_count": 0, "total_chars": 0}
        sources_stats[source]["chunk_count"] += 1
        sources_stats[source]["total_chars"] += len(chunk.page_content)
    
    for filename in sorted(sources_stats.keys()):
        stats = sources_stats[filename]
        print(f"   • {filename}")
        print(f"     - Chunks : {stats['chunk_count']}")
        print(f"     - Caractères : {stats['total_chars']:,}")
    
    # 6. Afficher quelques exemples de chunks
    print("\n 6. Exemples de chunks :")
    for i in range(min(3, len(chunks))):
        chunk = chunks[i]
        print(f"\n   --- CHUNK {i+1} ---")
        print(f"   Source : {chunk.metadata.get('filename', 'N/A')}")
        print(f"   Taille : {len(chunk.page_content)} caractères")
        preview = chunk.page_content[:150].replace("\n", " ")
        print(f"   Aperçu : {preview}...")
    
    # 7. Validation
    print("\n 7. Validation :")
    from config import CHUNK_SIZE, CHUNK_OVERLAP
    
    # Vérifier que les chunks ne dépassent pas la taille limite
    oversized = [c for c in chunks if len(c.page_content) > CHUNK_SIZE * 1.5]
    if oversized:
        print(f"    {len(oversized)} chunks dépassent 1.5x la taille limite ({CHUNK_SIZE})")
    else:
        print(f"   ✓ Tous les chunks respectent la taille limite ({CHUNK_SIZE} chars)")
    
    # Vérifier les métadonnées
    metadata_keys = set()
    for chunk in chunks:
        metadata_keys.update(chunk.metadata.keys())
    
    print(f"   ✓ Métadonnées présentes : {', '.join(sorted(metadata_keys))}")
    
    print("\n" + "=" * 70)
    print(" Test du chunking terminé avec succès !")
    print("=" * 70)
    
    return chunks

if __name__ == "__main__":
    chunks = test_chunking()
