#!/usr/bin/env python3
"""
Test des embeddings et stockage dans le vectorstore (ChromaDB)
"""
import sys
import shutil
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion import load_folder
from app.chunker import chunk_documents
from app.embedder import embed_document, get_embeddings
from app.retriever import load_vectorstore
from config import VECTORSTORE_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

def test_embeddings():
    print("=" * 70)
    print(" TEST EMBEDDINGS - OpenMind RAG")
    print("=" * 70)
    
    # 0. Nettoyer le vectorstore précédent (optionnel)
    print("\n  0. Nettoyage du vectorstore précédent...")
    vectorstore_dir = Path(VECTORSTORE_PATH)
    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)
        print(f"    Vectorstore supprimé : {VECTORSTORE_PATH}")
    
    # 1. Charger les documents
    print("\n 1. Chargement des documents...")
    docs_folder = Path(__file__).parent / "docs"
    docs = load_folder(str(docs_folder))
    
    if not docs:
        print(" Aucun document chargé !")
        return
    
    # 2. Chunking
    print("\n 2. Chunking des documents...")
    chunks = chunk_documents(docs, strategy="recursive")
    
    # 3. Afficher les infos sur le modèle d'embeddings
    print("\n 3. Informations du modèle d'embeddings :")
    embeddings_model = get_embeddings()
    print(f"   • Modèle : {EMBEDDING_MODEL}")
    print(f"   • Dimension : {len(embeddings_model.embed_query('test'))} dimensions")
    
    # 4. Embeddings et stockage
    print("\n  4. Génération des embeddings et stockage dans ChromaDB...")
    try:
        vector_store = embed_document(chunks)
        print(f"    Vectorstore créé avec succès")
    except Exception as e:
        print(f"   Erreur lors de la création du vectorstore : {e}")
        return
    
    # 5. Vérifier le vectorstore
    print("\n 5. Vérification du vectorstore :")
    try:
        loaded_store = load_vectorstore()
        collection_info = loaded_store._collection
        
        print(f"   • Collection : {COLLECTION_NAME}")
        print(f"   • Nombre de vecteurs : {collection_info.count()}")
        print(f"   • Chemin : {VECTORSTORE_PATH}")
        
        # Vérifier les métadonnées
        all_data = collection_info.get()
        if all_data and 'metadatas' in all_data and all_data['metadatas']:
            sample_metadata = all_data['metadatas'][0]
            print(f"   • Métadonnées : {', '.join(sample_metadata.keys())}")
    except Exception as e:
        print(f"   Erreur lors de la vérification : {e}")
        return
    
    # 6. Test de recherche sémantique
    print("\n 6. Test de recherche sémantique :")
    test_queries = [
        "Qu'est-ce que Microsoft ?",
        "NVIDIA et l'IA",
        "SpaceX innovations"
    ]
    
    for query in test_queries:
        try:
            results = loaded_store.similarity_search(query, k=TOP_K)
            print(f"\n   Query: '{query}'")
            print(f"   Résultats trouvés : {len(results)}")
            for i, result in enumerate(results[:2], 1):
                source = result.metadata.get('filename', 'unknown')
                preview = result.page_content[:80].replace("\n", " ")
                print(f"     {i}. [{source}] {preview}...")
        except Exception as e:
            print(f"    Erreur pour '{query}' : {e}")
    
    # 7. Afficher les informations de stockage
    print("\n 7. Informations de stockage :")
    if vectorstore_dir.exists():
        total_size = sum(f.stat().st_size for f in vectorstore_dir.rglob('*') if f.is_file())
        num_files = len(list(vectorstore_dir.rglob('*')))
        print(f"   • Dossier : {VECTORSTORE_PATH}")
        print(f"   • Nombre de fichiers : {num_files}")
        print(f"   • Taille totale : {total_size / (1024*1024):.2f} MB")
    
    print("\n" + "=" * 70)
    print(" Test des embeddings terminé avec succès !")
    print("=" * 70)

if __name__ == "__main__":
    test_embeddings()
