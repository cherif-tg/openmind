#!/usr/bin/env python3
"""
Test du RAG chain complet avec Ollama (gemma3b)
"""
import sys
from pathlib import Path
import requests

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag_chain import build_rag_chain
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_MODE

def test_rag_chain(question: str):
    print("=" * 70)
    print("TEST RAG CHAIN - OpenMind RAG")
    print("=" * 70)
    
    # 1. Vérifier la configuration
    print(f"\n  1. Vérification de la configuration :")
    print(f"   • Mode LLM : {LLM_MODE}")
    print(f"   • Modèle : {OLLAMA_MODEL}")
    print(f"   • URL Ollama : {OLLAMA_BASE_URL}")
    
    # 2. Vérifier la disponibilité d'Ollama
    print(f"\n 2. Vérification de la connexion à Ollama...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"   Ollama est connecté !")
        else:
            print(f"   Ollama répond mais avec le statut {response.status_code}")
            return
    except Exception as e:
        print(f"    Erreur de connexion à Ollama : {e}")
        print(f"      Assurez-vous que Ollama est lancé avec: ollama serve")
        return
    
    # 3. Afficher la question
    print(f"\n 3. Question posée :")
    print(f"   \"{question}\"")
    
    # 4. Exécuter la chaîne RAG
    print(f"\n  4. Exécution de la chaîne RAG...")
    print(f"   (Cela peut prendre quelques secondes...)")
    try:
        reponse, sources = build_rag_chain(question)
    except Exception as e:
        print(f"   Erreur lors de l'exécution : {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Afficher la réponse
    print(f"\n 5. Réponse du LLM :")
    print(f"   {reponse}")
    
    # 6. Afficher les sources
    print(f"\n 6. Sources utilisées ({len(sources)} chunk(s)) :")
    for i, source in enumerate(sources, 1):
        filename = source.metadata.get('filename', 'unknown')
        chunk_index = source.metadata.get('index', 'N/A')
        preview = source.page_content[:150].replace("\n", " ")
        print(f"\n   [{i}] {filename} (Chunk #{chunk_index})")
        print(f"       {preview}...")
    
    print("\n" + "=" * 70)
    print(" Test du RAG chain terminé !")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Qu'est ce que Microsoft?"
    
    test_rag_chain(question)
