---
title: OpenMind RAG
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# OpenMind RAG

> *Open your documents. Unlock your knowledge.*

Système de **Retrieval-Augmented Generation (RAG)** complet : ingestion de
documents, découpage, vectorisation, recherche sémantique, **re-ranking** et
génération de réponses **avec citations des sources**. Livré avec une API
FastAPI, une interface Streamlit et une **démo live Gradio**.

---

## Fonctionnalités

- **Ingestion multi-format** : PDF, CSV, DOCX, TXT, **Markdown**, HTML, Web
- **Chunking** : 3 stratégies (recursive, character, token) avec chevauchement
- **Embeddings** : modèles HuggingFace (`sentence-transformers`), local et gratuit
- **Vector store** : ChromaDB persistant
- **Retrieval** : recherche sémantique top-k
- **Re-ranking** : cross-encoder local (`ms-marco-MiniLM-L-6-v2`) pour affiner le classement
- **Génération avec citations** : les sources sont numérotées `[1]`, `[2]`, … et référencées dans la réponse
- **LLM hybride** : Groq (production) / Ollama (local) / HuggingFace (local)
- **API REST** : FastAPI (upload, liste, suppression, requête)
- **Interface** : Streamlit conversationnelle
- **Démo live** : Gradio (prête pour Hugging Face Spaces)
- **Tests** : suite pytest

---

## Démo live (Hugging Face Spaces)

Une démo interroge un petit corpus de démonstration (`demo_corpus/`) via le
fichier `app.py` (Gradio). Pour la déployer :

1. Créez un Space sur <https://huggingface.co> avec le SDK **Gradio**.
2. Liez-le à ce dépôt (ou poussez la branche `main` vers l'espace).
3. Ajoutez le secret **`GROQ_API_KEY`** dans *Settings → Variables and secrets*.

Au premier appel, les modèles d'embeddings et de re-ranking sont téléchargés
(~200 Mo), puis mis en cache. Le premier chargement peut donc être lent.

---

## Stack technique

| Composant       | Technologie                                    |
|-----------------|------------------------------------------------|
| Framework RAG   | LangChain                                      |
| Interface       | Streamlit                                      |
| Démo            | Gradio                                         |
| API             | FastAPI + Uvicorn                              |
| Embeddings      | sentence-transformers (HuggingFace)            |
| Re-ranking      | sentence-transformers (cross-encoder)          |
| Vector store    | ChromaDB                                       |
| LLM production  | Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`)           |
| LLM local       | Ollama / HuggingFace                           |

---

## Structure du projet

```
openmind/
├── app/                   # Cœur du système RAG
│   ├── ingestion.py       # Chargement des documents (PDF, MD, TXT, …)
│   ├── chunker.py         # Découpage en chunks
│   ├── embedder.py        # Vectorisation
│   ├── retriever.py       # Recherche sémantique
│   ├── reranker.py        # Re-ranking (cross-encoder)
│   ├── rag_chain.py       # Pipeline complet + prompt avec citations
│   └── llm_factory.py     # Gestion hybride des LLMs
├── api/                   # Backend FastAPI
│   ├── main.py
│   ├── routers/           # documents, query
│   └── schemas/           # Modèles Pydantic
├── scripts/
│   └── ingest.py          # Ingestion d'un dossier en ligne de commande
├── tests/                 # Suite pytest
├── demo_corpus/           # Corpus de démonstration (Markdown)
├── app.py                 # Démo Gradio (Hugging Face Spaces)
├── streamlit_app.py       # Interface Streamlit (application complète)
├── config.py              # Configuration centralisée
├── requirements.txt       # Dépendances de base (démo + cœur RAG)
├── requirements-full.txt  # Dépendances complètes (API, Streamlit, tests)
└── .env.example
```

Les dossiers `vectorstore/`, `documents/` et `data/` contiennent des données
générées ou fournies par l'utilisateur : ils ne sont pas versionnés.

---

## Installation

Prérequis : Python 3.11+, pip. (Ollama est optionnel, pour le mode local.)

```bash
git clone https://github.com/cherif-tg/openmind.git
cd openmind

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Démo seule (Gradio)
pip install -r requirements.txt

# Application complète (API + Streamlit + loaders + tests)
pip install -r requirements-full.txt

cp .env.example .env   # puis renseigner les clés
```

---

## Configuration

Copiez `.env.example` vers `.env` et renseignez :

```env
# Mode LLM : groq | ollama | huggingface
LLM_MODE=groq
GROQ_API_KEY=***

# Re-ranking (true/false)
RERANK_ENABLED=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Lancement

### 1. Démo Gradio (monoprocess)

```bash
python app.py
```

### 2. Ingestion des documents (CLI)

```bash
python scripts/ingest.py documents/
```

### 3. API FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

Docs interactives : <http://localhost:8000/docs>

### 4. Interface Streamlit

```bash
streamlit run streamlit_app.py --server.port 8501
```

---

## Utilisation

1. Uploader des documents via l'interface (ou `scripts/ingest.py`)
2. Patienter pendant l'indexation
3. Poser une question en langage naturel
4. Consulter la réponse **et** les sources numérotées

Exemple de requête API :

```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "Qu'est-ce que le RAG ?", "top_k": 5}'
```

Réponse (extrait) :

```json
{
  "answer": "Le RAG combine la récupération d'informations et la génération [1].",
  "sources": [
    {
      "content": "Le RAG (Retrieval-Augmented Generation) est une technique…",
      "metadata": { "source": "doc.pdf", "page": 1, "chunk": 0 }
    }
  ],
  "llm_mode": "groq"
}
```

---

## Mode local avec Ollama

```bash
ollama pull gemma4:e4b
# puis définir LLM_MODE=ollama dans .env
```

---

## Tests

```bash
pytest
```

Les tests unitaires sont isolés (mocks) ; certains tests d'intégration
(embedding réel) nécessitent le téléchargement du modèle au premier lancement.

---

## Roadmap

- [ ] Évaluation automatique de la qualité (RAGAS)
- [ ] Import d'URL exposé dans l'API et l'interface
- [ ] Upload de documents dans la démo Gradio
- [ ] Authentification utilisateurs
- [ ] Support multimodal (images)

---

## Licence

MIT — libre d'utilisation pour usage personnel et commercial.
