# OpenMind RAG

> *Open your documents. Unlock your knowledge.*

OpenMind RAG est un système de Retrieval-Augmented Generation (RAG) conçu pour les entreprises et les étudiants. Il permet d'interroger des documents en langage naturel et d'obtenir des réponses précises avec les sources associées.

---

## Fonctionnalités

- Ingestion multi-format : PDF, CSV, DOCX, TXT, HTML, Web
- Découpage intelligent des documents (chunking)
- Vectorisation avec des modèles HuggingFace
- Recherche sémantique via ChromaDB
- Mode hybride LLM : Groq (production) / Ollama / HuggingFace (local)
- Interface conversationnelle avec Streamlit
- Affichage des sources pour chaque réponse
- Évaluation de la qualité via RAGAS

---

## Stack technique

| Composant | Technologie |
|---|---|
| Framework RAG | LangChain |
| Interface | Streamlit |
| Embeddings | sentence-transformers (HuggingFace) |
| Vector Store | ChromaDB |
| LLM production | Groq API (llama-3.1-70b) |
| LLM local | Ollama / HuggingFace |
| Évaluation | RAGAS |

---

## Structure du projet

```
openmind-rag/
├── app/
│   ├── ingestion.py       # Chargement des documents
│   ├── chunker.py         # Découpage en chunks
│   ├── embedder.py        # Vectorisation
│   ├── retriever.py       # Recherche sémantique
│   ├── rag_chain.py       # Pipeline RAG complet
│   └── llm_factory.py     # Gestion hybride des LLMs
├── data/                  # Documents uploadés
├── vectorstore/           # Index ChromaDB persistant
├── tests/                 # Tests unitaires
├── streamlit_app.py       # Point d'entrée de l'interface
├── config.py              # Configuration centralisée
├── .env                   # Variables d'environnement
├── requirements.txt       # Dépendances Python
└── README.md
```

---

## Installation

### Prérequis

- Python 3.10+
- pip
- (Optionnel) Ollama pour le mode local

### Étapes

```bash
# 1. Cloner le projet
git clone https://github.com/ton-username/openmind-rag.git
cd openmind-rag

# 2. Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec tes clés API
```

---

## Configuration

Copie `.env.example` en `.env` et renseigne tes clés :

```env
# Mode LLM : groq | ollama | huggingface
LLM_MODE=groq

# Groq (production)
GROQ_API_KEY=gsk_...

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace (local)
HUGGINGFACE_TOKEN=hf_...
```

---

## Lancement

```bash
streamlit run streamlit_app.py
```

L'application sera accessible sur `http://localhost:8501`.

---

## Utilisation

1. Uploader un ou plusieurs documents via l'interface
2. Patienter pendant l'indexation
3. Poser une question en langage naturel
4. Consulter la réponse et les sources associées

---

## Mode local avec Ollama

```bash
# Installer Ollama depuis https://ollama.com
ollama pull llama3.2

# Définir dans .env
LLM_MODE=ollama
```

---

## Évaluation

OpenMind RAG intègre RAGAS pour mesurer :

- **Faithfulness** : la réponse est-elle fidèle aux documents ?
- **Answer Relevancy** : la réponse répond-elle à la question ?
- **Context Precision** : les chunks récupérés sont-ils pertinents ?

---

## Roadmap

- [ ] Support multilingue
- [ ] Authentification utilisateurs
- [ ] API REST (FastAPI)
- [ ] Support des images (multimodal)
- [ ] Déploiement Docker

---

## Contribuer

Les contributions sont les bienvenues. Merci d'ouvrir une issue avant de soumettre une pull request.

---

## Licence

MIT License — libre d'utilisation pour usage personnel et commercial.

---

*Construit avec LangChain, Streamlit et HuggingFace.*
