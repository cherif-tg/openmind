# Plan de développement — OpenMind RAG
> Instructions pour Claude Code

---

## Contexte du projet

**OpenMind RAG** est un système de Retrieval-Augmented Generation (RAG) permettant d'interroger des documents en langage naturel. Il cible les entreprises et les étudiants.

- **Interface** : Streamlit (`streamlit_app.py`)
- **Pipeline RAG** : LangChain
- **Vector Store** : ChromaDB (persistant dans `vectorstore/`)
- **Embeddings** : sentence-transformers (HuggingFace)
- **LLM** : Groq (production) / Ollama / HuggingFace (local), géré par `llm_factory.py`
- **Évaluation** : RAGAS

---

## Structure actuelle du projet

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
├── requirements.txt
└── README.md
```

---

## Priorités de travail

Les tâches sont classées par ordre de priorité. Traite-les **dans l'ordre indiqué**.

---

### 1. Architecture & Structure — Ajouter le backend FastAPI

Le projet n'a pas encore de backend REST. Il faut en créer un avec **FastAPI** sans casser le Streamlit existant.

**Structure cible à créer :**

```
openmind-rag/
├── api/                        # Nouveau — backend FastAPI
│   ├── __init__.py
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── documents.py        # Routes upload / liste / suppression
│   │   └── query.py            # Route pour les questions RAG
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── document.py         # Pydantic models pour les documents
│   │   └── query.py            # Pydantic models pour les requêtes/réponses
│   └── dependencies.py         # Injection de dépendances (RAG chain, etc.)
├── app/                        # Existant — inchangé
├── streamlit_app.py            # Existant — à adapter pour consommer l'API
├── config.py                   # Existant — enrichir si nécessaire
└── ...
```

**Contraintes impératives :**
- Ne pas modifier les fichiers dans `app/` sauf si absolument nécessaire
- Le code de `app/` doit rester réutilisable à la fois par l'API et par Streamlit
- Utiliser **Pydantic v2** pour tous les schémas
- Activer **CORS** dans `api/main.py` pour permettre les appels depuis Streamlit

---

### 2. Écriture du code fonctionnel

#### 2.1 — Backend FastAPI (`api/`)

Implémenter les endpoints suivants :

**Documents (`api/routers/documents.py`) :**

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload un ou plusieurs fichiers (PDF, CSV, DOCX, TXT, HTML) |
| `GET` | `/api/documents/` | Liste tous les documents indexés |
| `DELETE` | `/api/documents/{doc_id}` | Supprime un document et ses chunks du vector store |

**Query (`api/routers/query.py`) :**

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/api/query/` | Pose une question, retourne la réponse + les sources |

**Format de réponse attendu pour `/api/query/` :**
```json
{
  "answer": "string",
  "sources": [
    {
      "content": "string",
      "metadata": {
        "source": "string",
        "page": "int ou null"
      }
    }
  ],
  "llm_mode": "groq | ollama | huggingface"
}
```

#### 2.2 — Adaptation de `streamlit_app.py`

- Modifier Streamlit pour qu'il appelle l'API FastAPI via `httpx` ou `requests` au lieu d'importer directement les modules `app/`
- Garder exactement la même UX pour l'utilisateur final
- Gérer proprement les erreurs HTTP (afficher un message clair si l'API est indisponible)

#### 2.3 — Configuration (`config.py`)

S'assurer que `config.py` expose :
- `API_HOST` et `API_PORT` (défaut : `localhost:8000`)
- `STREAMLIT_API_BASE_URL` (URL complète de l'API consommée par Streamlit)
- Toutes les variables existantes déjà présentes

---

### 3. Tests & Qualité du code

#### 3.1 — Tests unitaires (dans `tests/`)

Créer ou compléter les fichiers suivants :

- `tests/test_ingestion.py` — Tester le chargement de chaque format (PDF, CSV, DOCX, TXT)
- `tests/test_chunker.py` — Vérifier que le découpage produit des chunks de taille cohérente
- `tests/test_embedder.py` — Vérifier que les embeddings ont la bonne dimension
- `tests/test_retriever.py` — Tester la recherche sémantique avec une requête simple
- `tests/test_api.py` — Tester les endpoints FastAPI avec `httpx.AsyncClient` + `pytest-asyncio`

**Règles pour les tests :**
- Utiliser **pytest**
- Mocker les appels LLM externes (Groq, HuggingFace) avec `unittest.mock` ou `pytest-mock`
- Utiliser des fixtures pour initialiser ChromaDB en mémoire (pas le vrai `vectorstore/`)
- Chaque test doit être indépendant et reproductible

#### 3.2 — Qualité du code

- Ajouter des **type hints** sur toutes les fonctions publiques de `app/`
- Ajouter des **docstrings** (format Google style) sur toutes les fonctions publiques
- S'assurer que le code passe **`ruff check .`** sans erreur (linter)
- S'assurer que le code passe **`mypy app/ api/`** sans erreur critique

---

### 4. Documentation

#### 4.1 — Mettre à jour `README.md`

Ajouter une section **"Lancement avec l'API backend"** :

```bash
# Terminal 1 — Lancer le backend FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Lancer le frontend Streamlit
streamlit run streamlit_app.py
```

Ajouter une section **"Endpoints API"** avec le tableau des routes.

#### 4.2 — Fichier `CONTRIBUTING.md`

Créer un fichier `CONTRIBUTING.md` expliquant :
- Comment installer l'environnement de dev
- Comment lancer les tests (`pytest tests/`)
- Conventions de nommage et de style (ruff, mypy)
- Comment ouvrir une issue ou une PR

#### 4.3 — Fichier `.env.example`

Vérifier que `.env.example` contient toutes les variables nécessaires, y compris les nouvelles (`API_HOST`, `API_PORT`, `STREAMLIT_API_BASE_URL`).

---

## Contraintes globales à respecter

- **Langage** : Python 3.10+ uniquement
- **Pas de breaking change** : Le Streamlit doit continuer à fonctionner pendant la migration
- **Pas de duplication** : La logique RAG reste dans `app/`, l'API l'appelle — elle ne la réimplémente pas
- **Variables d'environnement** : Aucune clé API ne doit être hardcodée — tout passe par `config.py` et `.env`
- **Dépendances** : Mettre à jour `requirements.txt` après chaque ajout de librairie (`fastapi`, `uvicorn`, `httpx`, `pytest-asyncio`, `ruff`, `mypy`)

---

## Ordre d'exécution recommandé

```
1. Créer la structure api/ (dossiers + __init__.py)
2. Implémenter api/schemas/
3. Implémenter api/routers/documents.py
4. Implémenter api/routers/query.py
5. Implémenter api/main.py (app FastAPI + CORS + routers)
6. Adapter streamlit_app.py pour consommer l'API
7. Écrire les tests (tests/test_api.py en priorité)
8. Ajouter type hints + docstrings dans app/
9. Faire passer ruff et mypy
10. Mettre à jour README.md, CONTRIBUTING.md, .env.example
```

---

## Définition de "terminé" (Definition of Done)

Une tâche est considérée comme terminée quand :
- [ ] Le code est écrit et fonctionne
- [ ] Les tests associés passent (`pytest` vert)
- [ ] Aucune erreur `ruff` ni `mypy`
- [ ] La documentation est à jour
- [ ] Aucune clé API n'est exposée dans le code
