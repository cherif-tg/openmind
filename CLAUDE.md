# CLAUDE.md — Système RAG (Retrieval-Augmented Generation)

> Ce fichier est le point d'entrée unique de Claude Code pour ce projet.
> Lis-le intégralement avant toute action. Ne saute aucune section.

---

## 1. Vue d'ensemble du projet

Système RAG end-to-end permettant d'interroger des documents via un LLM.
Le pipeline couvre l'ingestion, le chunking, la retrieval, la génération et l'évaluation.

**Stack principal :**
- Orchestration RAG : LangChain / LangGraph
- Vector store : ChromaDB
- UI : Streamlit
- Langage : Python 3.11 (utiliser `py -3.11` sur Windows si plusieurs versions installées)
- LLM backend : Claude via Anthropic API (ou modèle local via Ollama)

---

## 2. Structure du projet

```
rag-system/
├── CLAUDE.md                   # Ce fichier — lire en premier
├── docs/
│   ├── architecture.md         # Schéma du pipeline RAG complet
│   ├── plan.md                 # Plan de développement avec checkboxes [ ]
│   └── evaluation.md           # Métriques et protocole d'évaluation
├── src/
│   ├── ingestion/              # Chargement et chunking des documents
│   │   ├── loader.py
│   │   └── chunker.py
│   ├── retrieval/              # Embedding + requêtage ChromaDB
│   │   ├── embedder.py
│   │   └── retriever.py
│   ├── generation/             # Prompt templates + appel LLM
│   │   ├── prompt.py
│   │   └── chain.py
│   ├── evaluation/             # RAGAS, métriques, logs
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── app.py                  # Entry point Streamlit
├── tests/
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py
├── notebooks/
│   └── exploration.ipynb
├── data/
│   ├── raw/                    # Documents sources (ne jamais modifier)
│   └── processed/              # Chunks après traitement
├── chroma_db/                  # Persistance ChromaDB (gitignore)
├── mlflow_runs/                # Logs MLflow (gitignore)
├── requirements.txt
└── .env                        # Clés API (jamais commit)
```

> Si la structure réelle diffère, adapte-toi à ce qui existe — ne recrée pas de fichiers déjà présents.

---

## 3. Phase actuelle : Phase 3 — Génération & Évaluation

Le projet est en Phase 3. Les phases 1 (ingestion/chunking) et 2 (retrieval) sont terminées ou en cours de stabilisation.

**Objectifs de la Phase 3 :**
- [ ] Construire les prompt templates (system prompt, contexte, question)
- [ ] Implémenter la chain LangChain : retriever → prompt → LLM → output parser
- [ ] Intégrer la gestion du contexte multi-documents (context stuffing vs. map-reduce)
- [ ] Mettre en place le pipeline d'évaluation avec RAGAS
- [ ] Logger les runs avec MLflow (question, contexte récupéré, réponse, métriques)
- [ ] Exposer le pipeline via l'interface Streamlit

**Ne pas :**
- Refactorer le code d'ingestion ou de retrieval sans raison explicite
- Changer la configuration ChromaDB existante
- Ajouter des dépendances non listées dans `requirements.txt` sans le signaler

---

## 4. Règles de développement

### 4.1 Avant de coder
1. Lire `docs/plan.md` et identifier la tâche avec checkbox `[ ]` concernée
2. Identifier les fichiers impactés — ne toucher que ceux nécessaires
3. Si la tâche est ambiguë, demander une clarification AVANT d'écrire du code
4. Ne jamais supposer la structure d'un fichier existant — le lire d'abord avec Read

### 4.2 Standards de code Python
- Python 3.11 strict — pas de syntax 3.12+
- Type hints obligatoires sur toutes les fonctions publiques
- Docstrings en français (style Google) sur toutes les classes et fonctions publiques
- Nommage : `snake_case` pour fonctions/variables, `PascalCase` pour classes, `UPPER_SNAKE_CASE` pour constantes
- Longueur de ligne max : 100 caractères
- Pas de `print()` en production — utiliser `logging` (niveau approprié)
- Toujours utiliser `pathlib.Path` pour les chemins, jamais `os.path`

### 4.3 Gestion des erreurs
- Toujours wrapper les appels LLM et ChromaDB dans des blocs try/except explicites
- Logger les exceptions avec le contexte (question, chunk_id, etc.)
- Ne jamais swallower une exception silencieusement (`except: pass` interdit)

### 4.4 Variables d'environnement
- Toutes les clés API et chemins sensibles viennent de `.env` via `python-dotenv`
- Ne jamais hardcoder une clé, un chemin absolu, ou un nom de modèle en dur dans le code
- Utiliser un fichier `config.py` centralisé pour les constantes configurables

### 4.5 ChromaDB
- Collection name : toujours lire depuis la config, jamais hardcodé
- Persistance : utiliser le chemin défini dans `.env` / `config.py`
- Ne jamais supprimer ou recréer la collection sans confirmation explicite
- Toujours vérifier que la collection existe avant une opération de lecture

### 4.6 LangChain
- Utiliser `LCEL` (LangChain Expression Language) avec le pipe `|` pour les chains
- Pas de classes `LLMChain` dépréciées — utiliser `RunnableSequence`
- Les prompt templates doivent être dans `src/generation/prompt.py`, jamais inline
- Toujours inclure un `output_parser` explicite

---

## 5. Pipeline RAG — Architecture cible

```
Question utilisateur
      │
      ▼
[Embedder] → vecteur requête
      │
      ▼
[ChromaDB Retriever] → top-k chunks (k configurable)
      │
      ▼
[Reranker optionnel] → chunks réordonnés
      │
      ▼
[Prompt Builder] → system prompt + contexte + question
      │
      ▼
[LLM Chain] → réponse brute
      │
      ▼
[Output Parser] → réponse structurée
      │
      ▼
[Evaluator RAGAS] → faithfulness, answer_relevancy, context_recall
      │
      ▼
[MLflow Logger] → log du run complet
      │
      ▼
[Streamlit UI] → affichage réponse + sources + métriques
```

---

## 6. Évaluation — Protocole

Framework principal : **RAGAS**

**Métriques à tracker obligatoirement :**
| Métrique | Description | Cible |
|---|---|---|
| `faithfulness` | Réponse ancrée dans le contexte | > 0.80 |
| `answer_relevancy` | Réponse pertinente à la question | > 0.75 |
| `context_recall` | Contexte récupéré couvre la réponse | > 0.70 |
| `context_precision` | Contexte récupéré est précis | > 0.70 |

**Logging MLflow :**
- Chaque run doit logger : question, réponse, chunks récupérés, toutes les métriques RAGAS, paramètres (k, modèle, temperature)
- Experiment name : `rag_evaluation`
- Ne jamais logger de données personnelles dans MLflow

**Dataset d'évaluation :**
- Fichier : `data/eval_dataset.json` (format RAGAS : question + ground_truth + contexts)
- Ne pas modifier le dataset pendant une session d'évaluation

---

## 7. Interface Streamlit

**Règles spécifiques :**
- Un seul fichier entry point : `src/app.py`
- Utiliser `st.session_state` pour maintenir l'historique de conversation
- Afficher systématiquement les sources (nom du document + chunk) avec chaque réponse
- Afficher les métriques RAGAS si le mode debug est activé (`st.sidebar`)
- Pas de `st.experimental_*` — utiliser les APIs stables uniquement
- Le spinner (`st.spinner`) doit wrapper tous les appels LLM et retrieval

---

## 8. Tests

- Lancer les tests avec : `python -m pytest tests/ -v`
- Chaque nouvelle fonction dans `src/` doit avoir au moins un test unitaire
- Mocker les appels LLM et ChromaDB dans les tests (pas d'appels réels en CI)
- Les tests d'évaluation RAGAS sont dans `tests/test_evaluation.py` et peuvent être lents — les taguer avec `@pytest.mark.slow`

---

## 9. Workflow de travail avec Claude Code

### À chaque session
1. Lire `docs/plan.md` — identifier la prochaine checkbox `[ ]`
2. Lire les fichiers concernés avant de les modifier
3. Coder → tester → cocher `[x]` dans `docs/plan.md`
4. Ne passer à la tâche suivante que si les tests passent

### Commits
- Format : `type(scope): description` (conventional commits)
- Types : `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Exemples : `feat(generation): add LCEL chain with output parser`, `fix(retrieval): handle empty ChromaDB collection`
- Ne jamais committer `.env`, `chroma_db/`, `mlflow_runs/`

### Gestion du contexte
- Si le contexte devient trop long (session longue), utiliser `/clear` et reprendre depuis `docs/plan.md`
- Stocker les décisions d'architecture importantes dans `docs/architecture.md`, pas dans la conversation

---

## 10. Dépendances clés

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-anthropic>=0.3.0   # ou langchain-ollama pour local
chromadb>=0.5.0
ragas>=0.2.0
mlflow>=2.14.0
streamlit>=1.35.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

> Avant d'ajouter une nouvelle dépendance, vérifier qu'elle n'existe pas déjà sous un autre nom dans `requirements.txt`.

---

## 11. Ce que Claude Code NE DOIT PAS faire

- ❌ Supprimer ou écraser `chroma_db/` sans confirmation explicite
- ❌ Modifier `data/raw/` (documents sources immuables)
- ❌ Hardcoder des clés API ou des chemins absolus
- ❌ Utiliser des APIs LangChain dépréciées (`LLMChain`, `ConversationalRetrievalChain` legacy)
- ❌ Ajouter des dépendances sans les ajouter aussi dans `requirements.txt`
- ❌ Refactorer du code de phases antérieures sans raison explicite et validation
- ❌ Committer des fichiers listés dans `.gitignore`
- ❌ Supposer qu'un fichier a un certain contenu sans l'avoir lu au préalable

---

*Dernière mise à jour : Phase 3 — Génération & Évaluation*
*Pour mettre à jour ce fichier : modifier directement CLAUDE.md à la racine du projet.*
