# Le RAG (Retrieval-Augmented Generation)

Le RAG est une technique qui combine la recherche d'informations (retrieval)
et la génération de texte par un grand modèle de langage (LLM).

## Pipeline classique

1. **Ingestion** : les documents sont découpés en morceaux appelés chunks.
2. **Embeddings** : chaque chunk est transformé en vecteur numérique.
3. **Indexation** : les vecteurs sont stockés dans une base vectorielle.
4. **Retrieval** : à partir de la question, on récupère les chunks les plus proches.
5. **Génération** : le LLM rédige une réponse à partir des chunks récupérés.

## Avantages

- Réduit les hallucinations en ancrant la réponse dans des sources réelles.
- Permet d'interroger des documents privés sans ré-entraîner le modèle.
- Les réponses peuvent citer leurs sources, ce qui les rend vérifiables.

## Limites

- La qualité dépend du découpage (chunking) et de la pertinence du retrieval.
- Un mauvais découpage peut couper une information importante en deux.
