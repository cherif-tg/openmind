# Le re-ranking

Le re-ranking est une seconde passe de tri qui réordonne les chunks récupérés
par la recherche vectorielle.

## Pourquoi ?

La recherche par embeddings peut renvoyer des résultats approximatifs. Un
modèle de re-ranking (cross-encoder) compare directement la question et chaque
passage, ce qui donne un classement plus fin et plus pertinent.

## Exemple de modèle

- **cross-encoder/ms-marco-MiniLM-L-6-v2** : modèle local et gratuit qui
  attribue un score à chaque paire (question, passage).

## Effet sur la qualité

Le re-ranking améliore la précision du contexte envoyé au LLM, et donc la
qualité et la fidélité des réponses finales. C'est ce qui distingue un RAG
« simple » d'un RAG sérieux.
