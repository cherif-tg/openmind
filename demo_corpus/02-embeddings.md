# Les embeddings

Un embedding est une représentation vectorielle d'un texte. Deux textes au
sens proche produisent des vecteurs proches dans l'espace vectoriel.

## Modèles

- **sentence-transformers/all-MiniLM-L6-v2** : modèle léger de 384 dimensions,
  local et gratuit, utilisé par OpenMind RAG.
- Des modèles plus grands (ex. mpnet-base) offrent plus de précision mais
  demandent davantage de calcul.

## Mesure de similarité

La similarité cosinus mesure l'angle entre deux vecteurs. Plus le score est
proche de 1, plus les textes sont proches sémantiquement.

## Usage dans le RAG

Les embeddings servent à retrouver les chunks les plus proches d'une question.
C'est l'étape de retrieval.
