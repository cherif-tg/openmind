# Hugging Face Spaces — OpenMind RAG (démo)
FROM python:3.11-slim

WORKDIR /code

# Dépendances de la démo (copiées en premier pour profiter du cache Docker)
COPY demo/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code source (app/, demo_corpus/, demo_app.py, config.py)
COPY . .

# Cache persistant des modèles HuggingFace sur le volume /data de l'espace
ENV HF_HOME=/data/hf
RUN mkdir -p /data

EXPOSE 7860

CMD ["streamlit", "run", "demo_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
