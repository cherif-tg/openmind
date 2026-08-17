"""Démo monoprocess OpenMind RAG — prête pour Hugging Face Spaces.

Le pipeline RAG tourne entièrement dans ce processus (pas de serveur
FastAPI séparé). Le corpus de démonstration est indexé en mémoire au
démarrage, puis les questions sont traitées via retrieval + re-ranking +
génération avec citations.
"""

from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.chunker import chunk_documents
from app.embedder import get_embeddings
from app.rag_chain import build_rag_chain
from config import COLLECTION_NAME, RETRIEVAL_K, TOP_K

DEMO_CORPUS_DIR = "demo_corpus"

st.set_page_config(
    page_title="OpenMind RAG — Démo",
    page_icon="🧠",
    layout="wide",
)


def load_demo_corpus(folder: str = DEMO_CORPUS_DIR) -> list[Document]:
    """Charge les fichiers Markdown du corpus de démonstration."""
    docs: list[Document] = []
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "filename": path.name,
                    "file_type": ".md",
                    "source": str(path),
                },
            )
        )
    return docs


@st.cache_resource(show_spinner="Chargement du modèle et indexation du corpus…")
def get_demo_retriever():
    """Indexe le corpus de démo en mémoire et retourne un retriever."""
    docs = load_demo_corpus()
    chunks = chunk_documents(docs, strategy="recursive")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def run_query(question: str, top_k: int, rerank: bool):
    """Exécute le pipeline RAG complet sur le corpus de démo."""
    retriever = get_demo_retriever()
    return build_rag_chain(
        question,
        top_k=top_k,
        llm_mode="groq",
        rerank=rerank,
        retriever=retriever,
    )


# --- Interface ------------------------------------------------------------

st.title("🧠 OpenMind RAG — Démo")
st.markdown(
    "Posez une question sur le corpus de démo (RAG, embeddings, re-ranking). "
    "La réponse cite ses sources sous la forme `[1]`, `[2]`, …"
)

with st.sidebar:
    st.header("Paramètres")
    top_k = st.slider("Chunks finaux (top-k)", 1, 10, TOP_K)
    rerank = st.checkbox("Activer le re-ranking (cross-encoder)", value=True)
    st.markdown("---")
    st.caption("LLM : Groq (`llama-3.3-70b-versatile`). Clé définie via le secret `GROQ_API_KEY`.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}]** {source['file']} (chunk {source['chunk']})")
                    st.markdown(f"_{source['content']}_")
                    st.markdown("---")

if prompt := st.chat_input("Votre question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche + génération…"):
            try:
                answer, docs = run_query(prompt, top_k=top_k, rerank=rerank)
            except Exception as e:  # noqa: BLE001
                answer = f"Erreur lors de la génération : {e}"
                docs = []

        st.markdown(answer)

        sources = []
        if docs:
            with st.expander("Sources"):
                for i, doc in enumerate(docs, start=1):
                    filename = doc.metadata.get("filename", "inconnu")
                    chunk = doc.metadata.get("index", 0)
                    st.markdown(f"**[{i}]** {filename} (chunk {chunk})")
                    st.markdown(f"_{doc.page_content[:300]}_")
                    st.markdown("---")
            sources = [
                {
                    "file": doc.metadata.get("filename", "inconnu"),
                    "chunk": doc.metadata.get("index", 0),
                    "content": doc.page_content[:300],
                }
                for doc in docs
            ]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
