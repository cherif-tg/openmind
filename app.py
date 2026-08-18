"""Démo Gradio monoprocess OpenMind RAG - prête pour Hugging Face Spaces.

Le pipeline RAG tourne entièrement dans ce processus (pas de serveur
FastAPI séparé). Le corpus de démonstration est indexé en mémoire au premier
appel, puis les questions sont traitées via retrieval + re-ranking +
génération avec citations.

Compatible ZeroGPU (Hugging Face Spaces) : la fonction d'inférence est
décorée avec `@spaces.GPU`.
"""

from pathlib import Path

import gradio as gr
import spaces
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.chunker import chunk_documents
from app.embedder import get_embeddings
from app.rag_chain import build_rag_chain
from config import COLLECTION_NAME, GROQ_MODEL, RETRIEVAL_K, TOP_K

DEMO_CORPUS_DIR = "demo_corpus"

_retriever = None


def _get_retriever():
    """Indexe (une seule fois) le corpus de démo en mémoire."""
    global _retriever
    if _retriever is None:
        docs = []
        for path in sorted(Path(DEMO_CORPUS_DIR).glob("*.md")):
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
        chunks = chunk_documents(docs, strategy="recursive")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    return _retriever


def _format_sources(documents) -> str:
    if not documents:
        return "Aucune source."
    blocks = []
    for i, doc in enumerate(documents, start=1):
        filename = doc.metadata.get("filename", "inconnu")
        blocks.append(f"[{i}] {filename}\n{doc.page_content[:250]}")
    return "\n\n---\n\n".join(blocks)


@spaces.GPU(duration=120)
def answer(question: str, top_k: int, rerank: bool, history):
    """Répond à une question et met à jour l'historique de conversation."""
    history = history or []
    if not question or not question.strip():
        return history, "Posez une question."

    try:
        answer_text, docs = build_rag_chain(
            question,
            top_k=int(top_k),
            llm_mode="groq",
            rerank=rerank,
            retriever=_get_retriever(),
        )
        sources = _format_sources(docs)
    except Exception as e:  # noqa: BLE001
        answer_text = f"Erreur lors de la génération : {e}"
        sources = "-"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_text},
    ]
    return history, sources


# Pré-chargement du corpus au démarrage : évite de télécharger les modèles
# pendant le premier appel (soumis à la limite de temps de ZeroGPU).
try:
    _get_retriever()
    from app.reranker import get_reranker

    get_reranker()
except Exception as e:  # noqa: BLE001
    print(f"Pré-chargement différé (sera retenté au premier appel) : {e}")


with gr.Blocks(title="OpenMind RAG - Démo") as demo:
    gr.Markdown("# 🧠 OpenMind RAG - Démo")
    gr.Markdown(
        "Posez une question sur le corpus de démo (RAG, embeddings, re-ranking). "
        "La réponse cite ses sources sous la forme `[1]`, `[2]`, …"
    )

    chatbot = gr.Chatbot(label="Conversation", height=420)

    with gr.Row():
        question = gr.Textbox(
            label="Votre question",
            placeholder="Ex : Qu'est-ce que le re-ranking ?",
            scale=4,
        )
        submit = gr.Button("Envoyer", scale=1)

    with gr.Row():
        top_k = gr.Slider(1, 10, value=TOP_K, step=1, label="Chunks finaux (top-k)")
        rerank = gr.Checkbox(value=True, label="Activer le re-ranking (cross-encoder)")

    sources = gr.Textbox(label="Sources citées", interactive=False)

    submit.click(answer, [question, top_k, rerank, chatbot], [chatbot, sources])
    question.submit(answer, [question, top_k, rerank, chatbot], [chatbot, sources])

    gr.Markdown(f"LLM : Groq (`{GROQ_MODEL}`). Clé via le secret `GROQ_API_KEY`.")


if __name__ == "__main__":
    demo.launch()
