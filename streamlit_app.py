#!/usr/bin/env python3
"""
OpenMind RAG - Interface Streamlit
Interface conversationnelle pour interroger vos documents via l'API FastAPI
"""

import streamlit as st
import httpx
from pathlib import Path
from config import STREAMLIT_API_BASE_URL, TOP_K, CHUNK_SIZE, LLM_MODE
import os

# ── Configuration de la page ─────────────────────────────────
st.set_page_config(
    page_title="OpenMind RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Initialisation du state ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "llm_mode" not in st.session_state:
    st.session_state.llm_mode = LLM_MODE

# ── Configuration de l'API ──────────────────────────────────
API_BASE_URL = STREAMLIT_API_BASE_URL or os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")

# ── Styles CSS personnalisés ──────────────────────────────────
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Fonctions utilitaires ─────────────────────────────────────
def check_api_health():
    """Vérifie que l'API est accessible"""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def get_api_documents():
    """Récupère la liste des documents indexés via l'API"""
    try:
        response = httpx.get(f"{API_BASE_URL}/api/documents/", timeout=10.0)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def upload_documents(files):
    """Upload des documents via l'API"""
    try:
        # Préparer les fichiers pour l'upload
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), "application/octet-stream")))

        response = httpx.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files_data,
            timeout=300.0  # Timeout long pour l'indexation
        )

        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Erreur lors de l'upload : {e}")
        return None


def delete_document(filename):
    """Supprime un document via l'API"""
    try:
        response = httpx.delete(
            f"{API_BASE_URL}/api/documents/{filename}",
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Erreur lors de la suppression : {e}")
        return None


def query_rag(question, top_k=TOP_K, llm_mode=None):
    """Pose une question via l'API RAG"""
    try:
        payload = {
            "question": question,
            "top_k": top_k
        }
        if llm_mode:
            payload["llm_mode"] = llm_mode

        response = httpx.post(
            f"{API_BASE_URL}/api/query/",
            json=payload,
            timeout=120.0  # Timeout pour la génération
        )

        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Erreur lors de la requête : {e}")
        return None


def format_sources(sources_data):
    """Formate l'affichage des sources depuis la réponse API"""
    if not sources_data:
        return []

    formatted = []
    for source in sources_data:
        formatted.append({
            "file": source.get("metadata", {}).get("source", "unknown"),
            "chunk": source.get("metadata", {}).get("chunk", 0),
            "content": source.get("content", "")[:200]
        })
    return formatted


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title(" OpenMind RAG")
    st.markdown("---")

    # Vérification de la santé de l'API
    api_healthy = check_api_health()
    if api_healthy:
        st.success(" API connectée")
    else:
        st.error(" API non disponible")
        st.warning("Assurez-vous de lancer : `uvicorn api.main:app --reload`")

    st.markdown("---")

    # Section: Configuration LLM
    st.subheader(" Configuration")

    llm_options = {
        "groq": f"Groq API",
        "ollama": f"Ollama Local",
        "huggingface": f"HuggingFace"
    }

    selected_mode = st.selectbox(
        "Modèle LLM",
        options=list(llm_options.keys()),
        format_func=lambda x: llm_options[x],
        index=list(llm_options.keys()).index(st.session_state.llm_mode) if st.session_state.llm_mode in llm_options else 0
    )

    if selected_mode != st.session_state.llm_mode:
        st.session_state.llm_mode = selected_mode

    # Section: Upload de documents
    st.subheader(" Documents")

    uploaded_files = st.file_uploader(
        "Uploader des documents",
        type=["pdf", "csv", "docx", "txt", "html"],
        accept_multiple_files=True,
        help="Formats supportés: PDF, CSV, DOCX, TXT, HTML"
    )

    if uploaded_files:
        progress_bar = st.progress(0)

        if st.button("Indexer les documents", type="primary"):
            with st.spinner("Indexation en cours..."):
                results = upload_documents(uploaded_files)
                if results:
                    st.success(f"{len(results)} document(s) indexé(s) avec succès!")
                    for result in results:
                        st.session_state.indexed_docs.append({
                            "name": result["filename"],
                            "chunks": result["chunks_count"]
                        })
                    st.session_state.vectorstore_loaded = True
                else:
                    st.error("Échec de l'indexation")
                progress_bar.empty()

    # Récupérer les documents depuis l'API
    if st.button("Actualiser la liste"):
        docs_data = get_api_documents()
        if docs_data:
            st.session_state.indexed_docs = [
                {"name": doc["filename"], "chunks": doc["chunks_count"]}
                for doc in docs_data.get("documents", [])
            ]
            st.session_state.vectorstore_loaded = len(docs_data.get("documents", [])) > 0
            st.success("Liste actualisée!")

    # Afficher les documents indexés
    if st.session_state.indexed_docs:
        st.markdown("---")
        st.markdown("**Documents indexés:**")
        for doc in st.session_state.indexed_docs:
            st.markdown(f"   {doc['name']} ({doc['chunks']} chunks)")

    # Bouton pour réinitialiser
    st.markdown("---")
    if st.button(" Réinitialiser le vectorstore"):
        # Supprimer tous les documents un par un
        docs_data = get_api_documents()
        if docs_data and docs_data.get("documents"):
            for doc in docs_data["documents"]:
                delete_document(doc["filename"])
            st.session_state.indexed_docs = []
            st.session_state.vectorstore_loaded = False
            st.success("Vectorstore réinitialisé!")
        else:
            st.info("Aucun document à supprimer")

    # Section: URL (non implémentée dans l'API pour l'instant)
    st.markdown("---")
    st.info(" L'import d'URL sera disponible prochainement")

    # Infos système
    st.markdown("---")
    st.markdown("**Statut du vectorstore:**")
    status_color = "🟢" if st.session_state.vectorstore_loaded else "🔴"
    status_text = "Prêt" if st.session_state.vectorstore_loaded else "Non chargé"
    st.markdown(f"{status_color} {status_text}")

    if st.session_state.vectorstore_loaded and docs_data:
        total_chunks = sum(doc["chunks_count"] for doc in docs_data.get("documents", []))
        st.markdown(f"**Total chunks:** {total_chunks}")

# ── Main ──────────────────────────────────────────────────────
st.title(" Assistant de Recherche")
st.markdown("Posez vos questions en langage naturel sur vos documents")

# Vérifier la connexion API
if not api_healthy:
    st.error("L'API n'est pas disponible. Veuillez lancer le serveur FastAPI.")
    st.code("uvicorn api.main:app --reload --port 8000", language="bash")
else:
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Voir les sources"):
                    for source in message["sources"]:
                        st.markdown(f"** {source['file']}** (chunk {source['chunk']})")
                        st.markdown(f"_{source['content']}..._")
                        st.markdown("---")

    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                response_data = query_rag(
                    question=prompt,
                    top_k=TOP_K,
                    llm_mode=st.session_state.llm_mode
                )

                if response_data:
                    # Afficher la réponse
                    st.markdown(response_data["answer"])

                    # Afficher les sources
                    if response_data.get("sources"):
                        with st.expander(" Voir les sources"):
                            for source in response_data["sources"]:
                                st.markdown(f"** {source['metadata'].get('source', 'unknown')}** (chunk {source['metadata'].get('chunk', 0)})")
                                st.markdown(f"_{source['content']}..._")
                                st.markdown("---")

                    # Sauvegarder dans l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data["answer"],
                        "sources": format_sources(response_data.get("sources", []))
                    })
                else:
                    st.error("Erreur lors de la génération de la réponse. Vérifiez les logs de l'API.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Désolé, une erreur est survenue lors de la génération de la réponse."
                    })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Mode:** " + st.session_state.llm_mode.upper())
with col2:
    st.markdown(f"**Top-K:** {TOP_K}")
with col3:
    st.markdown("**Chunk Size:** " + str(CHUNK_SIZE))
