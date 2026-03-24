from langchain_chroma import Chroma
from app.embedder import get_embeddings
from config import TOP_K, VECTORSTORE_PATH,COLLECTION_NAME

def get_retriever():
    """retourne le retriever courant

    Returns:
        _type_: retriever de chroma
    """
    modele=get_embeddings()
    vector_store=Chroma(collection_name=COLLECTION_NAME,persist_directory=VECTORSTORE_PATH,embedding_function=modele)
    retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K})
    return retriever
    
    
def load_vectorstore():
    """charge le vectorstore depuis le disque"""
    modele=get_embeddings()
    return Chroma(collection_name=COLLECTION_NAME,persist_directory=VECTORSTORE_PATH,embedding_function=modele)