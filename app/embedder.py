from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document 
from typing import List
from config import (
    EMBEDDING_DEVICE,EMBEDDING_MODEL,
    VECTORSTORE_PATH,COLLECTION_NAME)
from dotenv import load_dotenv
load_dotenv()

def get_embedded():
    """retourne le modele d'embeddings courant

    Returns:
        _type_: modele d'embedding huggingface
    """
    modele=HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device':EMBEDDING_DEVICE}
    )
    return modele

def embedded_document(chunks:List[Document]):
    """Prend un list de chunk langchain et retourne les vecteurs stockées dans un vectorstore

    Args:
        chunks (List[Document]): List dedocuments langchain
    """
    modele=get_embedded()
    chroma = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=modele,
        persist_directory=VECTORSTORE_PATH
    )
    
    vector_store=Chroma.from_documents(chunks,embedding=modele,
                                       collection_name=COLLECTION_NAME,persist_directory=VECTORSTORE_PATH)
    
    return vector_store
    