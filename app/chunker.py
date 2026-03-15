from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE
)
from typing import List
from langchain_core.documents import Document



def chunk_documents(docs:List[Document],strategy ="recursive",chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP) -> List[Document]:
    """Divise les documents fournit en morceaux de taille prédefini selon la strategie choisit

    Args:
        docs (List[Document]): documents langchain
        strategy (str, optional): la strategy de chunking. Defaults to "recursive".
        chunk_size (_type_, optional): la taille des chunks. Defaults to CHUNK_SIZE.
        chunk_overlap (_type_, optional): le niveau de superposition des chunks. Defaults to CHUNK_OVERLAP.

    Raises:
        ValueError: La strategy choisit n'est pas prise en charge
        ValueError: le document fourni est vide 

    Returns:
        List[Document]: les chunks sous forme de liste de documents langchain
    """
    strategy_list = {
    "recursive": RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "."," ", ""]
        ),
    "character": CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ),
    "token" : TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    }
    
    if strategy not in strategy_list.keys():
        raise ValueError("La strategy choisi n'est pas supporter")  
    splitter =strategy_list[strategy]
    splitter=splitter
    chunks=splitter.split_documents(docs)
    if len(chunks) == 0:
        raise ValueError("Le document fournis est vide..")
    
    for i,doc in enumerate(chunks):
        doc.metadata['index'] = i
        doc.metadata['strategy'] =  strategy
        doc.metadata['chunk_size'] = len(doc.page_content)
    print(f"({len(docs)}) on été transformés en ({len(chunks)}) chunks.")
    
    return chunks
        
    