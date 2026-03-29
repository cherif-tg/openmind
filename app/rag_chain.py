from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.retriever import get_retriever
from app.llm_factory import get_llm

def build_rag_chain(question: str):
    """Construit et exécute la chaîne RAG complète"""
    llm = get_llm()
    retriever = get_retriever()
    
    # Récupérer les documents pertinents
    relevant_docs = retriever.invoke(question)
    
    # Formater le contexte
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Créer le prompt
    prompt = PromptTemplate(
        template="""Tu es un assistant de recherche expert. Utilise UNIQUEMENT les informations suivantes pour répondre à la question de manière précise et concise.

Contexte:
{context}

Question: {question}

Réponse:""",
        input_variables=["context", "question"]
    )
    
    # Créer la chaîne RAG
    chain = prompt | llm | StrOutputParser()
    
    # Invoquer la chaîne
    response = chain.invoke({"context": context, "question": question})
    
    return response, relevant_docs
    