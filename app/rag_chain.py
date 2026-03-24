from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from app.retriever import get_retriever
from app.llm_factory import get_llm
from langchain_core.documents import Document
from typing import List

def build_rag_chain(question:str):
    llm = get_llm()
    
    retriver =  get_retriever()
    context=retriver.invoke(question)
    
    prompt=f"""Tu es un assistant , tu ne doit répondre aux questions en te basant uniquement sur les documents fournis.Si tu n'a pas la reponse dis juste:"Je n'ai pas assez d'information pour repondre a la question."
    Documents:
    {context}
    Question:
    {question}
    Tu dois donner retourner la reponse a la question en plus de la source d'ou proviens la réponse
    """
    reponse=llm.invoke(prompt)
    return reponse
    