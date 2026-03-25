from langchain.prompts import PromptTemplate
from app.retriever import get_retriever
from langchain.chains.retrieval_qa import RetrievalQA
from app.llm_factory import get_llm

def build_rag_chain(question:str):
    llm = get_llm()
    
    retriver =  get_retriever()
    context=retriver.invoke(question)
    
    prompt=PromptTemplate(
        template="Tu es un assistant de recherche. Utilise les informations suivantes pour répondre à la question: {context} Question: {question}",
        input_variables=["context", "question"]
    )
    chains=RetrievalQA.from_chain_type(llm=llm,
                                       retriever=retriver,
                                       chain_type="stuff",
                                       chains_type_kwargs={"prompt": prompt},
                                       return_source_documents=True
 )
    result=chains.invoke({"query": question})
    reponse=result['result']
    sources=result['source_documents']
    return reponse, sources
    