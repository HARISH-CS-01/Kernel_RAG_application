from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
import chromadb
from sentence_transformers import CrossEncoder
from ollama import chat

class KernelRAG:
    def multi_query_retriever(self,query):
        vector_path="./ChromaDB"
        client=chromadb.PersistentClient(path=vector_path)
        embed=OllamaEmbeddings(model="mxbai-embed-large")
        vec_store=Chroma(client=client,collection_name="Linux_Kernel_new",embedding_function=embed)
        llm=OllamaLLM(model="llama2")
        retriever=MultiQueryRetriever.from_llm(retriever=vec_store.as_retriever(),llm=llm)
        relevant_document=retriever.invoke(query)
        advanced_rag=[]
        for i in relevant_document:
            advanced_rag.append(i.page_content)
        return advanced_rag
    
    def sentence_ranker(self,query,advanced_rag):
        sentence_model=CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        rank=sentence_model.rank(query,documents=advanced_rag,return_documents=True)
        re_ranked_documents=rank[:4]
        context=f""" """
        for i in re_ranked_documents:
            context=context+i['text']
        return context
    
    def llm_response(self,query,context):
        template=f""" You are a helpful assistant who has little knowledge about the linux kernel development and operating system
          you can use the following content: {context} to answer the question: {query}
          if you don't know the answer, just say that you don't know. Do not try to make up an answer.
          if the question is not related to the content, say that you are tuned to only answer questions that are related to the content.
          """
        system_prompt={'role':'System','content':template}
        response=chat(model="llama2",messages=[system_prompt,{"role":"user","content":query}])
        return response.message.content

    def documents_summarizer(self,query,context):
        template=f""" You are a helpful assistant who can summarize the given documents {context} based on the query: {query}"""
        system_prompt={'role':'System','content':template}
        response=chat(model="llama2",messages=[system_prompt,{"role":"user","content":query}])
        return response.message.content
                          

if __name__=="__main__":
    rag=KernelRAG()
    query=input("Enter your query: ")
    advanced_rag=rag.multi_query_retriever(query)
    context=rag.sentence_ranker(query,advanced_rag)
    summary=rag.documents_summarizer(query,context)
    response=rag.llm_response(query,summary)
    print(response)