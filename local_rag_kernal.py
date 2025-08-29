from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import embed
import chromadb
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from ollama import chat

#file_path="/Users/harishv/AI_projects/Rag_project_kernal/linux_kernal_docs.pdf"
#loader=PyPDFLoader(file_path)
#documents=loader.load()
#print(len(documents))
#print(documents[0])

vector_path="/Users/harishv/AI_projects/Rag_project_kernal/ChromaDB"
#client=chromadb.Client()
client=chromadb.PersistentClient(path=vector_path)
#collection=client.get_collection(name="Linux_Kernel_new")
embed=OllamaEmbeddings(model="mxbai-embed-large")
vec_store=Chroma(client=client,collection_name="Linux_Kernel_new",embedding_function=embed)

llm=OllamaLLM(model="llama2")
retriever=MultiQueryRetriever.from_llm(retriever=vec_store.as_retriever(),llm=llm)
print("Enter your query regarding the linux kernel development")
query=input()
relevant_document=retriever.invoke(query)
advanced_rag=[]
for i in relevant_document:
    advanced_rag.append(i.page_content)
sentence_model=CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
rank=sentence_model.rank(query,documents=advanced_rag,return_documents=True)
re_ranked_documents=rank[:4]
context=f""" """
for i in re_ranked_documents:
    context=context+i['text']
template=f""" You are a helpful assistant who has little knowledge about the linux kernel development and operating system
          you can use the following content: {context} to answer the question: {query}
          if you don't know the answer, just say that you don't know. Do not try to make up an answer.
          if the question is not related to the content, say that you are tuned to only answer questions that are related to the content.
          """
system_prompt={'role':'System','content':template}
response=chat(model="llama2",messages=[system_prompt,{"role":"user","content":query}])
print(response.message.content)