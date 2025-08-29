# 📚 RAG Application - Linux Kernel Book

An end-to-end **Retrieval-Augmented Generation (RAG)** application built for querying the **Linux Kernel Book** using advanced **embedding, retrieval, re-ranking, and LLM generation** techniques.  

This project demonstrates how to build a knowledge-based chatbot with **custom retrieval and LLM integration**, encapsulated under the `KernelRAG` class for modularity and reusability.

---

## 🚀 Features
- **Chunking:** Recursive text splitter used to split the Linux Kernel book into manageable chunks  
- **Embeddings:** `mxbai-embed-large` model converts text chunks into vector representations  
- **Vector Store:** ChromaDB stores all vectorized chunks for fast retrieval (stored in a separate folder)  
- **Query Processing:**  
  1. User query passed to **Multi-Query Retriever** (LangChain)  
  2. Retriever fetches relevant chunks from ChromaDB  
  3. Fetched results re-ranked using a **Cross-Encoder model** (Hugging Face) based on similarity to query  
  4. Top 4 ranked documents passed to **LLM (LLaMA 2)** via **Ollama service** for final response generation  
- **Modular & scalable:** Entire pipeline encapsulated in `KernelRAG` class, making it easy to integrate or extend  

---


## ⚡ Prerequisites
Before running the application, ensure the following are installed:

1. **Ollama** → Follow instructions: [https://ollama.com/docs](https://ollama.com/docs)  
2. **Pull LLaMA 2 model** for Ollama:
```bash
ollama pull llama2

---
```
## ⚡ Quickstart

1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Kernel_RAG_application.git
cd Kernel_RAG_application

2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Run the application

python Kernel_rag_modules.py



