FROM python:3.13-slim
RUN apt-get update && apt install -y curl && apt install -y python3-pip
WORKDIR /RAG
COPY ChromaDB/ ChromaDB/
COPY Kernel_rag_modules.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY entrypoint.sh .
CMD ["python","Kernel_rag_modules.py"]