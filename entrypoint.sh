#!/bin/bash
ollama pull llama2
ollama pull mxbai-embed-large
exec python Kernel_rag_modules.py