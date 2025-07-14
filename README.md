# 🔍 LangChain + Ollama RAG Chatbot (PDF/CSV/Excel)

This is a beginner-friendly chatbot project built using LangChain, Ollama, and Streamlit. It supports general conversation and document-based Q&A from PDF, CSV, and Excel files using vector search and memory.

##  Features

- General chat using Ollama LLM
- Ask questions from uploaded PDF/CSV/Excel
- Uses LangChain's memory for smooth conversations
- Built with RAG (Retrieval-Augmented Generation)
- Streamlit-based web interface

##  How It Works

1. Upload a document (PDF/CSV/Excel)
2. It is split into chunks and embedded
3. User asks a question → relevant chunks retrieved
4. LangChain prompts Ollama with memory + context

##  Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
