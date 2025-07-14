import os
import tempfile
import streamlit as st
import pandas as pd

from langchain_community.llms import Ollama
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document

# Streamlit setup
st.set_page_config(page_title="Hybrid Chatbot", layout="centered")
st.title("Chatbot")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_chain" not in st.session_state:
    st.session_state.doc_chain = None
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

# Load lightweight LLM and memory 
llm = Ollama(model="phi3:mini")  
memory = ConversationBufferMemory()
general_chain = ConversationChain(llm=llm, memory=memory, verbose=False)

# File uploader
uploaded_file = st.file_uploader("📁 Upload PDF, CSV, or Excel", type=["pdf", "csv", "xlsx"])

if uploaded_file:
    st.session_state.doc_uploaded = True

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            pages = loader.load()
        elif ext == ".csv":
            loader = CSVLoader(file_path)
            pages = loader.load()
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            text = df.to_csv(index=False)
            pages = [Document(page_content=text)]
        else:
            st.error("❌ Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"❌ File load error: {e}")
        st.stop()

    # Split and embed documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model="phi3:mini")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Prompt and memory for doc-based chat
    doc_prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
You are a helpful assistant. Use the context below to answer the user's question.
If the context doesn't have the answer, respond politely and generally.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Helpful Answer:""",
    )

    doc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    doc_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=doc_memory,
        combine_docs_chain_kwargs={"prompt": doc_prompt},
        verbose=False,
    )

    st.session_state.doc_chain = doc_chain
    st.success("📄 File processed. You can now ask questions from it!")

# Chat input
user_input = st.chat_input("Type a question (general or document-related)...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            if st.session_state.doc_uploaded and any(word in user_input.lower() for word in ["file", "pdf", "excel", "csv", "table", "data", "sheet"]):
                result = st.session_state.doc_chain.invoke({"question": user_input})
            else:
                response = general_chain.run(user_input)
                result = {"answer": response}
        except Exception as e:
            result = {"answer": f"⚠️ Error: {e}"}

        answer = result.get("answer", "🤖 Sorry, I couldn't generate a response.")
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for role, msg in st.session_state.chat_history:
    st.chat_message("user" if role == "You" else "assistant").write(msg)
