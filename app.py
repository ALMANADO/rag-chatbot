import streamlit as st
import os
import tempfile
import sys

# SQLite fix for Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Page Config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    /* Global background */
    .stApp {
        background-color: #050816;
    }

    /* Main block width + centering */
    .main > div {
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b1020;
        border-right: 1px solid #20263a;
    }

    /* Chat message bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stChatMessage"][data-testid="stChatMessage-user"] {
        background: #1a2335;
    }
    div[data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] {
        background: #111827;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        background-color: #1f2937;
        color: #f9fafb;
        border: 1px solid #374151;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        border-color: #2563eb;
    }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        border-radius: 8px;
        border: 1px dashed #4b5563;
        background-color: #020617;
    }

    /* Chat input */
    div[data-baseweb="textarea"] > textarea {
        border-radius: 999px !important;
        border: 1px solid #374151 !important;
        padding: 0.75rem 1rem !important;
        background-color: #020617 !important;
        color: #e5e7eb !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title & Description
st.title("🤖 RAG Chatbot with Groq")
st.markdown("Upload your documents and chat with them using ultra-fast LPU inference.")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar: Document Management
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        accept_multiple_files=True,
        type=["pdf", "txt"],
        help="Drag and drop files here. Limit 200MB per file."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("🚀 Process", type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear Chat")

    if clear_btn:
        st.session_state.messages = []
        st.rerun()

    if process_btn:
        if uploaded_files:
            with st.status("Processing documents...", expanded=True) as status:
                st.write("Loading files...")
                docs = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        if uploaded_file.type == "application/pdf":
                            loader = PyPDFLoader(tmp_path)
                        else:
                            loader = TextLoader(tmp_path, encoding="utf-8")
                        docs.extend(loader.load())
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                st.write("Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                st.write("Generating embeddings...")
                embeddings = FakeEmbeddings(size=384)
                
                st.write("Building vector index...")
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings
                )
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                st.success(f"Indexed {len(docs)} documents ({len(splits)} chunks).")
        else:
            st.warning("Please upload files first.")

    st.divider()
    st.info("Powered by Groq LPU Inference | Built with LangChain & Streamlit")

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.retriever is None:
            response = "⚠️ Please upload and process documents in the sidebar first."
        else:
            with st.spinner("Analyzing context..."):
                try:
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    if not groq_api_key:
                        response = "❌ **Error**: `GROQ_API_KEY` is missing. Please add it to Streamlit Secrets."
                    else:
                        llm = ChatGroq(
                            model="llama-3.1-8b-instant",
                            temperature=0.0,
                            api_key=groq_api_key
                        )

                        system_prompt = (
                            "You are a helpful AI assistant. "
                            "Use the provided context to answer the user's question accurately. "
                            "If the answer isn't in the context, say you don't know based on the documents. "
                            "Keep your response professional and helpful.\n\n"
                            "Context:\n{context}"
                        )
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}")
                        ])

                        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

                        result = rag_chain.invoke({"input": prompt})
                        response = result["answer"]
                except Exception as e:
                    response = f"❌ **Error**: {str(e)}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
