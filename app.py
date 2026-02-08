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
    /* Main background - soft gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area */
    .main > div {
        max-width: 1200px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }

    /* Sidebar - vibrant purple */
    section[data-testid="stSidebar"] {
        background: linear-gradient(145deg, #8b5cf6, #7c3aed);
        border-right: 1px solid #a78bfa;
        border-radius: 0 20px 20px 0;
    }

    /* Headers */
    h1 {
        color: #1e293b !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* FABULOUS Chat Bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 18px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* User messages - bright blue */
    [data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
    }
    
    /* Assistant messages - emerald green */
    [data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
    }

    /* Buttons - magical hover effects */
    .stButton>button {
        width: 100%;
        height: 3.5em;
        border-radius: 12px;
        background: linear-gradient(45deg, #f59e0b, #d97706);
        color: white;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.6);
        background: linear-gradient(45deg, #fbbf24, #f59e0b);
    }
    .stButton>button[type="primary"] {
        background: linear-gradient(45deg, #ef4444, #dc2626) !important;
    }
    .stButton>button[type="primary"]:hover {
        background: linear-gradient(45deg, #f87171, #ef4444) !important;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.6);
    }

    /* File uploader - clean glass effect */
    [data-testid="stFileUploader"] section {
        border-radius: 12px;
        border: 2px dashed #60a5fa;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
    }

    /* Chat input - rounded pill */
    .stChatInput > div > div > div > div {
        border-radius: 25px !important;
        border: 2px solid #f3f4f6 !important;
        background: white !important;
        padding: 0.75rem 1.5rem !important;
    }
    .stChatInput input {
        border-radius: 25px !important;
        border: none !important;
        color: #1f293b !important;
        font-size: 1rem;
    }

    /* Status messages */
    .stStatus {
        background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }

    /* Success messages */
    div[role="alert"] div {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-radius: 10px;
        border-left: 4px solid #10b981;
    }

    /* Warnings */
    .stAlert {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
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
