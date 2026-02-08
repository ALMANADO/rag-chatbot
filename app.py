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
    /* Global theme: Soft pastel, clean, inspired by Say Halo's mood-based design */
    .stApp {
        background: linear-gradient(135deg, #FEFEFE 0%, #F4EFF7 100%); /* Subtle white-to-light-pinkish gradient for airy feel */
    }
    
    /* Sidebar enhancements (unchanged as requested) */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Title styling: Soft, centered with subtle shadow */
    h1 {
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        text-align: center;
        font-weight: 300;
    }
    
    /* Subtitle/Caption: Center and style the upload instruction text */
    .stCaption {
        text-align: center !important;
        color: #666666 !important; /* Soft gray for readability */
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0.5rem 0;
        padding: 0 1rem;
        text-shadow: 0 1px 1px rgba(0,0,0,0.05);
    }
    
    /* File uploader and buttons: Soft rounded with pastel accents */
    .stFileUploader label {
        color: #555555;
        font-weight: 400;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #F35E5C, #F4EFF7); /* Inspired by palette pink-to-light */
        color: #333333;
        border: 1px solid rgba(243, 94, 92, 0.2);
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: 400;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Clear Chat button (secondary style) */
    .stButton > button[kind="secondary"] {
        background: rgba(243, 94, 92, 0.1);
        color: #666666;
        border: 1px solid rgba(243, 94, 92, 0.3);
    }
    
    /* Chat messages: Soft rounded cards with pastel backgrounds */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        backdrop-filter: blur(5px);
    }
    
    /* User message: Light pinkish pastel */
    div[role="img"][aria-label="user"] + div > div {
        background: linear-gradient(45deg, #F4EFF7, #FEFEFE);
        color: #333333;
        border-radius: 18px;
        border: 1px solid rgba(244, 239, 247, 0.5);
    }
    
    /* Assistant message: Soft gray pastel */
    div[role="img"][aria-label="assistant"] + div > div {
        background: linear-gradient(45deg, #D6CD1, #B23A4);
        color: #444444;
        border-radius: 18px;
        border: 1px solid rgba(214, 205, 1, 0.3);
    }
    
    /* Warning/Info messages: Soft neutral tones */
    div[data-testid="stAlert"] {
        background: rgba(243, 94, 92, 0.1);
        border: 1px solid rgba(243, 94, 92, 0.3);
        border-radius: 12px;
        color: #666666;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Chat input box: Subtle rounded with light border */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(214, 205, 1, 0.2); /* Light orchid haze border */
        border-radius: 25px;
        color: #333333;
        padding: 0.75rem;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(102, 102, 102, 0.7);
    }
    
    /* Footer text: Muted and subtle */
    .css-1d391kg {
        color: rgba(102, 102, 102, 0.8);
        font-size: 0.9rem;
    }
</style>
    """, unsafe_allow_html=True)

# Title & Description
st.title("🤖 RAG Chatbot with Groq")
st.header("Upload your documents and chat with them using ultra-fast LPU inference.")

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
