import streamlit as st
import os
import tempfile
import sys
import zipfile
import xml.etree.ElementTree as ET

# SQLite fix for Streamlit Cloud (Chroma sometimes needs this)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from docx import Document as DocxDocument
from langchain_core.documents import Document as LCDocument

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# DOCX Helpers (rich extraction)
# -----------------------------
def _extract_footnotes_from_docx(docx_path: str) -> str:
    """
    Best-effort extraction of footnotes from a .docx file by reading word/footnotes.xml.
    If missing or parsing fails, returns empty string.
    """
    try:
        with zipfile.ZipFile(docx_path) as z:
            if "word/footnotes.xml" not in z.namelist():
                return ""

            xml_bytes = z.read("word/footnotes.xml")
            root = ET.fromstring(xml_bytes)

            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            texts = [node.text for node in root.findall(".//w:t", ns) if node.text]

            footnotes_text = " ".join(texts).strip()
            return footnotes_text
    except Exception:
        return ""


def load_docx_rich(docx_path: str, source_name: str = None) -> list[LCDocument]:
    """
    Extract DOCX content with:
      - Headers & footers (per section)
      - Paragraphs (headings marked)
      - Tables (rows with ' | ')
      - Footnotes (best-effort)
    Returns list of LangChain Documents.
    """
    doc = DocxDocument(docx_path)
    parts = []

    # Headers & Footers
    for si, section in enumerate(doc.sections):
        header_texts = [p.text.strip() for p in section.header.paragraphs if p.text and p.text.strip()]
        if header_texts:
            parts.append(f"\n[HEADER section {si + 1}]\n" + "\n".join(header_texts))

        footer_texts = [p.text.strip() for p in section.footer.paragraphs if p.text and p.text.strip()]
        if footer_texts:
            parts.append(f"\n[FOOTER section {si + 1}]\n" + "\n".join(footer_texts))

    # Body paragraphs (mark headings)
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        style_name = (p.style.name if p.style else "").lower()
        if "heading" in style_name:
            parts.append(f"\n[HEADING] {text}\n")
        else:
            parts.append(text)

    # Tables
    for ti, table in enumerate(doc.tables):
        table_lines = [f"\n[TABLE {ti + 1}]"]
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                # Flatten line breaks inside a cell
                cell_text = " ".join(t.strip() for t in cell.text.splitlines() if t.strip()).strip()
                row_cells.append(cell_text)
            table_lines.append(" | ".join(row_cells))
        parts.append("\n".join(table_lines))

    # Footnotes (best-effort)
    footnotes_text = _extract_footnotes_from_docx(docx_path)
    if footnotes_text:
        parts.append(f"\n[FOOTNOTES]\n{footnotes_text}")

    full_text = "\n".join(parts).strip()

    return [
        LCDocument(
            page_content=full_text,
            metadata={
                "source": source_name or os.path.basename(docx_path),
                "file_type": "docx",
            },
        )
    ]


# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

html, body { color-scheme: dark !important; }
# Custom CSS (your original, unchanged)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FEFEFE 0%, #F4EFF7 100%);
    }
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    h1 {
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        text-align: center;
        font-weight: 300;
    }
    .stCaption {
        text-align: center !important;
        color: #666666 !important;
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0.5rem 0;
        padding: 0 1rem;
        text-shadow: 0 1px 1px rgba(0,0,0,0.05);
    }
    .stFileUploader label {
        color: #555555;
        font-weight: 400;
    }
    .stButton > button {
        background: linear-gradient(45deg, #F35E5C, #F4EFF7);
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
    .stButton > button[kind="secondary"] {
        background: rgba(243, 94, 92, 0.1);
        color: #666666;
        border: 1px solid rgba(243, 94, 92, 0.3);
    }
    .stChatMessage {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        backdrop-filter: blur(5px);
    }
    div[role="img"][aria-label="user"] + div > div {
        background: linear-gradient(45deg, #F4EFF7, #FEFEFE);
        color: #333333;
        border-radius: 18px;
        border: 1px solid rgba(244, 239, 247, 0.5);
    }
    /* Note: Your assistant gradient values look invalid but leaving as-is. */
    div[role="img"][aria-label="assistant"] + div > div {
        background: linear-gradient(45deg, #D6CD1, #B23A4);
        color: #444444;
        border-radius: 18px;
        border: 1px solid rgba(214, 205, 1, 0.3);
    }
    div[data-testid="stAlert"] {
        background: rgba(243, 94, 92, 0.1);
        border: 1px solid rgba(243, 94, 92, 0.3);
        border-radius: 12px;
        color: #666666;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(214, 205, 1, 0.2);
        border-radius: 25px;
        color: #333333;
        padding: 0.75rem;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(102, 102, 102, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 RAG Chatbot with Groq")
st.header("Upload your documents and chat with them using ultra-fast LPU inference.")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        help="Drag and drop files here. Limit 200MB per file.",
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
                    suffix = os.path.splitext(uploaded_file.name)[1].lower()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        # Use extension routing (more reliable than MIME types)
                        if suffix == ".pdf":
                            loader = PyPDFLoader(tmp_path)
                            docs.extend(loader.load())
                        elif suffix == ".docx":
                            docs.extend(load_docx_rich(tmp_path, source_name=uploaded_file.name))
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
                    embedding=embeddings,
                )
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

                status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                st.success(f"Indexed {len(docs)} documents ({len(splits)} chunks).")
        else:
            st.warning("Please upload files first.")

    st.divider()
    st.info("Powered by Groq LPU Inference | Built with LangChain & Streamlit")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
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
                            api_key=groq_api_key,
                        )

                        system_prompt = (
                            "You are a helpful AI assistant. "
                            "Use the provided context to answer the user's question accurately. "
                            "If the answer isn't in the context, say you don't know based on the documents. "
                            "Keep your response professional and helpful.\n\n"
                            "Context:\n{context}"
                        )

                        prompt_template = ChatPromptTemplate.from_messages(
                            [("system", system_prompt), ("human", "{input}")]
                        )

                        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

                        result = rag_chain.invoke({"input": prompt})
                        response = result["answer"]
                except Exception as e:
                    response = f"❌ **Error**: {str(e)}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
