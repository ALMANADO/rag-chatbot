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

# Custom CSS (your original, unchanged)
st.markdown(
    """
<style>
/* -------------------------
   Theme tokens (easy tweaks)
--------------------------*/
:root{
  --bg0: #05060a;          /* deep background */
  --bg1: #0b0d14;          /* secondary background */
  --card: rgba(255,255,255,.06);
  --card2: rgba(255,255,255,.09);
  --stroke: rgba(255,255,255,.12);
  --stroke2: rgba(255,255,255,.08);
  --text: #e9eef7;
  --muted: rgba(233,238,247,.65);
  --muted2: rgba(233,238,247,.45);

  --accent: #8b5cf6;       /* purple */
  --accent2: #22d3ee;      /* cyan */
  --glow: rgba(139,92,246,.35);
  --glow2: rgba(34,211,238,.22);

  --radius-xl: 26px;
  --radius-lg: 18px;
  --radius-md: 14px;
}

/* -------------------------
   App background: starfield + subtle grid + vignette
--------------------------*/
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 50% 20%, rgba(139,92,246,.14), transparent 55%),
              radial-gradient(800px 500px at 80% 70%, rgba(34,211,238,.10), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
  position: relative;
  overflow: hidden;
}

/* Starfield + grid overlay */
[data-testid="stAppViewContainer"]::before{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;

  /* stars */
  background:
    radial-gradient(1px 1px at 12% 18%, rgba(255,255,255,.40) 50%, transparent 52%),
    radial-gradient(1px 1px at 22% 65%, rgba(255,255,255,.28) 50%, transparent 52%),
    radial-gradient(1px 1px at 35% 30%, rgba(255,255,255,.22) 50%, transparent 52%),
    radial-gradient(1px 1px at 44% 78%, rgba(255,255,255,.18) 50%, transparent 52%),
    radial-gradient(1px 1px at 58% 22%, rgba(255,255,255,.26) 50%, transparent 52%),
    radial-gradient(1px 1px at 70% 55%, rgba(255,255,255,.20) 50%, transparent 52%),
    radial-gradient(1px 1px at 82% 35%, rgba(255,255,255,.24) 50%, transparent 52%),
    radial-gradient(1px 1px at 90% 82%, rgba(255,255,255,.18) 50%, transparent 52%),

    /* subtle grid */
    linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);

  background-size:
    auto,auto,auto,auto,auto,auto,auto,auto,
    44px 44px, 44px 44px;

  opacity: .55;
  filter: blur(.2px);
}

/* Vignette */
[data-testid="stAppViewContainer"]::after{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background: radial-gradient(900px 520px at 50% 30%, transparent 40%, rgba(0,0,0,.55) 85%);
}

/* Bring actual app content above overlays */
[data-testid="stAppViewContainer"] > *{
  position: relative;
  z-index: 1;
}

/* Slightly narrow and center content like the screenshot */
section.main > div{
  max-width: 1050px;
  padding-top: 1.2rem;
  padding-bottom: 2.5rem;
}

/* -------------------------
   Sidebar: dark glass
--------------------------*/
section[data-testid="stSidebar"]{
  background: rgba(10,12,18,.65) !important;
  border-right: 1px solid rgba(255,255,255,.06);
  backdrop-filter: blur(14px);
}

section[data-testid="stSidebar"] *{
  color: var(--text);
}

/* -------------------------
   Typography (title like screenshot)
--------------------------*/
h1, h2, h3, p, label, span{
  color: var(--text);
}

/* Main title glow + gradient accent */
h1{
  text-align: center;
  font-weight: 650;
  letter-spacing: -.02em;
  margin-bottom: .2rem;
  text-shadow: 0 0 24px rgba(255,255,255,.08), 0 0 45px rgba(139,92,246,.12);
}

/* Streamlit header/subheader */
h2{
  text-align: center;
  font-weight: 500;
  color: var(--muted);
}

/* Captions / helper text */
.stCaption, [data-testid="stCaptionContainer"]{
  text-align: center !important;
  color: var(--muted) !important;
}

/* -------------------------
   Cards: chat messages + containers
--------------------------*/
div[data-testid="stChatMessage"]{
  background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.04));
  border: 1px solid rgba(255,255,255,.08);
  border-radius: var(--radius-xl);
  box-shadow:
    0 10px 30px rgba(0,0,0,.35),
    0 0 0 1px rgba(255,255,255,.03) inset;
  backdrop-filter: blur(12px);
  padding: 1rem 1rem;
  margin: .65rem 0;
}

/* Differentiate user vs assistant bubbles (based on avatar aria-label like your old CSS) */
div[role="img"][aria-label="user"] + div > div{
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: 18px !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,.03) inset;
}

div[role="img"][aria-label="assistant"] + div > div{
  background: linear-gradient(180deg, rgba(139,92,246,.14), rgba(255,255,255,.05)) !important;
  border: 1px solid rgba(139,92,246,.20) !important;
  border-radius: 18px !important;
  box-shadow:
    0 0 22px rgba(139,92,246,.12),
    0 0 0 1px rgba(255,255,255,.03) inset;
}

/* Markdown text inside bubbles */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li{
  color: var(--text);
}

/* -------------------------
   Buttons: pill + subtle glow like screenshot
--------------------------*/
.stButton > button{
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  background: rgba(255,255,255,.06) !important;
  color: var(--text) !important;
  box-shadow: 0 10px 26px rgba(0,0,0,.35);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}

.stButton > button:hover{
  transform: translateY(-1px);
  border-color: rgba(139,92,246,.35) !important;
  box-shadow: 0 14px 32px rgba(0,0,0,.45), 0 0 20px rgba(139,92,246,.12);
}

/* Primary buttons: gentle purple/cyan edge */
.stButton > button[kind="primary"]{
  background: linear-gradient(90deg, rgba(139,92,246,.22), rgba(34,211,238,.16)) !important;
  border: 1px solid rgba(139,92,246,.35) !important;
}

/* Secondary buttons: darker */
.stButton > button[kind="secondary"]{
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  color: var(--muted) !important;
}

/* -------------------------
   Inputs: file uploader + chat input to match dark glass
--------------------------*/
[data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,.05) !important;
  border: 1px dashed rgba(255,255,255,.16) !important;
  border-radius: var(--radius-xl);
  backdrop-filter: blur(10px);
}

[data-testid="stFileUploaderDropzone"] *{
  color: var(--muted) !important;
}

/* Chat input (the bottom textbox) */
[data-testid="stChatInput"] textarea{
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  border-radius: 999px !important;
  color: var(--text) !important;
  padding: 0.75rem 1rem !important;
  box-shadow: 0 10px 24px rgba(0,0,0,.35);
}

[data-testid="stChatInput"] textarea::placeholder{
  color: var(--muted2) !important;
}

/* Make spinners/status feel consistent */
div[data-testid="stStatusWidget"]{
  background: rgba(255,255,255,.05) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: var(--radius-xl);
  backdrop-filter: blur(12px);
}

/* Alerts (warning/info) as dark glass */
div[data-testid="stAlert"]{
  background: rgba(255,255,255,.05) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: var(--radius-lg);
  color: var(--text) !important;
  box-shadow: 0 10px 26px rgba(0,0,0,.35);
}

/* Links */
a{
  color: rgba(34,211,238,.95) !important;
}

/* Remove Streamlit default top padding gap on some layouts */
header[data-testid="stHeader"]{
  background: transparent;
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
