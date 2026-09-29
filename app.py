import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

import re


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PatentLens", page_icon="🔬", layout="wide")


# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Header banner ---------- */
    .plens-header {
        background: linear-gradient(120deg, #1A1A1D 0%, #2B2411 100%);
        border: 1px solid #C9A54A;
        padding: 2rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(201, 165, 74, 0.15);
    }
    .plens-header h1 {
        color: #E8C97A;
        font-size: 2.1rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .plens-header p {
        color: rgba(242,239,233,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }

    /* ---------- Stat cards ---------- */
    .plens-stat-card {
        flex: 1;
        background: #17181A;
        border: 1px solid rgba(201, 165, 74, 0.35);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }
    .plens-stat-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #E8C97A;
    }
    .plens-stat-label {
        font-size: 0.78rem;
        color: #B7AF9C;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ---------- Source chips ---------- */
    .plens-source-chip {
        display: inline-block;
        background: rgba(232, 201, 122, 0.12);
        color: #E8C97A;
        border: 1px solid rgba(232, 201, 122, 0.45);
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* ---------- Chat message cards ---------- */
    [data-testid="stChatMessage"] {
        background: #17181A;
        border: 1px solid rgba(232, 201, 122, 0.2);
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="plens-header">
        <h1>🔬 PatentLens</h1>
        <p>AI-powered patent intelligence — ask questions, get grounded answers with citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Prompt template (same as notebook)
# ---------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_template("""
You are PatentLens, an AI-powered patent intelligence assistant.

Answer the user's question using ONLY the provided patent context.

IMPORTANT RULES:
- Do not use outside knowledge.
- Do not invent or assume patent information.
- If the answer is not available in the context, say:
  "I could not find sufficient information in the provided patent documents."
- Keep the answer clear, concise, and professional.

PATENT CONTEXT:
{context}

USER QUESTION:
{question}

Give the answer in the following format:

ANSWER:
Give a direct and concise answer to the user's question.

KEY DETAILS:
- Mention important supporting details from the patent context.
- Include technical details only when they are relevant to the question.

SOURCE:
Use the patent ID and page information provided in the context.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_documents(documents) -> str:
    formatted = []
    for doc in documents:
        patent_id = doc.metadata.get("patent_id", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        formatted.append(f"[Patent: {patent_id}, Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(uploaded_files, embedding_model):
    """Load PDFs, chunk them, and index into an in-memory Qdrant collection."""
    all_documents = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # patent_id derived from the original filename (without extension)
            patent_id = os.path.splitext(uploaded_file.name)[0]
            for doc in documents:
                doc.metadata["patent_id"] = patent_id

            all_documents.extend(documents)

    for doc in all_documents:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)

    qdrant_client = QdrantClient(":memory:")
    collection_name = "patentlens"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    vector_store.add_documents(chunks)

    return vector_store, len(all_documents), len(chunks)


def build_rag_chain(vector_store, groq_api_key):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    rag_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def get_sources(retriever, question, k=5):
    documents = retriever.invoke(question)
    sources = []
    seen = set()
    for doc in documents[:k]:
        patent_id = doc.metadata.get("patent_id", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        key = (patent_id, page)
        if key not in seen:
            seen.add(key)
            sources.append({"patent_id": patent_id, "page": page})
    return sources


def render_sources(sources):
    chips = "".join(
        f'<span class="plens-source-chip">📄 {s["patent_id"]} · p.{s["page"]}</span>'
        for s in sources
    )
    st.markdown(chips, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: API key + PDF upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com",
    )

    uploaded_files = st.file_uploader(
        "Upload patent PDFs", type=["pdf"], accept_multiple_files=True
    )

    process_clicked = st.button(
        "🚀 Process documents", type="primary", disabled=not uploaded_files, use_container_width=True
    )

    if process_clicked:
        if not groq_api_key:
            st.error("Please enter your Groq API key first.")
        else:
            with st.spinner("Loading, chunking, and indexing PDFs..."):
                embedding_model = get_embedding_model()
                vector_store, n_pages, n_chunks = build_vector_store(uploaded_files, embedding_model)
                rag_chain, retriever = build_rag_chain(vector_store, groq_api_key)

                st.session_state["rag_chain"] = rag_chain
                st.session_state["retriever"] = retriever
                st.session_state["messages"] = []
                st.session_state["n_pages"] = n_pages
                st.session_state["n_chunks"] = n_chunks
                st.session_state["n_files"] = len(uploaded_files)

            st.success(f"Indexed {n_pages} page(s) into {n_chunks} chunk(s).")

    if "rag_chain" in st.session_state:
        st.markdown("---")
        st.markdown("✅ **Ready.** Ask a question in the chat.")


# ---------------------------------------------------------------------------
# Stats bar
# ---------------------------------------------------------------------------
if "rag_chain" in st.session_state:
    c1, c2, c3 = st.columns(3)
    for col, value, label in [
        (c1, st.session_state.get("n_files", 0), "Files indexed"),
        (c2, st.session_state.get("n_pages", 0), "Pages"),
        (c3, st.session_state.get("n_chunks", 0), "Chunks"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="plens-stat-card">
                    <div class="plens-stat-value">{value}</div>
                    <div class="plens-stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    avatar = "🧑‍💻" if message["role"] == "user" else "🔬"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask PatentLens about your uploaded patents...")

if question:
    if "rag_chain" not in st.session_state:
        st.warning("Upload PDFs and click 'Process documents' in the sidebar first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("Analyzing patent context..."):
                answer = st.session_state["rag_chain"].invoke(question)
                sources = get_sources(st.session_state["retriever"], question)

            st.markdown(answer)
            render_sources(sources)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )