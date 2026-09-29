import os
import glob
import re

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# 1. Load patent PDFs
# ---------------------------------------------------------

PDF_DIR = "Patent_Data"
pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in '{PDF_DIR}'. "
        "Make sure your patent PDFs are inside the Patent_Data folder."
    )

print("PDF files found:", len(pdf_files))

all_documents = []

for pdf_file in pdf_files:
    print(f"Loading: {pdf_file}")

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    all_documents.extend(documents)

print("Total pages loaded:", len(all_documents))


# ---------------------------------------------------------
# 2. Clean text
# ---------------------------------------------------------

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


for doc in all_documents:
    doc.page_content = clean_text(doc.page_content)


# ---------------------------------------------------------
# 3. Split documents into chunks
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_documents)

print("Original pages:", len(all_documents))
print("Total chunks:", len(chunks))


# ---------------------------------------------------------
# 4. Add patent IDs to metadata
# ---------------------------------------------------------

for chunk in chunks:
    source = chunk.metadata.get("source", "unknown")

    chunk.metadata["patent_id"] = os.path.splitext(
        os.path.basename(source)
    )[0]


# ---------------------------------------------------------
# 5. Create embedding model
# ---------------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

test_embedding = embedding_model.embed_query(
    "What is the purpose of this patent?"
)

print("Embedding dimensions:", len(test_embedding))


# ---------------------------------------------------------
# 6. Create Qdrant vector database
# ---------------------------------------------------------

qdrant_client = QdrantClient(":memory:")

collection_name = "patentlens"

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model
)

vector_store.add_documents(chunks)

print("Documents stored in Qdrant successfully!")


# ---------------------------------------------------------
# 7. Create retriever
# ---------------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)


# ---------------------------------------------------------
# 8. Load Groq API key
# ---------------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Set the GROQ_API_KEY environment variable before starting FastAPI."
    )

print("Groq API key loaded successfully!")


# ---------------------------------------------------------
# 9. Create Groq LLM
# ---------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)


# ---------------------------------------------------------
# 10. PatentLens prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
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


# ---------------------------------------------------------
# 11. Format retrieved documents
# ---------------------------------------------------------

def format_documents(documents):
    formatted = []

    for doc in documents:
        patent_id = doc.metadata.get("patent_id", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        formatted.append(
            f"[Patent: {patent_id}, Page: {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


# ---------------------------------------------------------
# 12. RAG chain
# ---------------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------
# 13. Get sources
# ---------------------------------------------------------

def get_sources(question, k=5):

    documents = retriever.invoke(question)

    sources = []
    seen = set()

    for doc in documents[:k]:

        patent_id = doc.metadata.get("patent_id", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        patent_id = str(patent_id).replace(" (1)", "").strip()

        key = (patent_id, page)

        if key not in seen:
            seen.add(key)

            sources.append({
                "patent_id": patent_id,
                "page": page
            })

    return sources


print("PatentLens RAG pipeline loaded successfully!")




































