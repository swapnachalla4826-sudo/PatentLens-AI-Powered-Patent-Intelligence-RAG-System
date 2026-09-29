from fastapi import FastAPI
from pydantic import BaseModel

# Import the working RAG components from your RAG pipeline.
#
# Your rag_pipeline.py should contain:
#   - rag_chain
#   - retriever
#
# Example:
# from rag_pipeline import rag_chain, retriever

from rag_pipeline import rag_chain, retriever


# ============================================================
# PatentLens - FastAPI Backend
# ============================================================

app = FastAPI(
    title="PatentLens API",
    description="AI-Powered Patent Intelligence & RAG System",
    version="1.0"
)


# ============================================================
# Request Model
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# Source Extraction
# ============================================================

def get_sources(question: str, k: int = 5):

    documents = retriever.invoke(question)

    sources = []
    seen = set()

    for doc in documents[:k]:

        patent_id = doc.metadata.get("patent_id", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        # Remove duplicate filename suffix such as " (1)"
        patent_id = str(patent_id).replace(" (1)", "").strip()

        # Avoid duplicate patent + page combinations
        key = (patent_id, page)

        if key not in seen:
            seen.add(key)

            sources.append({
                "patent_id": patent_id,
                "page": page
            })

    return sources


# ============================================================
# Home Endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "PatentLens API is running"
    }


# ============================================================
# Ask Endpoint
# ============================================================

@app.post("/ask")
def ask_patentlens(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        return {
            "question": question,
            "answer": "Please provide a question.",
            "sources": []
        }

    answer = rag_chain.invoke(question)

    sources = get_sources(question)

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
