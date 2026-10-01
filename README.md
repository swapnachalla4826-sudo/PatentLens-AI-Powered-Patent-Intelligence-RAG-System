# 🔬 PatentLens — AI-Powered Patent Intelligence & RAG System

PatentLens is a Retrieval-Augmented Generation application that lets users upload patent PDFs and ask natural-language questions about their contents. Responses are grounded in retrieved patent context and accompanied by patent/page references.

## Overview

```text
Patent PDFs
   ↓
PDF Loading
   ↓
Text Cleaning
   ↓
Chunking
   ↓
Embeddings
   ↓
Qdrant
   ↓
Similarity Retrieval
   ↓
Groq LLM
   ↓
Grounded Answer + Sources
```

## Objectives

- Search multiple patent PDFs semantically
- Identify relevant patent information
- Answer questions from uploaded documents
- Provide patent and page references
- Reduce unsupported patent claims
- Provide an interactive Streamlit interface

## Document Processing

The application uses `PyPDFLoader`, cleans extracted text, and assigns a patent ID from the uploaded filename.

Current chunking configuration:

```text
chunk_size = 1000
chunk_overlap = 200
```

The implementation uses Hugging Face `all-MiniLM-L6-v2` embeddings and a 384-dimensional Qdrant collection with cosine distance. fileciteturn2file3L165-L208

## Retrieval

The retriever returns the top 5 relevant chunks:

```text
k = 5
```

Retrieved metadata includes patent ID and page number.

## Generation

The application uses:

```text
Groq
Model: openai/gpt-oss-20b
Temperature: 0
```

The prompt instructs the model to use only the supplied patent context and explicitly return a fallback when the required information is unavailable. fileciteturn2file3L116-L145

## RAG Chain

```text
Question
   ↓
Retriever
   ↓
Format Documents
   ↓
Prompt
   ↓
ChatGroq
   ↓
String Output
   ↓
Answer
```

The implementation uses LangChain Runnables to compose this chain. fileciteturn2file3L213-L228

## UI

The Streamlit interface supports:

- Multiple patent PDF upload
- Groq API key input
- Document processing
- Indexed-file/page/chunk metrics
- Chat-style questions
- Grounded answers
- Patent/page source chips

## Tech Stack

- Python
- Streamlit
- LangChain
- Qdrant
- Hugging Face Sentence Transformers
- PyPDF
- Groq
- RAG
- Vector Search
- LCEL

## Installation

```bash
pip install streamlit langchain langchain-community langchain-groq langchain-huggingface langchain-qdrant qdrant-client pypdf sentence-transformers
```

Set:

```text
GROQ_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

## Applications

- Patent document exploration
- Similar-invention discovery
- Technical patent Q&A
- Patent landscape exploration
- Prior-art research support
- Invention ideation support

## Future Improvements

- Query rewriting
- Retrieval grading
- Hybrid keyword + vector retrieval
- Reranking
- Persistent Qdrant storage
- Metadata filtering
- Langfuse tracing
- Retrieval/answer evaluation
- Multi-agent reasoning

## Disclaimer

PatentLens is an AI-assisted document analysis system and does not provide legal advice or determine patentability.

## Author

**Challa Swapna**
