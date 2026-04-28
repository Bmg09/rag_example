# Gemini RAG Lab

Small interview-ready multimodal RAG app using `gemini-embedding-2`, Chroma, and Streamlit.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GEMINI_API_KEY
streamlit run app.py
```

## Deploy on Render

Use a Python web service.

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
- Optional env var: `GEMINI_API_KEY`

The app also accepts the Gemini API key in the sidebar, so the env var is convenient but not required.

## What It Shows

- Text documents become overlapping chunks.
- Images and videos become one media item each.
- Text chunks and media become vectors via `gemini-embedding-2`.
- Chroma stores vectors, text/media placeholders, and source metadata.
- User question becomes a query vector.
- Top matching text chunks or media items become prompt context.
- Gemini answer model responds with citations and can inspect retrieved image/video parts.

## Interview Talk Track

RAG is a grounding pattern. The LLM does not memorize my documents; retrieval provides relevant context at query time. Embeddings convert text, images, and videos into vectors, the vector database finds semantically similar items, and the generation prompt tells the model to answer only from retrieved evidence.

Key tradeoffs:

- Chunk size controls context completeness versus retrieval precision.
- Overlap prevents boundary loss between chunks.
- Embedding dimension affects quality, latency, and storage.
- Top K controls recall versus prompt noise.
- Citation display makes answers inspectable.
- Multimodal embeddings let a text query retrieve images or videos in the same vector space.

## Files

- `app.py`: Streamlit UI.
- `rag_core.py`: loading, chunking, embedding, Chroma search, generation.
- `sample_docs/`: demo knowledge base.
- `.chroma/`: generated local vector database.
