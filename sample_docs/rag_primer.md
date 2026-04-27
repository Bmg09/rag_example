# RAG Primer

Retrieval-Augmented Generation (RAG) connects a language model to external knowledge. The core loop is:

1. Load documents.
2. Split documents into chunks.
3. Convert chunks into embedding vectors.
4. Store vectors and metadata in a vector database.
5. Embed the user question.
6. Retrieve similar chunks by vector search.
7. Put retrieved context into the generation prompt.
8. Ask the language model to answer with citations.

RAG is useful when a model needs fresh, private, or domain-specific information. It reduces hallucination because the model has explicit context. It also makes answers auditable because retrieved chunks can be shown as sources.

The hard parts are chunking, retrieval quality, evaluation, and prompt discipline. Good RAG systems track citations, tune chunk size and overlap, add metadata filters, evaluate retrieval recall, and say when context is insufficient.
