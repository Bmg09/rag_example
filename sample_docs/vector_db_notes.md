# Vector Database Notes

This project uses Chroma as the local vector database. Chroma stores embedding vectors, original text chunks, and metadata such as source file and chunk number. The data persists in the `.chroma` directory.

Similarity search compares the embedded question with stored chunk embeddings. The closest chunks are passed into the prompt. This app uses cosine distance because embedding direction usually matters more than vector magnitude for semantic search.

In production, a vector database may also support metadata filters, hybrid keyword plus vector search, reranking, access control, backups, and monitoring. Local Chroma is enough for a clear interview demo because the full RAG loop is visible and easy to explain.
