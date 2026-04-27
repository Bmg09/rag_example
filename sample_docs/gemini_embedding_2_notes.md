# Gemini Embedding 2 Notes

Gemini Embedding 2 is a multimodal embedding model. It maps text, images, video, audio, and documents into one embedding space. For this demo, text documents, PNG/JPEG images, and MP4/MOV videos can be embedded and searched.

For retrieval, documents are formatted as:

`title: {title} | text: {content}`

Questions are formatted as:

`task: question answering | query: {question}`

This keeps the query-document relationship consistent for asymmetric retrieval. The app uses 768 dimensions by default because it is smaller and faster for a local demo. Larger dimensions such as 1536 or 3072 can improve quality at higher storage and latency cost.

The answer model is separate from the embedding model. Embeddings find relevant context. The generation model reads that context and writes the final response.
