from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader


EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-2.5-flash"
TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif", ".avif"}
VIDEO_EXTENSIONS = {".mp4", ".mpeg", ".mpg", ".mov"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | MEDIA_EXTENSIONS
MEDIA_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/quicktime",
}


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    kind: str = "text"
    mime_type: str = ""


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    chunk_index: int
    score: float
    kind: str = "text"
    mime_type: str = ""


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.name}")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[page {page_number}]\n{page_text}")
        return clean_text("\n\n".join(pages))

    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    words = text.split()
    if not words:
        return []

    overlap = min(overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - overlap, 1)
    chunks: list[TextChunk] = []

    for chunk_index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue

        chunk = " ".join(chunk_words)
        digest = hashlib.sha256(f"{source}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()
        chunks.append(
            TextChunk(
                id=digest[:24],
                text=chunk,
                source=source,
                chunk_index=chunk_index,
            )
        )

        if start + chunk_size >= len(words):
            break

    return chunks


def collect_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def media_chunk(path: Path) -> TextChunk:
    media_bytes = path.read_bytes()
    suffix = path.suffix.lower()
    digest = hashlib.sha256(str(path).encode("utf-8") + b":" + media_bytes).hexdigest()
    kind = "video" if suffix in VIDEO_EXTENSIONS else "image"
    return TextChunk(
        id=digest[:24],
        text=f"{kind.title()} file: {path.name}",
        source=str(path),
        chunk_index=0,
        kind=kind,
        mime_type=MEDIA_MIME_TYPES[suffix],
    )


def prepare_document(content: str, title: str | None = None) -> str:
    title = title or "none"
    return f"title: {title} | text: {content}"


def prepare_query(query: str) -> str:
    return f"task: question answering | query: {query}"


def requested_media_kind(query: str) -> str | None:
    normalized = query.lower()
    asks_for_inventory = any(
        marker in normalized
        for marker in (
            "db",
            "database",
            "indexed",
            "uploaded",
            "have",
            "list",
            "show",
            "describe",
            "what is in",
            "what's in",
            "what image",
            "which image",
            "what video",
            "which video",
        )
    )
    if not asks_for_inventory:
        return None

    if any(term in normalized for term in ("image", "images", "photo", "picture")):
        return "image"
    if "video" in normalized:
        return "video"
    if "media" in normalized:
        return "media"
    return None


class GeminiRAG:
    def __init__(
        self,
        api_key: str | None,
        db_path: str | Path = ".chroma",
        embedding_model: str = EMBEDDING_MODEL,
        generation_model: str = GENERATION_MODEL,
        dimensions: int = 768,
    ) -> None:
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.dimensions = dimensions
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.chroma = chromadb.PersistentClient(path=str(db_path))
        model_key = re.sub(r"[^a-zA-Z0-9_]+", "_", embedding_model).strip("_").lower()
        self.collection = self.chroma.get_or_create_collection(
            name=f"rag_docs_{model_key}_{dimensions}",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        collection_name = self.collection.name
        self.chroma.delete_collection(collection_name)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_text(self, text: str) -> list[float]:
        if self.client is None:
            raise ValueError("Set GEMINI_API_KEY before indexing or asking.")

        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        return list(result.embeddings[0].values)

    def embed_document(self, text: str, source: str) -> list[float]:
        return self.embed_text(prepare_document(text, title=Path(source).name))

    def embed_media(self, source: str, mime_type: str) -> list[float]:
        if self.client is None:
            raise ValueError("Set GEMINI_API_KEY before indexing or asking.")

        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=[
                types.Part.from_bytes(
                    data=Path(source).read_bytes(),
                    mime_type=mime_type,
                )
            ],
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        return list(result.embeddings[0].values)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(prepare_query(query))

    def index_paths(
        self,
        paths: Iterable[Path],
        chunk_size: int = 550,
        overlap: int = 90,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> int:
        files = collect_files(paths)
        chunks: list[TextChunk] = []

        for file_path in files:
            if file_path.suffix.lower() in MEDIA_EXTENSIONS:
                chunks.append(media_chunk(file_path))
                continue

            text = load_document(file_path)
            chunks.extend(
                chunk_text(
                    text=text,
                    source=str(file_path),
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

        for index, chunk in enumerate(chunks, start=1):
            if progress:
                progress(index, len(chunks), Path(chunk.source).name)

            if chunk.kind in {"image", "video"}:
                embedding = self.embed_media(chunk.source, chunk.mime_type)
            else:
                embedding = self.embed_document(chunk.text, chunk.source)

            self.collection.upsert(
                ids=[chunk.id],
                embeddings=[embedding],
                documents=[chunk.text],
                metadatas=[
                    {
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "kind": chunk.kind,
                        "mime_type": chunk.mime_type,
                    }
                ],
            )

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if self.count == 0:
            return []

        query_embedding = self.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[SearchHit] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            hits.append(
                SearchHit(
                    text=document,
                    source=str(metadata.get("source", "unknown")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    score=max(0.0, 1.0 - float(distance)),
                    kind=str(metadata.get("kind", "text")),
                    mime_type=str(metadata.get("mime_type", "")),
                )
            )

        return hits

    def list_media(self, kind: str | None = None, limit: int = 10) -> list[SearchHit]:
        if self.count == 0:
            return []

        result = self.collection.get(include=["documents", "metadatas"])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        hits: list[SearchHit] = []

        for document, metadata in zip(documents, metadatas):
            hit_kind = str(metadata.get("kind", "text"))
            if hit_kind not in {"image", "video"}:
                continue
            if kind in {"image", "video"} and hit_kind != kind:
                continue

            hits.append(
                SearchHit(
                    text=document,
                    source=str(metadata.get("source", "unknown")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    score=1.0,
                    kind=hit_kind,
                    mime_type=str(metadata.get("mime_type", "")),
                )
            )

            if len(hits) >= limit:
                break

        return hits

    def generate_from_hits(
        self,
        query: str,
        hits: list[SearchHit],
        media_inventory: bool = False,
    ) -> tuple[str, list[SearchHit]]:
        context_blocks: list[str] = []
        media_parts: list[types.Part] = []

        for index, hit in enumerate(hits, start=1):
            if hit.kind in {"image", "video"}:
                context_blocks.append(
                    f"[{index}] Source: {Path(hit.source).name} {hit.kind}\n"
                    f"A retrieved {hit.kind} is attached after the prompt. Use evidence from it."
                )
                if Path(hit.source).exists() and hit.mime_type:
                    media_parts.extend(
                        [
                            types.Part.from_text(
                                text=f"Retrieved {hit.kind} [{index}]: {Path(hit.source).name}"
                            ),
                            types.Part.from_bytes(
                                data=Path(hit.source).read_bytes(),
                                mime_type=hit.mime_type,
                            ),
                        ]
                    )
            else:
                context_blocks.append(
                    f"[{index}] Source: {Path(hit.source).name} chunk {hit.chunk_index}\n{hit.text}"
                )

        context = "\n\n".join(context_blocks)
        prompt = f"""You are a RAG assistant.
Answer the question using only the context.
If the context is insufficient, say what is missing.
Cite sources with bracket numbers like [1].
{"The user is asking about indexed media inventory. Describe each retrieved media item directly." if media_inventory else ""}

Question:
{query}

Context:
{context}
"""

        if self.client is None:
            raise ValueError("Set GEMINI_API_KEY before indexing or asking.")

        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=[types.Part.from_text(text=prompt), *media_parts],
        )
        return response.text or "", hits

    def answer(self, query: str, top_k: int = 5) -> tuple[str, list[SearchHit]]:
        media_kind = requested_media_kind(query)
        if media_kind:
            media_hits = self.list_media(kind=media_kind, limit=top_k)
            if media_hits:
                return self.generate_from_hits(query, media_hits, media_inventory=True)

        hits = self.search(query, top_k=top_k)
        if not hits:
            return "No indexed context found. Index documents first.", []

        return self.generate_from_hits(query, hits)

    def preview(self, limit: int = 8) -> list[SearchHit]:
        if self.count == 0:
            return []

        result = self.collection.peek(limit=limit)
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        return [
            SearchHit(
                text=document,
                source=str(metadata.get("source", "unknown")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                score=0.0,
                kind=str(metadata.get("kind", "text")),
                mime_type=str(metadata.get("mime_type", "")),
            )
            for document, metadata in zip(documents, metadatas)
        ]
