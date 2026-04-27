from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_core import EMBEDDING_MODEL, GENERATION_MODEL, GeminiRAG, SUPPORTED_EXTENSIONS


ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / ".rag_uploads"


load_dotenv()


st.set_page_config(
    page_title="Gemini RAG Lab",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fb;
        color: #14171f;
    }
    [data-testid="stSidebar"] {
        background: #151923;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #f4f6fb;
    }
    button[data-baseweb="tab"] {
        color: #374151;
        font-weight: 650;
    }
    button[data-baseweb="tab"] p {
        color: #374151;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #b4232a;
    }
    button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
        color: inherit;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 6px 0 18px;
    }
    .metric-tile {
        background: #ffffff;
        border: 1px solid #d9deea;
        border-left: 4px solid #16877a;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .metric-label {
        color: #5f6878;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #111827;
        font-size: 1.35rem;
        font-weight: 720;
        line-height: 1.2;
    }
    .source-box {
        border-left: 4px solid #cf8f24;
        padding: 8px 12px;
        background: #fffaf0;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 650;
    }
    @media (max-width: 760px) {
        .metric-row {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_key_value() -> str | None:
    key = st.session_state.get("api_key", "").strip()
    return key or os.getenv("GEMINI_API_KEY") or None


def save_uploads(files) -> list[Path]:
    UPLOAD_DIR.mkdir(exist_ok=True)
    saved: list[Path] = []
    for file in files:
        suffix = Path(file.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        target = UPLOAD_DIR / file.name
        target.write_bytes(file.getbuffer())
        saved.append(target)
    return saved


def build_rag() -> GeminiRAG:
    return GeminiRAG(
        api_key=api_key_value(),
        embedding_model=st.session_state.embedding_model,
        generation_model=st.session_state.generation_model,
        dimensions=st.session_state.dimensions,
    )


with st.sidebar:
    st.header("Config")
    st.text_input(
        "Gemini API key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        key="api_key",
    )
    st.text_input("Embedding model", EMBEDDING_MODEL, key="embedding_model")
    st.text_input("Answer model", GENERATION_MODEL, key="generation_model")
    st.selectbox("Embedding dims", [768, 1536, 3072], index=0, key="dimensions")
    st.slider("Top K", min_value=2, max_value=10, value=5, key="top_k")
    st.slider("Chunk words", min_value=250, max_value=1200, value=550, step=50, key="chunk_size")
    st.slider("Overlap words", min_value=0, max_value=250, value=90, step=10, key="overlap")

rag = build_rag()
media_count = len(rag.list_media(limit=1000))

st.title("Gemini RAG Lab")
st.caption("Gemini Embedding 2 + Chroma + grounded answers")

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-label">Indexed items</div>
            <div class="metric-value">{rag.count}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Embedding model</div>
            <div class="metric-value">{st.session_state.embedding_model}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Media items</div>
            <div class="metric-value">{media_count}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Vector DB</div>
            <div class="metric-value">Chroma</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

index_tab, ask_tab, inspect_tab = st.tabs(["Index", "Ask", "Inspect"])

with index_tab:
    left, right = st.columns([1, 1])

    with left:
        uploaded_files = st.file_uploader(
            "Files",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )
        st.caption("Upload selects files only. Click Index files to embed and store them.")
        folder = st.text_input("Folder", value=str(ROOT / "sample_docs"))

    with right:
        st.write("Flow")
        st.code(
            "docs/images/video -> gemini-embedding-2 -> Chroma -> retrieve -> Gemini answer",
            language="text",
        )
        if st.button("Clear index", type="secondary", use_container_width=True):
            rag.reset()
            st.rerun()

    if st.button("Index files", type="primary", use_container_width=True):
        if not api_key_value():
            st.error("Set GEMINI_API_KEY first.")
        else:
            paths: list[Path] = []
            if uploaded_files:
                paths.extend(save_uploads(uploaded_files))
            if folder.strip():
                paths.append(Path(folder).expanduser())

            progress_bar = st.progress(0)
            status = st.empty()

            def update_progress(done: int, total: int, name: str) -> None:
                progress_bar.progress(done / max(total, 1))
                status.write(f"Embedding {done}/{total}: {name}")

            try:
                total = rag.index_paths(
                    paths=paths,
                    chunk_size=st.session_state.chunk_size,
                    overlap=st.session_state.overlap,
                    progress=update_progress,
                )
                st.success(f"Indexed {total} items.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

with ask_tab:
    query = st.text_area(
        "Question",
        value="Explain this RAG pipeline like I am in an interview.",
        height=110,
    )
    if st.button("Ask", type="primary", use_container_width=True):
        if not api_key_value():
            st.error("Set GEMINI_API_KEY first.")
        elif not query.strip():
            st.error("Ask a question.")
        else:
            with st.spinner("Retrieving and answering..."):
                try:
                    answer, hits = rag.answer(query.strip(), top_k=st.session_state.top_k)
                    st.subheader("Answer")
                    st.write(answer)
                    st.subheader("Sources")
                    for index, hit in enumerate(hits, start=1):
                        source_name = Path(hit.source).name
                        location = "media" if hit.kind in {"image", "video"} else f"chunk {hit.chunk_index}"
                        st.markdown(
                            f"""
                            <div class="source-box">
                                [{index}] {source_name} / {location}
                                / score {hit.score:.3f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if hit.kind == "image" and Path(hit.source).exists():
                            st.image(hit.source, caption=source_name, width=360)
                        elif hit.kind == "video" and Path(hit.source).exists():
                            st.video(hit.source)
                        else:
                            st.caption(hit.text[:900] + ("..." if len(hit.text) > 900 else ""))
                except Exception as exc:
                    st.error(str(exc))

with inspect_tab:
    st.write("Indexed items")
    st.caption("These are stored Chroma text chunks and media items. Open one to inspect retrieval context.")
    for hit in rag.preview(limit=10):
        location = "media" if hit.kind in {"image", "video"} else f"chunk {hit.chunk_index}"
        with st.expander(f"{Path(hit.source).name} / {location}"):
            if hit.kind == "image" and Path(hit.source).exists():
                st.image(hit.source, caption=Path(hit.source).name, width=420)
            elif hit.kind == "video" and Path(hit.source).exists():
                st.video(hit.source)
            else:
                st.write(hit.text)
