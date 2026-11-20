import streamlit as st

from rag import config
from rag.chunker import chunk_pages
from rag.embeddings import Embedder, EmbeddingError
from rag.health import ollama_problems
from rag.ingest import IngestError, read_pdf
from rag.llm import LLMError, answer_stream
from rag.store import VectorStore

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def load_embedder():
    return Embedder()


@st.cache_data(ttl=10)
def check_ollama():
    return ollama_problems()


def init_state():
    if "store" not in st.session_state:
        st.session_state.store = VectorStore(load_embedder())
        st.session_state.signature = ()
        st.session_state.messages = []
        st.session_state.errors = []
        st.session_state.indexed_files = []


def rebuild_index(files):
    """Read every uploaded PDF and rebuild the vector index from scratch.

    Returns False if the embedding step failed (for example Ollama is not running).
    """
    all_chunks = []
    errors = []
    indexed = []

    for f in files:
        try:
            pages = read_pdf(f.name, f.getvalue())
        except IngestError as error:
            errors.append(str(error))
            continue
        all_chunks.extend(chunk_pages(pages, f.name))
        indexed.append(f.name)

    try:
        st.session_state.store.build(all_chunks)
    except EmbeddingError as error:
        st.session_state.store.build([])
        st.session_state.errors = [str(error)]
        st.session_state.indexed_files = []
        return False

    st.session_state.errors = errors
    st.session_state.indexed_files = indexed
    st.session_state.messages = []  # old answers belong to the old documents
    return True


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for number, s in enumerate(sources, start=1):
            st.markdown(
                f"**[{number}] {s['source']}** · page {s['page']} · "
                f"chunk {s['chunk_id']} · similarity {s['score']:.2f}"
            )
            st.caption(s["text"])


try:
    init_state()
except EmbeddingError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Your documents")
    st.caption(
        f"Chat: {config.LLM_PROVIDER} · "
        f"{config.OLLAMA_CHAT_MODEL if config.LLM_PROVIDER == 'ollama' else config.GROQ_MODEL}\n\n"
        f"Embeddings: {st.session_state.store.embedder.provider} · "
        f"{st.session_state.store.embedder.model_name}"
    )
    for problem in check_ollama():
        st.warning(problem)

    files = st.file_uploader("Upload PDF files", type=None, accept_multiple_files=True)
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, 4)

    signature = tuple((f.name, f.size) for f in files)
    if signature != st.session_state.signature:
        with st.spinner("Reading and indexing..."):
            done = rebuild_index(files)
        if done:  # on failure, try again on the next interaction
            st.session_state.signature = signature

    for message in st.session_state.errors:
        st.error(message)
    if st.session_state.indexed_files:
        st.success(
            f"{len(st.session_state.indexed_files)} file(s) indexed, "
            f"{len(st.session_state.store.chunks)} chunks."
        )

st.title("📄 Document Assistant")
st.caption("Ask questions about your PDFs. Answers come only from the uploaded text.")

if not st.session_state.indexed_files:
    st.info("Upload one or more PDF files in the sidebar to get started.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask something about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                sources = st.session_state.store.search(question, k=top_k)
            reply = st.write_stream(answer_stream(question, sources))
        except (LLMError, EmbeddingError) as error:
            st.error(str(error))
            st.session_state.messages.pop()  # drop the unanswered question
            st.stop()
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )