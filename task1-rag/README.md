# Task 1 – Document Assistant (RAG)

A Streamlit app where you upload PDFs and ask questions about them. Answers are generated only from the text of the uploaded files, and every answer shows the passages it was based on.

**Live demo:** [LIVE_URL]

The app runs fully on your own machine with [Ollama](https://ollama.com) (`bge-m3` for embeddings, `llama3.2` for answers). The public demo cannot reach a local Ollama, so it uses Groq's free tier for answers and a small embedding model instead. The switch is two environment variables, no code change.

## How it works

```
 PDF files
    │  pypdf: text per page                (rag/ingest.py)
    ▼
 page text
    │  recursive splitter, 800 / 150       (rag/chunker.py)
    ▼
 chunks  ── source file + page number kept with each chunk
    │  bge-m3 via Ollama                   (rag/embeddings.py)
    ▼
 FAISS index (in memory)                   (rag/store.py)
    ▲
    │  question → embedding → top-k cosine similarity
    │
 top-k chunks + question
    │  strict system prompt, streamed      (rag/llm.py)
    ▼
 llama3.2 via Ollama  →  answer with [1], [2] citations
```

1. **Ingestion.** `pypdf` reads each page. Pages with no text are skipped. Unsupported file types, corrupted PDFs, password-protected PDFs and scanned PDFs (no text layer) each show a clear message in the sidebar instead of crashing.
2. **Chunking.** Each page is split on paragraph breaks first, then line breaks, then sentence ends, then spaces. Small pieces are merged back up to the chunk size, and the end of each chunk is repeated at the start of the next one.
3. **Embedding and storage.** Chunks are embedded and stored in a FAISS flat index. Whenever the set of uploaded files changes, the whole index is rebuilt and the chat is cleared, so answers never mix old and new documents.
4. **Retrieval.** The question is embedded with the same model and the top-k most similar chunks are returned (default 4, adjustable from 1 to 8). Each result carries its file name, page, chunk number and similarity score.
5. **Generation.** The chunks are numbered and sent to the LLM with a system prompt that says to use only that context and to say so when the answer is not there. Temperature is 0 to keep answers stable, and the reply is streamed so you see it as it is written.

## Choices and why

| Choice | Value | Reason |
|---|---|---|
| Chunk size | 800 characters (about 130–160 words) | Big enough to hold a full idea, small enough that one chunk covers one topic. It also fits the limit of the small embedding model used by the deployed demo, so local and deployed behave the same. `bge-m3` itself accepts much longer text. |
| Overlap | 150 characters (about 19%) | A sentence that falls on a chunk boundary still appears whole in one of the two chunks. More overlap would just store the same text twice. |
| Chunks per page | Never cross a page | Keeps the page number in each citation correct. The cost is that an idea running over a page break is split. |
| Embedding model | `bge-m3` (Ollama), 1024 dimensions | Strong multilingual retrieval model, runs locally with no API key, handles long input. The deployed demo uses `all-MiniLM-L6-v2` (384 dimensions) through `fastembed` because it is small enough for free hosting. |
| Vector store | FAISS `IndexFlatIP` on normalised vectors | Inner product on unit vectors is cosine similarity. A flat index is exact, and for a few thousand chunks it is instant, so an approximate index would add complexity for nothing. |
| LLM | `llama3.2` (Ollama) | Small (2 GB), fast on a laptop, and follows "answer only from the context" well enough for this task. Set `OLLAMA_CHAT_MODEL=qwen3` for a larger model. The deployed demo uses Llama 3.3 70B on Groq's free tier. |
| Framework | Plain Python, no LangChain | The pipeline is short enough to read in one sitting, and every step is visible. |

## Run it locally

You need Python 3.10 or newer and Ollama.

```bash
ollama pull bge-m3
ollama pull llama3.2

git clone https://github.com/Yeshajagad/ghanextech-technical-assessment.git
cd ghanextech-technical-assessment/task1-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt

cp .env.example .env             # the defaults already point at Ollama
streamlit run app.py
```

If Ollama is not running, or a model is missing, the sidebar says so and shows the exact command to fix it.

Run the tests (they use fake clients, so they need neither Ollama nor an API key):

```bash
pip install pytest
python -m pytest -q
```

## Deployment

The live demo runs on Streamlit Community Cloud, from this repository with the main file set to `task1-rag/app.py`. It uses `requirements.txt` (which adds `fastembed` and `groq`) and these secrets:

```toml
LLM_PROVIDER = "groq"
EMBED_PROVIDER = "minilm"
GROQ_API_KEY = "..."
```

The key is stored in the app's Secrets settings, never in the repository.

## Known limitations

- **No OCR.** Scanned PDFs are rejected with a message.
- **Tables and multi-column layouts** can come out of `pypdf` in the wrong reading order, which hurts retrieval.
- **The index lives in memory.** It is lost when the app restarts, and uploads are re-embedded each time the file set changes.
- **Each question is answered on its own.** The chat history is shown on screen but not sent to the model, so follow-ups like "and the second one?" do not work.
- **Retrieval always returns k chunks**, even if none are relevant. The prompt tells the model to refuse in that case, but a similarity cut-off would be safer.
- **Small local models can ignore instructions.** `llama3.2` sometimes adds outside knowledge despite the prompt. Showing the sources lets you check every answer.
- **The demo and the local app use different models**, so answers and similarity scores differ between them.
- **Free-tier limits.** Groq rate limits apply and the app shows a message when they are hit. Streamlit Community Cloud puts apps to sleep after a period without visitors, and the first load after that is slow.

## Ideas for improvement

- Add a similarity threshold and an option to rerank the top results.
- Send recent chat turns to the model so follow-up questions work.
- Persist the index to disk, keyed by file hash, so unchanged files are not re-embedded.
- Add OCR for scanned PDFs.