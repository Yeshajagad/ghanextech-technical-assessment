# Ghanex Tech Technical Assessment – Yesha Jagad

GenAI & Open-Source Developer Intern assessment. Three working applications, each with its own README, tests and live demo, in one repository.

| Task | What it is | Live demo | Code |
|---|---|---|---|
| 1 | **AI Document Assistant (RAG).** Upload PDFs or text files and ask questions. Answers are streamed with page citations. | [https://yesha-rag-assistant.streamlit.app/] | [`task1-rag`](task1-rag) |
| 2 | **AI Agent with tool calling.** Chat with an agent that chooses between a calculator, weather and Wikipedia, and shows which tool it used. | [https://yesha-ai-agent.streamlit.app/] | [`task2-agent`](task2-agent) |
| 3A | **GitHub Repository Analyzer.** Paste a public repo and get facts, a checklist and an AI analysis with recommendations. | [https://yesha-repo-analyzer.streamlit.app/] | [`task3-analyzer`](task3-analyzer) |
| 3B | **Open-source contribution.** | [https://github.com/ollama/ollama-python/issues/803] | See [SUBMISSION.md](SUBMISSION.md) |

Submission details, including how AI tools were used, are in [SUBMISSION.md](SUBMISSION.md).

## How the three apps are built

- **Python and Streamlit** for all three.
- **Ollama locally, Groq when deployed.** The apps run fully offline on local models (`bge-m3`, `llama3.2`, `qwen3:4b`). Free hosting cannot reach a local Ollama, so the live demos switch to Groq (and a small embedding model for Task 1) with one environment variable and no code change.
- **No secrets in the repository.** Each task has a `.env.example`, and the live demos read their keys from the hosting platform's secrets.
- **Tests that need no network or model.** Fake models and fake web responses are used, and GitHub Actions runs each task's tests on every push.
- **Friendly errors.** Every failure a user can cause (bad input, model off, rate limit, missing file) shows a clear message instead of a crash.

## Repository layout

```
.github/workflows/ci.yml   tests for all three tasks
task1-rag/                 Task 1: README, app.py, rag/, tests/
task2-agent/               Task 2: README, app.py, agent/, tests/
task3-analyzer/            Task 3: README, app.py, analyzer/, tests/
SUBMISSION.md              filled submission form
```

## Run any task locally

You need Python 3.10 or newer and [Ollama](https://ollama.com).

```bash
ollama pull bge-m3 && ollama pull llama3.2 && ollama pull qwen3

git clone https://github.com/Yeshajagad/ghanextech-technical-assessment.git
cd ghanextech-technical-assessment/task1-rag        # or task2-agent, or task3-analyzer

python -m venv .venv
source .venv/bin/activate                            # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
cp .env.example .env
streamlit run app.py
```

Each task's README lists the models it needs, an architecture diagram, design choices and known limitations.