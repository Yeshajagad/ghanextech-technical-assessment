# Task 3 – GitHub Repository Analyzer

A Streamlit dashboard. Paste a public GitHub repository and get its key facts, a checklist of good project practices, and an AI-written analysis with a summary, strengths, at least three improvement recommendations and ways to contribute.

**Live demo:** [LIVE_URL]

The analysis runs fully on your own machine with [Ollama](https://ollama.com) (`qwen3`). The public demo cannot reach a local Ollama, so it uses Groq's free tier instead. The switch is one environment variable, no code change.

## What you get

| Section | Where it comes from |
|---|---|
| Stars, forks, open issues, main language, license, topics | GitHub REST API |
| Language shares | GitHub `languages` endpoint, turned into percentages in code |
| Project checklist (tests, CI, license, contributing guide, docs folder, Dockerfile) | Worked out in code from the file list, not guessed by the model |
| Summary, tech stack, strengths, architecture | The model |
| Recommendations (always at least three) | The model, tied to what it saw (for example "no tests folder") |
| Contribution ideas | The model |
| Open `good first issue` issues | GitHub API (real links, never written by the model) |
| File structure | GitHub `git/trees` endpoint |

## How it works

```
 "owner/repo" or a github.com link
    │
    ▼
 parse_repo (analyzer/github.py)         ── invalid input → friendly error
    │
    ▼
 fetch_repo (analyzer/github.py)         ── not found / private / rate limit / timeout → friendly error
    │  1 repository details
    │  2 languages
    │  3 README
    │  4 file tree
    │  5 package files (package.json, pyproject.toml, requirements.txt ...)
    │  6 open "good first issue" issues
    ▼
 build_context (analyzer/analysis.py)    ── README cut to 6,000 characters, tree summarised, checks added
    │
    ▼
 complete (analyzer/llm.py)              ── Ollama qwen3, or Groq when deployed; answer forced to JSON
    │
    ▼
 parse_analysis (analyzer/analysis.py)   ── checks the JSON, needs a summary and 3+ recommendations;
    │                                       one retry if the answer is unusable
    ▼
 dashboard (app.py)
```

**What the model sees, and what it does not.** It gets the repository details, a summary of the file layout, the checklist, up to three package files and the README. It does **not** read the source code, and the dashboard says so. The prompt tells it to say "cannot tell" instead of guessing.

**Error handling.**

| Situation | What the user sees |
|---|---|
| Input is not `owner/repo` or a github.com link | Red notice explaining the accepted formats |
| Repository does not exist or is private | Red notice: "Repository not found. It may not exist, or it may be private" |
| GitHub rate limit reached | Yellow notice with the reset time and a hint to set `GITHUB_TOKEN` |
| GitHub slow or unreachable | Red notice |
| Model off, missing, slow or rate limited | The GitHub facts are still shown, with a red notice where the analysis would be |
| Model returns unusable JSON twice | Red notice asking the user to try again |

**Rate limits.** Without a token GitHub allows 60 requests per hour per IP address, and one analysis uses up to 8. Results are cached for 15 minutes (GitHub data) and 30 minutes (analysis), so repeating a repository costs nothing. A personal access token with no permissions raises the limit to 5,000 per hour. It is optional and read from `GITHUB_TOKEN`.

## Example repositories

| Input | Result |
|---|---|
| `pallets/flask` | Full dashboard |
| `https://github.com/psf/requests` | Links are accepted too |
| `ollama/ollama-python` | Full dashboard |
| `octocat/this-repo-does-not-exist` | "Repository not found" notice |
| `hello` | "That does not look like a GitHub repository" notice |

## Choices and why

| Choice | Reason |
|---|---|
| Facts come from code, advice comes from the model | Numbers, links and the checklist are exact. The model is only asked to interpret them, so it cannot invent a star count or an issue link. |
| Model answers in JSON, checked before use | The dashboard needs separate fields. Ollama is given a JSON schema and Groq is put in JSON mode, and the answer is still validated, with one retry. |
| Only README, layout and package files are sent | Fits a small local model, keeps the Groq free tier usable, and avoids hundreds of API calls. The limit is stated in the UI. |
| `httpx` and plain functions, no GitHub library | Six endpoints are used, so a wrapper library would add more than it saves. |
| `qwen3` with `think` turned off and a larger context window | Reliable at following a JSON format. Ollama's default context window is small and would silently cut the prompt, so it is set to 8,192 tokens. |

## Run it locally

You need Python 3.10 or newer and Ollama.

```bash
ollama pull qwen3

git clone https://github.com/Yeshajagad/ghanextech-technical-assessment.git
cd ghanextech-technical-assessment/task3-analyzer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt

cp .env.example .env             # the defaults already point at Ollama
streamlit run app.py
```

Optionally put a `GITHUB_TOKEN` in `.env` to avoid rate limits.

Run the tests (they use fake GitHub responses and a fake model, so they need neither Ollama nor internet):

```bash
pip install pytest
python -m pytest -q
```

## Deployment

The live demo runs on Streamlit Community Cloud, from this repository with the main file set to `task3-analyzer/app.py`. It uses `requirements.txt` (which adds `groq`) and these secrets:

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "..."
GITHUB_TOKEN = "..."
```

`GITHUB_TOKEN` matters on shared hosting, where many apps share one IP address and the 60-per-hour limit is used up quickly. Keys are stored in the app's Secrets settings, never in the repository.

## Known limitations

- **The source code is not read.** Advice is based on the README, file layout and package files, so it can miss problems that only show in the code.
- **Only the top of the file tree is summarised.** Very large repositories (GitHub truncates their file list) get a partial picture.
- **The checklist is based on file names.** A repository that keeps its tests in an unusual folder may show a false "no".
- **Only the first three package files in the root folder are read.** Monorepos with files in subfolders are not covered.
- **Good first issues need the exact label `good first issue`.** Repositories that use another label show none.
- **Small local models vary.** The same repository can get slightly different wording between runs, and the demo (Groq) and local app use different models.
- **Free-tier limits.** GitHub and Groq limits apply and are reported. Streamlit Community Cloud puts apps to sleep after a period without visitors.

## Ideas for improvement

- Read a few key source files (entry point, a test file) for deeper architecture comments.
- Let the user paste their own GitHub token in the sidebar.
- Compare two repositories side by side.
- Export the analysis as Markdown.