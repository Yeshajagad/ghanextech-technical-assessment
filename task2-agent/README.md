# Task 2 – Tool-Using AI Agent

A Streamlit chat app where an AI agent decides, for each message, whether to answer directly or to use a tool first. It remembers the conversation, shows which tools it used and what they returned, and keeps working when a tool fails.

**Live demo:** [LIVE_URL]

The agent runs fully on your own machine with [Ollama](https://ollama.com) (`qwen3:4b`). The public demo cannot reach a local Ollama, so it uses Groq's free tier instead. The switch is one environment variable, no code change.

## Tools

| Tool | What it does | Data source |
|---|---|---|
| `calculator` | Evaluates a math expression exactly | Local, no network |
| `get_weather` | Current temperature, feels-like, humidity, wind and sky for a city, in °C or °F | [Open-Meteo](https://open-meteo.com) (free, no key) |
| `search_wikipedia` | Introduction of the best matching Wikipedia article, with a link | Wikipedia API (free, no key) |

Each tool is defined with a name, a description the model reads to decide when to use it, and a JSON schema for its arguments:

```python
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the CURRENT weather for a city ...",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string", "description": "City name, optionally with a country or state ..."},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }
}
```

Example schemas for all three tools are in `agent/tools/`.

## How it works

```
 user message
    │
    ▼
 agent loop (agent/agent.py)  ◄────────────────────────────┐
    │  messages + tool schemas                              │
    ▼                                                       │
 model (agent/llm.py: Ollama qwen3, or Groq when deployed)  │
    │                                                       │
    ├── no tool call → final answer → shown in the chat     │
    │                                                       │
    └── tool call(s) → run_tool (agent/tools/__init__.py)   │
            │  check arguments against the schema           │
            │  run the tool, catch every error              │
            └── result (or "Error: ...") added to messages ─┘
```

1. The model gets the conversation and the three tool definitions.
2. If it replies with a tool call, the agent checks the arguments against the tool's schema, runs the tool, and sends the result back to the model. This repeats, so a question can use several tools in a row (for example weather, then the calculator on the temperature).
3. When the model replies without a tool call, that text is the answer. A greeting or general question goes straight here, with no tool.
4. The loop stops after 5 model calls, so a confused model cannot run forever.

**Conversation memory.** The text of earlier turns (last 20 messages) is sent with every new question, so follow-ups like "and in Fahrenheit?" work. Tool details are not kept in memory, only the final answers.

**Error handling.** `run_tool` never raises. Unknown tools, missing or wrongly typed arguments, network timeouts, rate limits, places that do not exist and crashes inside a tool all come back to the model as text starting with `Error:`. The system prompt tells the model to explain the problem to the user instead of inventing a result. Model-level problems (Ollama off, model not installed, rate limit, bad key) show a clear message in the chat instead of crashing.

**Safety of the calculator.** Expressions are parsed into a syntax tree and only numbers, arithmetic operators, parentheses, `pi`, `e` and a short list of math functions are accepted. Nothing is passed to `eval()`, expressions are limited to 200 characters, and huge powers are refused.

## Example prompts

| Prompt | What happens |
|---|---|
| `Hi! What can you do?` | Answered directly, no tool |
| `What is 18% of 2450?` | `calculator` |
| `What's the weather in Ahmedabad?` | `get_weather` |
| `Who was Ada Lovelace?` | `search_wikipedia` |
| `What's the temperature in Mumbai right now, multiplied by 3?` | `get_weather`, then `calculator` |
| `What's the weather in Atlantis?` | `get_weather` fails with "could not find a place"; the agent explains it |
| `What is 1 divided by 0?` | `calculator` fails; the agent explains it |

## Choices and why

| Choice | Reason |
|---|---|
| Native tool calling, not parsing text | The model returns structured calls that are validated against a schema. Text parsing breaks whenever the model words things differently. |
| `qwen3` locally (`think` turned off) | Reliable at tool calling for its size. Its reasoning mode is switched off so replies are fast and contain only the answer. `llama3.2` also works (`OLLAMA_CHAT_MODEL=llama3.2`). |
| One neutral message format in the agent | `llm.py` translates it for Ollama or Groq, so the agent loop does not know which provider is used. |
| Plain Python, no agent framework | The loop is about 30 lines. Every step is visible and easy to change. |
| Tools return text, errors included | The model can read an error and explain it, which is the behaviour the task asks for. |
| Free APIs without keys for the tools | Nothing to configure and nothing secret to leak. |

## Run it locally

You need Python 3.10 or newer and Ollama.

```bash
ollama pull qwen3

git clone https://github.com/Yeshajagad/ghanextech-technical-assessment.git
cd ghanextech-technical-assessment/task2-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt

cp .env.example .env             # the defaults already point at Ollama
streamlit run app.py
```

If Ollama is not running, or the model is missing, the sidebar says so and shows the exact command to fix it.

Run the tests (they use fake models and fake web responses, so they need neither Ollama nor internet):

```bash
pip install pytest
python -m pytest -q
```

## Deployment

The live demo runs on Streamlit Community Cloud, from this repository with the main file set to `task2-agent/app.py`. It uses `requirements.txt` (which adds `groq`) and these secrets:

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "..."
```

The key is stored in the app's Secrets settings, never in the repository.

## Known limitations

- **Weather is current conditions only**, not a forecast for future days.
- **Wikipedia returns the top search hit.** For an ambiguous name it may pick the wrong article.
- **Memory is text only and limited to 20 messages.** Tool results from earlier turns are not remembered, and a long chat loses its start.
- **Replies are not streamed.** The app shows "Thinking..." until the whole answer is ready, which can take a while on a CPU-only laptop.
- **Small local models make mistakes.** They sometimes skip a tool or pass odd arguments. The tool checks catch bad arguments, and the tool box under each answer lets you see what really happened.
- **The demo and the local app use different models**, so answers differ between them.
- **Free-tier limits.** Groq, Open-Meteo and Wikipedia rate limits apply and the agent reports them. Streamlit Community Cloud puts apps to sleep after a period without visitors, and the first load after that is slow.

## Ideas for improvement

- Stream the final answer as it is written.
- Add a forecast option to the weather tool and a currency tool.
- Remember tool results across turns.
- Let the user pick the model in the sidebar.