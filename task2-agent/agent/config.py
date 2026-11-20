"""All settings live here. Values come from environment variables or a .env file.

Local development uses Ollama (qwen3 by default). The deployed demo cannot reach
your Ollama, so it switches to Groq by setting LLM_PROVIDER=groq (see the README).
"""

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "groq"

# Ollama (local)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3")
OLLAMA_TIMEOUT = 180  # seconds; a local model on CPU can be slow

# Groq (used only by the deployed demo)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Agent behaviour
MAX_STEPS = 5  # most model calls for one question, so a confused model cannot loop forever
MAX_HISTORY = 20  # most earlier messages sent back to the model

# Tools that call web APIs
HTTP_TIMEOUT = 10  # seconds
USER_AGENT = "ghanex-assessment-agent/1.0 (student project)"
