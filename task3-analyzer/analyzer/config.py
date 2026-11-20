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
OLLAMA_TIMEOUT = 300  # seconds; writing a full analysis on a CPU can take a few minutes
OLLAMA_NUM_CTX = 8192  # Ollama's default window is small and would silently cut the prompt

# Groq (used only by the deployed demo)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# GitHub
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # optional
HTTP_TIMEOUT = 15  # seconds
USER_AGENT = "ghanex-assessment-analyzer/1.0 (student project)"

# How much of the repository is sent to the model
MAX_README_CHARS = 6000
MAX_PACKAGE_FILES = 3
MAX_PACKAGE_CHARS = 1500
MAX_TREE_LINES = 80