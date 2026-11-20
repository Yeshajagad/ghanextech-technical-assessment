"""All settings live here. Values come from environment variables or a .env file.

Local development uses Ollama for both the chat model and the embeddings.
The deployed demo cannot reach your Ollama, so it switches to cloud options
just by changing LLM_PROVIDER and EMBED_PROVIDER (see the README).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Which backend answers questions and which one makes embeddings.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "groq"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")  # "ollama" or "minilm"

# Ollama (local)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
OLLAMA_TIMEOUT = 180  # seconds; a 3B model on CPU can be slow

# Cloud options used only by the deployed demo
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"