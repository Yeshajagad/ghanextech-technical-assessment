"""Quick check that Ollama is running and has the chat model this app needs."""

import ollama

from agent import config


def _installed(names, wanted):
    """True if `wanted` (for example 'qwen3') matches an installed name ('qwen3:latest')."""
    if ":" in wanted:  # a tag was given, so it must match exactly
        return wanted in names
    return any(name.split(":")[0] == wanted for name in names)


def ollama_problems(client=None):
    """Return a list of problems (empty list means everything is fine)."""
    if config.LLM_PROVIDER != "ollama":
        return []  # nothing here depends on Ollama

    client = client or ollama.Client(host=config.OLLAMA_HOST, timeout=5)
    try:
        names = [m.model for m in client.list().models]
    except ConnectionError:
        return [f"Ollama is not running at {config.OLLAMA_HOST}. Start it with `ollama serve`."]

    if not _installed(names, config.OLLAMA_CHAT_MODEL):
        return [f"Model '{config.OLLAMA_CHAT_MODEL}' is missing. Run: ollama pull {config.OLLAMA_CHAT_MODEL}"]
    return []