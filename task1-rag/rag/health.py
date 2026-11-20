"""Quick check that Ollama is running and has the models this app needs."""

import ollama

from rag import config


def _installed(names, wanted):
    """True if `wanted` (for example 'bge-m3') matches an installed name ('bge-m3:latest')."""
    if ":" in wanted:  # a tag was given, so it must match exactly
        return wanted in names
    return any(name.split(":")[0] == wanted for name in names)


def ollama_problems(client=None):
    """Return a list of problems (empty list means everything is fine)."""
    needed = []
    if config.LLM_PROVIDER == "ollama":
        needed.append(config.OLLAMA_CHAT_MODEL)
    if config.EMBED_PROVIDER == "ollama":
        needed.append(config.OLLAMA_EMBED_MODEL)
    if not needed:
        return []  # nothing here depends on Ollama

    client = client or ollama.Client(host=config.OLLAMA_HOST, timeout=5)
    try:
        names = [m.model for m in client.list().models]
    except ConnectionError:
        return [f"Ollama is not running at {config.OLLAMA_HOST}. Start it with `ollama serve`."]

    return [
        f"Model '{model}' is missing. Run: ollama pull {model}"
        for model in needed
        if not _installed(names, model)
    ]