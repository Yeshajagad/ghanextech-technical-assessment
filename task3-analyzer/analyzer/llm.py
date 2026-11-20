"""Ask the chat model for a JSON answer.

Local: Ollama (qwen3 by default). Deployed demo: Groq (LLM_PROVIDER=groq).
"""

import httpx
import ollama

from analyzer import config


class LLMError(Exception):
    """Raised with a message that is safe to show to the user."""


def complete(system, user, schema, client=None):
    """Send a system and a user message; return the model's text, which should be JSON."""
    if config.LLM_PROVIDER == "ollama":
        return _complete_ollama(system, user, schema, client)
    if config.LLM_PROVIDER == "groq":
        return _complete_groq(system, user, client)
    raise LLMError(f"LLM_PROVIDER must be 'ollama' or 'groq' (got '{config.LLM_PROVIDER}').")


def _complete_ollama(system, user, schema, client):
    client = client or ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT)
    extra = {}
    if "qwen3" in config.OLLAMA_CHAT_MODEL:
        extra["think"] = False  # qwen3 would otherwise print its reasoning first

    try:
        response = client.chat(
            model=config.OLLAMA_CHAT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=schema,  # Ollama forces the reply to match this JSON schema
            options={"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
            **extra,
        )
    except ConnectionError:
        raise LLMError(
            f"Cannot reach Ollama at {config.OLLAMA_HOST}. "
            "Start it with `ollama serve` (or open the Ollama app)."
        )
    except httpx.TimeoutException:
        raise LLMError("The model took too long to answer. Try again, or use a smaller model.")
    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise LLMError(
                f"The model '{config.OLLAMA_CHAT_MODEL}' is not installed. "
                f"Run: ollama pull {config.OLLAMA_CHAT_MODEL}"
            )
        raise LLMError(f"Ollama returned an error: {error.error}")
    return response.message.content or ""


def _complete_groq(system, user, client):
    import groq  # only the deployed demo needs this package

    if client is None:
        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set. Add it to the app's secrets.")
        client = groq.Groq(api_key=config.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except groq.AuthenticationError:
        raise LLMError("The Groq API key was rejected. Check that it is correct.")
    except groq.RateLimitError:
        raise LLMError("Groq free-tier rate limit reached. Wait a minute and try again.")
    except groq.APIConnectionError:
        raise LLMError("Could not reach the Groq API. Check your internet connection.")
    except groq.APIStatusError as error:
        if error.status_code == 400:
            raise LLMError("The model could not produce a valid answer. Please try again.")
        raise LLMError(f"The Groq API returned an error ({error.status_code}).")
    return response.choices[0].message.content or ""