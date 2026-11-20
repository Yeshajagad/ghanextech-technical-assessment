"""Ask the LLM to answer from the retrieved passages only, streaming the reply.

Local: Ollama running llama3.2 (or any model set in OLLAMA_CHAT_MODEL).
Deployed demo: Groq's free tier (LLM_PROVIDER=groq).
"""

import ollama

from rag import config

SYSTEM_PROMPT = (
    "You answer questions about the user's documents. "
    "Use ONLY the context passages provided. "
    "If the context does not contain the answer, reply exactly: "
    "\"I could not find this in the uploaded documents.\" "
    "Do not use outside knowledge and do not guess. "
    "When you use a passage, mention it like [1] or [2] using its number."
)


class LLMError(Exception):
    """Raised with a message that is safe to show to the user."""


def build_prompt(question, chunks):
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        header = f"[{number}] {chunk['source']}, page {chunk['page']}"
        blocks.append(f"{header}\n{chunk['text']}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer_stream(question, chunks, client=None):
    """Yield the answer in small pieces of text as the model writes it."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]
    if config.LLM_PROVIDER == "ollama":
        yield from _stream_ollama(messages, client)
    elif config.LLM_PROVIDER == "groq":
        yield from _stream_groq(messages, client)
    else:
        raise LLMError(f"LLM_PROVIDER must be 'ollama' or 'groq' (got '{config.LLM_PROVIDER}').")


def _stream_ollama(messages, client):
    client = client or ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT)
    extra = {}
    if "qwen3" in config.OLLAMA_CHAT_MODEL:
        extra["think"] = False  # qwen3 would otherwise print its reasoning first

    try:
        stream = client.chat(
            model=config.OLLAMA_CHAT_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0},
            **extra,
        )
        for part in stream:
            if part.message.content:
                yield part.message.content
    except ConnectionError:
        raise LLMError(
            f"Cannot reach Ollama at {config.OLLAMA_HOST}. "
            "Start it with `ollama serve` (or open the Ollama app)."
        )
    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise LLMError(
                f"The model '{config.OLLAMA_CHAT_MODEL}' is not installed. "
                f"Run: ollama pull {config.OLLAMA_CHAT_MODEL}"
            )
        raise LLMError(f"Ollama returned an error: {error.error}")


def _stream_groq(messages, client):
    import groq  # only the deployed demo needs this package

    if client is None:
        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set. Add it to the app's secrets.")
        client = groq.Groq(api_key=config.GROQ_API_KEY)

    try:
        stream = client.chat.completions.create(
            model=config.GROQ_MODEL, messages=messages, temperature=0, stream=True
        )
        for part in stream:
            text = part.choices[0].delta.content
            if text:
                yield text
    except groq.AuthenticationError:
        raise LLMError("The Groq API key was rejected. Check that it is correct.")
    except groq.RateLimitError:
        raise LLMError("Groq free-tier rate limit reached. Wait a minute and try again.")
    except groq.APIConnectionError:
        raise LLMError("Could not reach the Groq API. Check your internet connection.")
    except groq.APIStatusError as error:
        raise LLMError(f"The Groq API returned an error ({error.status_code}).")