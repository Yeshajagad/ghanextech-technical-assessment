"""The agent loop: ask the model, run any tools it asks for, repeat until it answers."""

import re
from dataclasses import dataclass, field

from agent import config, llm
from agent.tools import run_tool, tool_schemas

SYSTEM_PROMPT = (
    "You are a friendly assistant with three tools: calculator, get_weather and search_wikipedia.\n"
    "Rules:\n"
    "1. Answer greetings, opinions and general chat directly, without any tool.\n"
    "2. Use calculator for every calculation. Never do arithmetic in your head.\n"
    "3. Use get_weather for the current weather or temperature in a place.\n"
    "4. Use search_wikipedia for facts about people, places, events or concepts you are not sure about.\n"
    "5. If a question needs several steps, call the tools one after another. "
    "For example, get the weather first and then use calculator on the number.\n"
    "6. If a tool returns an error, tell the user plainly what went wrong and what they could try. "
    "Never invent a tool result.\n"
    "7. Keep answers short and clear."
)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Step:
    """One tool call made while answering, kept so the UI can show it."""

    name: str
    arguments: dict
    result: str
    is_error: bool


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)


def run_agent(history, question, chat=None):
    """Answer `question`, given earlier messages in `history` ({"role", "content"} dicts)."""
    chat = chat or llm.chat
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-config.MAX_HISTORY:]
    messages.append({"role": "user", "content": question})

    tools = tool_schemas()
    steps = []

    for _ in range(config.MAX_STEPS):
        reply = chat(messages, tools)

        if not reply.tool_calls:
            answer = THINK_BLOCK.sub("", reply.content).strip()
            return AgentResult(answer or "Sorry, I could not come up with an answer.", steps)

        messages.append(
            {"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls}
        )
        for call in reply.tool_calls:
            result, is_error = run_tool(call.name, call.arguments)
            steps.append(Step(call.name, call.arguments, result, is_error))
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result}
            )

    return AgentResult("I tried several steps but could not finish. Please try a simpler question.", steps)