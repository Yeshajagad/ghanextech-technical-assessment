import streamlit as st

from agent import config
from agent.agent import run_agent
from agent.health import ollama_problems
from agent.llm import LLMError
from agent.tools import TOOLS

st.set_page_config(page_title="Tool-Using Agent", page_icon="🤖")

EXAMPLES = [
    "Hi! What can you do?",
    "What is 18% of 2450?",
    "What's the weather in Ahmedabad?",
    "Who was Ada Lovelace?",
    "What's the temperature in Mumbai right now, multiplied by 3?",
    "What's the weather in Atlantis?",
]


@st.cache_data(ttl=10)
def check_ollama():
    return ollama_problems()


def show_steps(steps):
    with st.expander(f"Tools used ({len(steps)})"):
        for step in steps:
            icon = "⚠️" if step.is_error else "🔧"
            arguments = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
            st.markdown(f"{icon} **{step.name}**({arguments})")
            st.code(step.result, language=None)


if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role", "content", "steps"}

model_name = config.OLLAMA_CHAT_MODEL if config.LLM_PROVIDER == "ollama" else config.GROQ_MODEL

with st.sidebar:
    st.header("Agent")
    st.caption(f"Model: {config.LLM_PROVIDER} · {model_name}")
    for problem in check_ollama():
        st.warning(problem)

    st.subheader("Tools")
    for tool in TOOLS.values():
        st.markdown(f"**{tool.name}**")
        st.caption(tool.description.split(". ")[0] + ".")

    st.subheader("Try an example")
    clicked = None
    for number, text in enumerate(EXAMPLES):
        if st.button(text, key=f"example_{number}", use_container_width=True):
            clicked = text

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Tool-Using Agent")
st.caption("Chat normally. The agent decides when to use a tool and shows you what it used.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("steps"):
            show_steps(message["steps"])

question = st.chat_input("Ask anything, for example a calculation, the weather or a fact") or clicked
if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Only the text of earlier turns is sent back to the model, not the tool details.
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = run_agent(history, question)
        except LLMError as error:
            st.error(str(error))
            st.stop()
        st.markdown(result.answer)
        if result.steps:
            show_steps(result.steps)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "steps": result.steps}
    )