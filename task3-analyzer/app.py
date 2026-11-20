import streamlit as st

from analyzer import analysis, config, github
from analyzer.analysis import AnalysisError
from analyzer.llm import LLMError

st.set_page_config(page_title="GitHub Repo Analyzer", page_icon="🔍", layout="wide")

EXAMPLES = ["pallets/flask", "psf/requests", "ollama/ollama-python", "octocat/this-repo-does-not-exist"]


@st.cache_data(ttl=900, show_spinner=False)
def load_repo(owner, name):
    return github.fetch_repo(owner, name)


@st.cache_data(ttl=1800, show_spinner=False)
def run_analysis(owner, name):
    return analysis.analyze(load_repo(owner, name))


def show_overview(repo):
    st.subheader(f"[{repo.full_name}]({repo.url})")
    if repo.description:
        st.write(repo.description)
    if repo.archived:
        st.warning("This repository is archived, so it no longer accepts changes.")

    columns = st.columns(5)
    columns[0].metric("Stars", f"{repo.stars:,}")
    columns[1].metric("Forks", f"{repo.forks:,}")
    columns[2].metric("Open issues", f"{repo.open_issues:,}")
    columns[3].metric("Main language", repo.language)
    columns[4].metric("License", repo.license)
    if repo.topics:
        st.write(" ".join(f"`{topic}`" for topic in repo.topics))

    left, right = st.columns(2)
    with left:
        st.markdown("**Languages**")
        shares = analysis.language_shares(repo.languages)
        for name, percent in shares.items():
            st.progress(percent / 100, text=f"{name} {percent}%")
        if not shares:
            st.caption("No code languages were detected.")
    with right:
        st.markdown("**Project checklist**")
        for label, found in analysis.detect_signals(repo.tree).items():
            st.markdown(f"{'✅' if found else '❌'} {label}")
        st.caption("Worked out from file names, so a missing item may just be named differently.")

    if repo.rate_remaining is not None:
        st.caption(f"GitHub API calls left this hour when this was loaded: {repo.rate_remaining}")


def show_analysis(result, repo):
    st.divider()
    st.header("AI analysis")
    st.caption("Based on the README, file layout and package files. The source code itself is not read.")

    st.markdown("### Summary")
    st.write(result.summary)
    if result.tech_stack:
        st.markdown("**Tech stack:** " + ", ".join(result.tech_stack))

    left, right = st.columns(2)
    with left:
        st.markdown("### Strengths")
        for point in result.strengths:
            st.markdown(f"- {point}")
    with right:
        st.markdown("### Architecture")
        st.write(result.architecture or "Not enough information to say.")

    st.markdown("### Recommendations")
    for number, item in enumerate(result.recommendations, start=1):
        st.markdown(f"**{number}. {item['title']}**")
        if item["why"]:
            st.write(item["why"])

    st.markdown("### Contribution opportunities")
    for idea in result.contribution_ideas:
        st.markdown(f"- {idea}")
    if repo.good_first_issues:
        st.markdown("**Open `good first issue` issues:**")
        for issue in repo.good_first_issues:
            st.markdown(f"- [#{issue['number']} {issue['title']}]({issue['url']})")
    else:
        st.caption("No open issues labelled 'good first issue' were found.")

    with st.expander("File structure"):
        st.code("\n".join(analysis.structure_lines(repo.tree)) or "(empty)", language=None)


model_name = config.OLLAMA_CHAT_MODEL if config.LLM_PROVIDER == "ollama" else config.GROQ_MODEL

with st.sidebar:
    st.header("About")
    st.write("Paste a public GitHub repository and get a summary, strengths and improvement ideas.")
    st.caption(f"Model: {config.LLM_PROVIDER} · {model_name}")
    st.caption("GitHub token: " + ("set" if config.GITHUB_TOKEN else "not set (60 requests per hour)"))

    st.subheader("Try an example")
    clicked = None
    for number, text in enumerate(EXAMPLES):
        if st.button(text, key=f"example_{number}", use_container_width=True):
            clicked = text

st.title("🔍 GitHub Repository Analyzer")

with st.form("analyze"):
    entered = st.text_input("Repository", placeholder="owner/repo or https://github.com/owner/repo")
    submitted = st.form_submit_button("Analyze", type="primary")

if clicked:
    st.session_state.target = clicked
elif submitted:
    st.session_state.target = entered

target = st.session_state.get("target")
if target is not None:
    try:
        owner, name = github.parse_repo(target)
        with st.spinner("Reading the repository from GitHub..."):
            repo = load_repo(owner, name)
    except github.GitHubError as error:
        if error.kind == "rate_limit":
            st.warning(str(error))
        else:
            st.error(str(error))
        st.stop()

    show_overview(repo)

    try:
        with st.spinner(f"Analysing with {model_name}... a local model can take a minute or two."):
            result = run_analysis(owner, name)
    except (LLMError, AnalysisError) as error:
        st.error(str(error))
        st.stop()
    show_analysis(result, repo)