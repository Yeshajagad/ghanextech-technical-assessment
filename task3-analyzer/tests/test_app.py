"""Run the real Streamlit app in a test harness, with GitHub and the model replaced by fakes."""

import json
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from analyzer import config, github, llm
from analyzer.github import GitHubError, Repo

APP = Path(__file__).resolve().parent.parent / "app.py"

REPO = Repo(
    owner="pallets", name="flask", url="https://github.com/pallets/flask",
    description="A micro web framework", stars=68000, forks=16000, open_issues=20,
    language="Python", license="BSD 3-Clause License", topics=["python"],
    pushed_at="2026-11-01", archived=False, languages={"Python": 900, "HTML": 100},
    readme="# Flask", tree=[{"path": "tests/test_app.py", "type": "blob"}, {"path": "src/a.py", "type": "blob"}],
    good_first_issues=[{"number": 7, "title": "Fix typo", "url": "https://github.com/pallets/flask/issues/7"}],
    rate_remaining=50,
)

REPLY = json.dumps({
    "summary": "Flask is a small web framework.",
    "tech_stack": ["Python"],
    "strengths": ["Clear layout"],
    "architecture": "A src layout.",
    "recommendations": [
        {"title": "Add type hints", "why": "None seen."},
        {"title": "Add a changelog", "why": "No file found."},
        {"title": "Add a Dockerfile", "why": "None found."},
    ],
    "contribution_ideas": ["Fix issue #7"],
})


def start_app(monkeypatch, fetch=None, complete=None):
    st.cache_data.clear()  # results are cached, so each test must start clean
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(github, "fetch_repo", fetch or (lambda owner, name: REPO))
    monkeypatch.setattr(llm, "complete", complete or (lambda system, user, schema: REPLY))
    return AppTest.from_file(str(APP), default_timeout=30).run()


def analyze(at, text):
    at.text_input[0].set_value(text)
    at.button[0].click().run()  # the form's Analyze button is the first button in the main area
    return at


def all_markdown(at):
    return "\n".join(item.value for item in at.markdown)


def test_app_starts_empty_with_examples(monkeypatch):
    at = start_app(monkeypatch)
    assert not at.exception
    assert len(at.sidebar.button) == 4
    assert len(at.error) == 0 and len(at.metric) == 0


def test_a_repository_is_analysed_and_shown(monkeypatch):
    at = analyze(start_app(monkeypatch), "pallets/flask")
    assert not at.exception
    assert [m.label for m in at.metric] == ["Stars", "Forks", "Open issues", "Main language", "License"]
    assert at.metric[0].value == "68,000"
    text = all_markdown(at)
    assert "Flask is a small web framework." in text
    assert "1. Add type hints" in text and "3. Add a Dockerfile" in text
    assert "#7 Fix typo" in text
    assert "✅ Tests" in text and "❌ CI workflow" in text


def test_a_github_link_works_too(monkeypatch):
    seen = []

    def fetch(owner, name):
        seen.append((owner, name))
        return REPO

    analyze(start_app(monkeypatch, fetch=fetch), "https://github.com/pallets/flask.git")
    assert seen == [("pallets", "flask")]


def test_example_button_runs_an_analysis(monkeypatch):
    at = start_app(monkeypatch)
    at.sidebar.button[0].click().run()
    assert at.metric[0].value == "68,000"


def test_invalid_input_shows_a_friendly_error(monkeypatch):
    at = analyze(start_app(monkeypatch), "not a repo")
    assert not at.exception
    assert "does not look like a GitHub repository" in at.error[0].value
    assert len(at.metric) == 0


def test_private_or_missing_repository(monkeypatch):
    def fetch(owner, name):
        raise GitHubError("Repository not found. It may not exist, or it may be private.", "not_found")

    at = analyze(start_app(monkeypatch, fetch=fetch), "secret/repo")
    assert "may be private" in at.error[0].value


def test_rate_limit_is_a_warning_not_an_error(monkeypatch):
    def fetch(owner, name):
        raise GitHubError("GitHub's rate limit has been reached.", "rate_limit")

    at = analyze(start_app(monkeypatch, fetch=fetch), "pallets/flask")
    assert "rate limit" in at.warning[0].value
    assert len(at.error) == 0


def test_model_problem_keeps_the_repository_overview(monkeypatch):
    def broken(system, user, schema):
        raise llm.LLMError("Cannot reach Ollama.")

    at = analyze(start_app(monkeypatch, complete=broken), "pallets/flask")
    assert not at.exception
    assert at.metric[0].value == "68,000"  # the GitHub facts are still shown
    assert at.error[0].value == "Cannot reach Ollama."


def test_unusable_model_answer_is_reported(monkeypatch):
    at = analyze(start_app(monkeypatch, complete=lambda system, user, schema: "nonsense"), "pallets/flask")
    assert "usable analysis" in at.error[0].value