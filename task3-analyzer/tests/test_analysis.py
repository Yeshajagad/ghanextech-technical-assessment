import json

import pytest

from analyzer import analysis, config, llm
from analyzer.analysis import (
    AnalysisError,
    analyze,
    build_context,
    detect_signals,
    language_shares,
    parse_analysis,
    structure_lines,
)
from analyzer.github import Repo


def blobs(*paths):
    return [{"path": path, "type": "blob"} for path in paths]


def make_repo(**changes):
    values = dict(
        owner="pallets", name="flask", url="https://github.com/pallets/flask",
        description="A micro web framework", stars=68000, forks=16000, open_issues=20,
        language="Python", license="BSD 3-Clause License", topics=["python", "web"],
        pushed_at="2026-11-01", archived=False,
        languages={"Python": 900, "HTML": 100},
        readme="# Flask\nA small web framework.",
        tree=blobs("src/flask/app.py", "tests/test_app.py", "README.md"),
        package_files={"pyproject.toml": '[project]\nname = "flask"'},
        good_first_issues=[{"number": 7, "title": "Fix typo", "url": "u"}],
    )
    values.update(changes)
    return Repo(**values)


GOOD_REPLY = {
    "summary": "Flask is a small web framework.",
    "tech_stack": ["Python"],
    "strengths": ["Clear layout"],
    "architecture": "A src layout with tests.",
    "recommendations": [
        {"title": "Add type hints", "why": "None seen."},
        {"title": "Add a changelog", "why": "No file found."},
        {"title": "Add a Dockerfile", "why": "None found."},
    ],
    "contribution_ideas": ["Fix issue #7"],
}


# ---------- signals ----------

def test_signals_are_found():
    tree = blobs(
        "tests/test_app.py", ".github/workflows/ci.yml", "LICENSE", "CONTRIBUTING.md",
        "docs/index.md", "Dockerfile",
    )
    assert all(detect_signals(tree).values())


def test_signals_are_false_for_a_bare_repository():
    assert not any(detect_signals(blobs("main.py", "README.md")).values())


@pytest.mark.parametrize("path", ["test_app.py", "app_test.go", "src/app.test.ts", "web/Button.spec.jsx", "Tests/x.py"])
def test_different_test_layouts_are_recognised(path):
    assert detect_signals(blobs(path))["Tests"]


def test_a_file_that_only_looks_like_a_test_is_not_counted():
    assert not detect_signals(blobs("src/latest.py", "contest/readme.md"))["Tests"]


def test_other_ci_systems_are_recognised():
    assert detect_signals(blobs(".gitlab-ci.yml"))["CI workflow"]
    assert detect_signals(blobs("Jenkinsfile"))["CI workflow"]


def test_contributing_guide_inside_github_folder():
    assert detect_signals(blobs(".github/CONTRIBUTING.md"))["Contributing guide"]


def test_license_must_be_in_the_root():
    assert not detect_signals(blobs("vendor/LICENSE"))["License file"]
    assert detect_signals(blobs("LICENSE.txt"))["License file"]


# ---------- languages and structure ----------

def test_language_shares_add_up_and_are_sorted():
    assert language_shares({"HTML": 100, "Python": 900}) == {"Python": 90.0, "HTML": 10.0}


def test_language_shares_keep_only_the_top_few():
    many = {f"L{i}": 100 - i for i in range(10)}
    assert len(language_shares(many)) == 6


def test_no_languages():
    assert language_shares({}) == {}


def test_structure_shows_folders_with_counts_and_root_files():
    tree = blobs("src/a/x.py", "src/a/y.py", "src/b/z.py", "src/top.py", "README.md", "setup.py")
    assert structure_lines(tree) == [
        "src/ (4 files)",
        "  src/a/ (2 files)",
        "  src/b/ (1 files)",
        "README.md",
        "setup.py",
    ]


def test_structure_ignores_folders_and_submodules():
    tree = [{"path": "src", "type": "tree"}, {"path": "vendor/lib", "type": "commit"}] + blobs("a.py")
    assert structure_lines(tree) == ["a.py"]


def test_structure_is_limited():
    tree = blobs(*[f"file{i}.py" for i in range(200)])
    assert len(structure_lines(tree, limit=10)) == 10


# ---------- the prompt ----------

def test_context_contains_the_important_facts():
    text = build_context(make_repo())
    assert "Repository: pallets/flask" in text
    assert "Python 90.0%" in text
    assert "Tests: yes" in text
    assert "CI workflow: no" in text
    assert "--- pyproject.toml ---" in text
    assert "#7 Fix typo" in text
    assert text.rstrip().endswith("A small web framework.")  # the README comes last


def test_long_readme_is_cut():
    text = build_context(make_repo(readme="x" * (config.MAX_README_CHARS + 500)))
    assert "[README cut here]" in text
    assert "x" * (config.MAX_README_CHARS + 1) not in text


def test_missing_readme_and_issues_are_stated():
    text = build_context(make_repo(readme="", good_first_issues=[], tree=[]))
    assert "no README" in text
    assert "good first issue': none found" in text
    assert "empty repository" in text


def test_truncated_tree_is_mentioned():
    assert "only part of the file list" in build_context(make_repo(tree_truncated=True))


# ---------- reading the model's answer ----------

def test_a_good_reply_is_parsed():
    result = parse_analysis(json.dumps(GOOD_REPLY))
    assert result.summary == "Flask is a small web framework."
    assert result.recommendations[0] == {"title": "Add type hints", "why": "None seen."}
    assert result.contribution_ideas == ["Fix issue #7"]


def test_json_inside_a_code_fence_or_with_chatter_is_accepted():
    wrapped = "Sure! Here you go:\n```json\n" + json.dumps(GOOD_REPLY) + "\n```"
    assert parse_analysis(wrapped).summary


def test_thinking_text_is_removed():
    reply = "<think>{not this}</think>" + json.dumps(GOOD_REPLY)
    assert parse_analysis(reply).summary


def test_no_json():
    with pytest.raises(AnalysisError, match="no JSON"):
        parse_analysis("I cannot help with that.")


def test_broken_json():
    with pytest.raises(AnalysisError, match="could not be read"):
        parse_analysis('{"summary": "x",}')


def test_missing_summary():
    with pytest.raises(AnalysisError, match="summary"):
        parse_analysis(json.dumps(dict(GOOD_REPLY, summary="  ")))


def test_at_least_three_recommendations_are_required():
    with pytest.raises(AnalysisError, match="three recommendations"):
        parse_analysis(json.dumps(dict(GOOD_REPLY, recommendations=GOOD_REPLY["recommendations"][:2])))


def test_recommendations_given_as_plain_strings_are_accepted():
    reply = dict(GOOD_REPLY, recommendations=["Add tests", "Add CI", "Add docs"])
    assert parse_analysis(json.dumps(reply)).recommendations[1] == {"title": "Add CI", "why": ""}


def test_wrong_types_do_not_crash():
    reply = dict(GOOD_REPLY, tech_stack="Python", strengths=[1, None, "ok"], architecture=5, recommendations="abc")
    with pytest.raises(AnalysisError, match="three recommendations"):
        parse_analysis(json.dumps(reply))
    reply["recommendations"] = GOOD_REPLY["recommendations"]
    result = parse_analysis(json.dumps(reply))
    assert result.tech_stack == [] and result.strengths == ["ok"] and result.architecture == ""


# ---------- analyze ----------

def test_analyze_returns_the_parsed_result(monkeypatch):
    seen = []

    def fake(system, user, schema):
        seen.append((system, user, schema))
        return json.dumps(GOOD_REPLY)

    monkeypatch.setattr(llm, "complete", fake)
    result = analyze(make_repo())
    assert result.summary.startswith("Flask")
    system, user, schema = seen[0]
    assert "NOT given the source code" in system
    assert "pallets/flask" in user
    assert schema is analysis.SCHEMA


def test_analyze_tries_again_after_a_bad_reply(monkeypatch):
    replies = iter(["nonsense", json.dumps(GOOD_REPLY)])
    requests = []

    def fake(system, user, schema):
        requests.append(user)
        return next(replies)

    monkeypatch.setattr(llm, "complete", fake)
    assert analyze(make_repo()).summary
    assert "not usable" in requests[1] and "no JSON" in requests[1]


def test_analyze_gives_up_after_two_bad_replies(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *args: "nonsense")
    with pytest.raises(AnalysisError, match="did not return a usable analysis"):
        analyze(make_repo())


def test_model_errors_are_passed_on(monkeypatch):
    def broken(*args):
        raise llm.LLMError("Cannot reach Ollama.")

    monkeypatch.setattr(llm, "complete", broken)
    with pytest.raises(llm.LLMError):
        analyze(make_repo())