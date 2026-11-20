"""Turn raw repository data into facts, a prompt for the model, and a checked analysis."""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from analyzer import config, llm

TEST_FOLDERS = {"test", "tests", "__tests__", "spec", "specs"}
TEST_FILE = re.compile(r"(^test_.*\.py$)|(_test\.(py|go)$)|(\.(test|spec)\.[jt]sx?$)")
CI_FOLDERS = (".github/workflows/", ".circleci/")
CI_FILES = {".gitlab-ci.yml", ".travis.yml", "jenkinsfile", "azure-pipelines.yml"}
DOCKER_FILES = {"dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yaml"}
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

SYSTEM_PROMPT = """You are a senior software engineer reviewing an open-source repository for a newcomer.
You are given the repository's details, file layout, package files and README. You are NOT given the source code.

Rules:
- Use only the information provided. If something cannot be known from it, say so instead of guessing.
- Be specific to this repository. No generic advice.
- Reply with one JSON object and nothing else, using these keys:
  summary: 2 to 3 sentences on what the project is and who it is for
  tech_stack: a short list of languages, frameworks and tools
  strengths: 3 to 5 short points
  architecture: 2 to 4 sentences on how the project appears to be organised
  recommendations: 3 to 5 objects, each {"title": a short action, "why": one sentence tied to something you saw, such as "no tests folder was found"}
  contribution_ideas: 2 to 4 concrete ways a newcomer could contribute; mention the listed good first issues if there are any"""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "tech_stack": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "architecture": {"type": "string"},
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "why": {"type": "string"}},
                "required": ["title", "why"],
            },
        },
        "contribution_ideas": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "tech_stack", "strengths", "architecture", "recommendations", "contribution_ideas"],
}


class AnalysisError(Exception):
    """The model's answer could not be used."""


@dataclass
class Analysis:
    summary: str
    tech_stack: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    architecture: str = ""
    recommendations: list = field(default_factory=list)  # [{"title", "why"}]
    contribution_ideas: list = field(default_factory=list)


# ---------- facts worked out in code, not by the model ----------

def detect_signals(tree):
    """Check the file list for tests, CI, a license and so on. Returns {label: True/False}."""
    files = [item["path"] for item in tree if item["type"] == "blob"]
    root_files = {path.lower() for path in files if "/" not in path}

    def has_tests(path):
        folders = {part.lower() for part in path.split("/")[:-1]}
        return bool(folders & TEST_FOLDERS) or bool(TEST_FILE.search(path.split("/")[-1]))

    def has_ci(path):
        return path.startswith(CI_FOLDERS) or path.lower() in CI_FILES

    def is_contributing(path):
        return path.count("/") <= 1 and path.split("/")[-1].lower().startswith("contributing")

    return {
        "Tests": any(has_tests(path) for path in files),
        "CI workflow": any(has_ci(path) for path in files),
        "License file": any(name.startswith(("license", "licence", "copying")) for name in root_files),
        "Contributing guide": any(is_contributing(path) for path in files),
        "Docs folder": any(path.split("/")[0].lower() in ("docs", "doc") for path in files if "/" in path),
        "Dockerfile": bool(root_files & DOCKER_FILES),
    }


def language_shares(languages, top=6):
    """Percentage of code per language, biggest first. GitHub reports bytes."""
    total = sum(languages.values())
    if not total:
        return {}
    ranked = sorted(languages.items(), key=lambda item: item[1], reverse=True)[:top]
    return {name: round(size * 100 / total, 1) for name, size in ranked}


def structure_lines(tree, limit=None):
    """A short outline of the repository: folders with file counts, two levels deep, then root files."""
    limit = limit or config.MAX_TREE_LINES
    top = Counter()
    second = defaultdict(Counter)
    root_files = []
    for item in tree:
        if item["type"] != "blob":
            continue
        parts = item["path"].split("/")
        if len(parts) == 1:
            root_files.append(parts[0])
            continue
        top[parts[0]] += 1
        if len(parts) > 2:
            second[parts[0]][parts[1]] += 1

    lines = []
    for folder in sorted(top):
        lines.append(f"{folder}/ ({top[folder]} files)")
        for child, count in sorted(second[folder].items())[:8]:
            lines.append(f"  {folder}/{child}/ ({count} files)")
    lines += sorted(root_files)
    return lines[:limit]


# ---------- the prompt ----------

def build_context(repo):
    """Everything the model is allowed to see, as plain text."""
    signals = ", ".join(f"{label}: {'yes' if found else 'no'}" for label, found in detect_signals(repo.tree).items())
    shares = ", ".join(f"{name} {percent}%" for name, percent in language_shares(repo.languages).items())

    parts = [
        f"Repository: {repo.full_name}",
        f"Description: {repo.description or 'none'}",
        f"Stars: {repo.stars}, forks: {repo.forks}, open issues: {repo.open_issues}, "
        f"license: {repo.license}, last push: {repo.pushed_at}, archived: {'yes' if repo.archived else 'no'}",
        f"Topics: {', '.join(repo.topics) or 'none'}",
        f"Languages by share of code: {shares or 'unknown'}",
        f"Found in the file list: {signals}",
        "Layout (folders with file counts):\n" + ("\n".join(structure_lines(repo.tree)) or "(empty repository)"),
    ]
    if repo.tree_truncated:
        parts.append("Note: the repository is so large that GitHub returned only part of the file list.")

    for filename, text in repo.package_files.items():
        parts.append(f"--- {filename} ---\n{text}")

    if repo.good_first_issues:
        issues = "\n".join(f"- #{issue['number']} {issue['title']}" for issue in repo.good_first_issues)
        parts.append("Open issues labelled 'good first issue':\n" + issues)
    else:
        parts.append("Open issues labelled 'good first issue': none found")

    readme = repo.readme[: config.MAX_README_CHARS]
    if len(repo.readme) > len(readme):
        readme += "\n[README cut here]"
    parts.append("--- README ---\n" + (readme or "(this repository has no README)"))
    return "\n\n".join(parts)


# ---------- the model's answer ----------

def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _list(value):
    return [item.strip() for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def parse_analysis(text):
    """Read the model's reply. Raises AnalysisError saying what is wrong with it."""
    text = THINK_BLOCK.sub("", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise AnalysisError("no JSON object was found")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        raise AnalysisError("the JSON could not be read")

    summary = _text(data.get("summary"))
    if not summary:
        raise AnalysisError("the summary is missing")

    recommendations = []
    items = data.get("recommendations")
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and _text(item.get("title")):
            recommendations.append({"title": _text(item["title"]), "why": _text(item.get("why"))})
        elif isinstance(item, str) and item.strip():
            recommendations.append({"title": item.strip(), "why": ""})
    if len(recommendations) < 3:
        raise AnalysisError("fewer than three recommendations were given")

    return Analysis(
        summary=summary,
        tech_stack=_list(data.get("tech_stack")),
        strengths=_list(data.get("strengths")),
        architecture=_text(data.get("architecture")),
        recommendations=recommendations,
        contribution_ideas=_list(data.get("contribution_ideas")),
    )


def analyze(repo):
    """Ask the model for an analysis of `repo`. Tries twice if the first answer is unusable."""
    request = "Analyse this repository.\n\n" + build_context(repo)
    for _ in range(2):
        reply = llm.complete(SYSTEM_PROMPT, request, SCHEMA)
        try:
            return parse_analysis(reply)
        except AnalysisError as problem:
            request += f"\n\nYour previous reply was not usable ({problem}). Reply again with valid JSON only."
    raise AnalysisError("The model did not return a usable analysis. Please try again.")