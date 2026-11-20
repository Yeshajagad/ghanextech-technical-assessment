"""Read a public repository through the GitHub REST API."""

import base64
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from analyzer import config

API_URL = "https://api.github.com"

# Files that describe a project's dependencies, in the order we look for them.
PACKAGE_FILES = [
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
]

OWNER_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


class GitHubError(Exception):
    """A problem worded for the user. `kind` lets the UI choose how to show it."""

    def __init__(self, message, kind="other"):
        super().__init__(message)
        self.kind = kind  # "invalid", "not_found", "rate_limit" or "other"


@dataclass
class Repo:
    owner: str
    name: str
    url: str
    description: str
    stars: int
    forks: int
    open_issues: int
    language: str
    license: str
    topics: list
    pushed_at: str
    archived: bool
    languages: dict = field(default_factory=dict)  # language -> bytes of code
    readme: str = ""
    tree: list = field(default_factory=list)  # [{"path": ..., "type": "blob" or "tree"}]
    tree_truncated: bool = False
    package_files: dict = field(default_factory=dict)  # file name -> text
    good_first_issues: list = field(default_factory=list)  # [{"number", "title", "url"}]
    rate_remaining: int = None  # API calls left this hour, as last reported by GitHub

    @property
    def full_name(self):
        return f"{self.owner}/{self.name}"


# ---------- reading the user's input ----------

def parse_repo(text):
    """Turn 'owner/repo' or a github.com link into (owner, repo). Raises GitHubError if it is neither."""
    problem = GitHubError(
        "That does not look like a GitHub repository. Enter owner/repo (for example pallets/flask) "
        "or a link such as https://github.com/pallets/flask.",
        "invalid",
    )
    text = text.strip()
    is_link = "://" in text or text.lower().startswith(("github.com/", "www.github.com/"))

    if is_link:
        if "://" not in text:
            text = "https://" + text
        link = urlparse(text)
        if link.hostname not in ("github.com", "www.github.com"):
            raise problem
        parts = [part for part in link.path.split("/") if part]
        if len(parts) < 2:
            raise problem
    else:
        parts = text.split("/")
        if len(parts) != 2:
            raise problem

    owner, name = parts[0], parts[1]
    if name.endswith(".git"):
        name = name[:-4]
    if not OWNER_PATTERN.match(owner) or not NAME_PATTERN.match(name) or name in (".", ".."):
        raise problem
    return owner, name


# ---------- talking to GitHub ----------

def _rate_limit_message(response):
    text = "GitHub's rate limit has been reached."
    reset = response.headers.get("x-ratelimit-reset", "")
    if reset.isdigit():
        minutes = max(1, round((int(reset) - time.time()) / 60))
        text += f" It resets in about {minutes} minute(s)."
    if not config.GITHUB_TOKEN:
        text += " Setting a GITHUB_TOKEN raises the limit from 60 to 5,000 requests per hour."
    return text


def _error_for(response):
    status = response.status_code
    limited = response.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in response.headers
    if status in (403, 429) and limited:
        return GitHubError(_rate_limit_message(response), "rate_limit")
    if status == 401:
        return GitHubError("The GitHub token was rejected. Check that GITHUB_TOKEN is correct.")
    if status == 404:
        return GitHubError(
            "Repository not found. It may not exist, or it may be private "
            "(only public repositories can be analysed).",
            "not_found",
        )
    return GitHubError(f"GitHub returned an error ({status}). Please try again later.")


class _Api:
    """A small wrapper that turns failures into GitHubError and remembers the rate limit."""

    def __init__(self, client):
        self.client = client
        self.remaining = None

    def get(self, path, params=None, missing=()):
        """Return the JSON for `path`, or None when the status is one of `missing`."""
        try:
            response = self.client.get(API_URL + path, params=params)
        except httpx.TimeoutException:
            raise GitHubError("GitHub took too long to respond. Try again in a moment.")
        except httpx.RequestError:
            raise GitHubError("Could not reach GitHub. Check your internet connection.")

        header = response.headers.get("x-ratelimit-remaining", "")
        if header.isdigit():
            self.remaining = int(header)

        if response.status_code in missing:
            return None
        if response.status_code != 200:
            raise _error_for(response)
        try:
            return response.json()
        except ValueError:
            raise GitHubError("GitHub sent a reply that could not be read.")


def _decode(data):
    """The text of a file from the 'contents' or 'readme' endpoints."""
    if not isinstance(data, dict) or data.get("encoding") != "base64":
        return ""
    return base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")


def _get_tree(api, base, branch):
    data = api.get(f"{base}/git/trees/{branch}", {"recursive": 1}, missing=(404, 409))  # 409 = empty repo
    if not data:
        return [], False
    items = [{"path": item["path"], "type": item["type"]} for item in data.get("tree", [])]
    return items, bool(data.get("truncated"))


def _get_package_files(api, base, tree):
    root_files = {item["path"] for item in tree if item["type"] == "blob" and "/" not in item["path"]}
    found = {}
    for filename in PACKAGE_FILES:
        if filename in root_files and len(found) < config.MAX_PACKAGE_FILES:
            text = _decode(api.get(f"{base}/contents/{filename}", missing=(404,)))
            if text:
                found[filename] = text[: config.MAX_PACKAGE_CHARS]
    return found


def _get_good_first_issues(api, base, info):
    if not info.get("has_issues"):
        return []
    data = api.get(
        f"{base}/issues",
        {"labels": "good first issue", "state": "open", "per_page": 10},
        missing=(404, 410),
    )
    # GitHub lists pull requests as issues too, so leave those out.
    issues = [item for item in (data or []) if "pull_request" not in item]
    return [
        {"number": item["number"], "title": item["title"], "url": item["html_url"]}
        for item in issues[:5]
    ]


def _fetch(api, owner, name):
    base = f"/repos/{owner}/{name}"
    info = api.get(base)
    languages = api.get(f"{base}/languages") or {}
    readme = _decode(api.get(f"{base}/readme", missing=(404,)))
    tree, truncated = _get_tree(api, base, info["default_branch"])
    packages = _get_package_files(api, base, tree)
    issues = _get_good_first_issues(api, base, info)

    return Repo(
        owner=info["owner"]["login"],
        name=info["name"],
        url=info["html_url"],
        description=info.get("description") or "",
        stars=info["stargazers_count"],
        forks=info["forks_count"],
        open_issues=info["open_issues_count"],
        language=info.get("language") or "Not detected",
        license=(info.get("license") or {}).get("name") or "No license found",
        topics=info.get("topics") or [],
        pushed_at=(info.get("pushed_at") or "")[:10],
        archived=bool(info.get("archived")),
        languages=languages,
        readme=readme,
        tree=tree,
        tree_truncated=truncated,
        package_files=packages,
        good_first_issues=issues,
        rate_remaining=api.remaining,
    )


def fetch_repo(owner, name, client=None):
    """Collect everything the analysis needs about one public repository."""
    own_client = client is None
    if own_client:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": config.USER_AGENT,
        }
        if config.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
        client = httpx.Client(timeout=config.HTTP_TIMEOUT, headers=headers)
    try:
        return _fetch(_Api(client), owner, name)
    finally:
        if own_client:
            client.close()