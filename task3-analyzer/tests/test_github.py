import base64
import time

import httpx
import pytest

from analyzer import config
from analyzer.github import GitHubError, fetch_repo, parse_repo

INFO = {
    "name": "flask",
    "owner": {"login": "pallets"},
    "html_url": "https://github.com/pallets/flask",
    "description": "A micro web framework",
    "stargazers_count": 68000,
    "forks_count": 16000,
    "open_issues_count": 20,
    "language": "Python",
    "license": {"name": "BSD 3-Clause License"},
    "topics": ["python", "web"],
    "pushed_at": "2026-11-01T10:00:00Z",
    "archived": False,
    "default_branch": "main",
    "has_issues": True,
}


def encode(text):
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


def fake_github(routes, seen=None):
    """A fake GitHub. `routes` maps an API path to a JSON body or to an httpx.Response."""

    def handler(request):
        if seen is not None:
            seen.append(request)
        answer = routes.get(request.url.path)
        if answer is None:
            return httpx.Response(404, json={"message": "Not Found"})
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer, headers={"x-ratelimit-remaining": "55"})

    return httpx.Client(transport=httpx.MockTransport(handler))


BASE = "/repos/pallets/flask"
FULL_ROUTES = {
    BASE: INFO,
    f"{BASE}/languages": {"Python": 900, "HTML": 100},
    f"{BASE}/readme": encode("# Flask\nA small web framework."),
    f"{BASE}/git/trees/main": {
        "truncated": False,
        "tree": [
            {"path": "src", "type": "tree"},
            {"path": "src/flask/app.py", "type": "blob"},
            {"path": "pyproject.toml", "type": "blob"},
            {"path": "README.md", "type": "blob"},
        ],
    },
    f"{BASE}/contents/pyproject.toml": encode('[project]\nname = "flask"'),
    f"{BASE}/issues": [
        {"number": 1, "title": "Fix typo", "html_url": "https://github.com/pallets/flask/issues/1"},
        {"number": 2, "title": "A pull request", "html_url": "x", "pull_request": {}},
    ],
}


# ---------- parse_repo ----------

@pytest.mark.parametrize(
    "text",
    [
        "pallets/flask",
        "  pallets/flask  ",
        "https://github.com/pallets/flask",
        "https://github.com/pallets/flask/",
        "https://github.com/pallets/flask.git",
        "http://www.github.com/pallets/flask",
        "github.com/pallets/flask",
        "https://github.com/pallets/flask/tree/main/src",
        "https://github.com/pallets/flask/issues/12?x=1",
    ],
)
def test_valid_inputs(text):
    assert parse_repo(text) == ("pallets", "flask")


def test_dots_and_underscores_in_names_are_allowed():
    assert parse_repo("ollama/ollama-python") == ("ollama", "ollama-python")
    assert parse_repo("user/my_repo.js") == ("user", "my_repo.js")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "flask",
        "a/b/c",
        "https://gitlab.com/pallets/flask",
        "https://github.com/pallets",
        "https://github.com/",
        "https://example.com/github.com/pallets/flask",
        "pallets/",
        "/flask",
        "pallets/..",
        "-bad/flask",
        "pal lets/flask",
        "pallets/fl@sk",
    ],
)
def test_invalid_inputs(text):
    with pytest.raises(GitHubError) as error:
        parse_repo(text)
    assert error.value.kind == "invalid"


# ---------- fetch_repo ----------

def test_everything_is_collected():
    repo = fetch_repo("pallets", "flask", client=fake_github(FULL_ROUTES))
    assert repo.full_name == "pallets/flask"
    assert (repo.stars, repo.forks, repo.open_issues) == (68000, 16000, 20)
    assert repo.language == "Python"
    assert repo.license == "BSD 3-Clause License"
    assert repo.topics == ["python", "web"]
    assert repo.pushed_at == "2026-11-01"
    assert repo.languages == {"Python": 900, "HTML": 100}
    assert repo.readme.startswith("# Flask")
    assert len(repo.tree) == 4 and not repo.tree_truncated
    assert list(repo.package_files) == ["pyproject.toml"]
    assert repo.rate_remaining == 55


def test_pull_requests_are_not_listed_as_good_first_issues():
    repo = fetch_repo("pallets", "flask", client=fake_github(FULL_ROUTES))
    assert repo.good_first_issues == [
        {"number": 1, "title": "Fix typo", "url": "https://github.com/pallets/flask/issues/1"}
    ]


def test_good_first_issues_use_the_label_filter():
    seen = []
    fetch_repo("pallets", "flask", client=fake_github(FULL_ROUTES, seen))
    issue_request = [r for r in seen if r.url.path.endswith("/issues")][0]
    assert issue_request.url.params["labels"] == "good first issue"


def test_missing_readme_is_not_an_error():
    routes = {k: v for k, v in FULL_ROUTES.items() if not k.endswith("/readme")}
    assert fetch_repo("pallets", "flask", client=fake_github(routes)).readme == ""


def test_empty_repository_gives_an_empty_tree():
    routes = dict(FULL_ROUTES)
    routes[f"{BASE}/git/trees/main"] = httpx.Response(409, json={"message": "Git Repository is empty."})
    repo = fetch_repo("pallets", "flask", client=fake_github(routes))
    assert repo.tree == [] and repo.package_files == {}


def test_disabled_issues_are_skipped():
    routes = dict(FULL_ROUTES)
    routes[BASE] = dict(INFO, has_issues=False)
    assert fetch_repo("pallets", "flask", client=fake_github(routes)).good_first_issues == []


def test_missing_optional_fields_get_friendly_defaults():
    routes = dict(FULL_ROUTES)
    routes[BASE] = dict(INFO, description=None, language=None, license=None, topics=None)
    repo = fetch_repo("pallets", "flask", client=fake_github(routes))
    assert repo.description == ""
    assert repo.language == "Not detected"
    assert repo.license == "No license found"
    assert repo.topics == []


def test_only_a_few_package_files_are_read():
    names = ["package.json", "pyproject.toml", "requirements.txt", "setup.py", "go.mod"]
    routes = dict(FULL_ROUTES)
    routes[f"{BASE}/git/trees/main"] = {"tree": [{"path": n, "type": "blob"} for n in names]}
    for n in names:
        routes[f"{BASE}/contents/{n}"] = encode("content of " + n)
    repo = fetch_repo("pallets", "flask", client=fake_github(routes))
    assert len(repo.package_files) == config.MAX_PACKAGE_FILES


def test_long_package_files_are_cut():
    routes = dict(FULL_ROUTES)
    routes[f"{BASE}/contents/pyproject.toml"] = encode("x" * 5000)
    repo = fetch_repo("pallets", "flask", client=fake_github(routes))
    assert len(repo.package_files["pyproject.toml"]) == config.MAX_PACKAGE_CHARS


def test_a_package_file_in_a_subfolder_is_ignored():
    routes = dict(FULL_ROUTES)
    routes[f"{BASE}/git/trees/main"] = {"tree": [{"path": "docs/package.json", "type": "blob"}]}
    assert fetch_repo("pallets", "flask", client=fake_github(routes)).package_files == {}


# ---------- errors ----------

def test_unknown_or_private_repository():
    with pytest.raises(GitHubError) as error:
        fetch_repo("nobody", "nothing", client=fake_github({}))
    assert error.value.kind == "not_found"
    assert "may be private" in str(error.value)


def test_rate_limit_message_has_the_reset_time(monkeypatch):
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    reset = str(int(time.time()) + 600)
    limited = httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset})
    with pytest.raises(GitHubError) as error:
        fetch_repo("pallets", "flask", client=fake_github({BASE: limited}))
    assert error.value.kind == "rate_limit"
    assert "about 10 minute" in str(error.value)
    assert "GITHUB_TOKEN" in str(error.value)


def test_rate_limit_message_skips_the_token_hint_when_a_token_is_set(monkeypatch):
    monkeypatch.setattr(config, "GITHUB_TOKEN", "abc")
    limited = httpx.Response(429, headers={"retry-after": "30"})
    with pytest.raises(GitHubError) as error:
        fetch_repo("pallets", "flask", client=fake_github({BASE: limited}))
    assert error.value.kind == "rate_limit"
    assert "GITHUB_TOKEN" not in str(error.value)


def test_a_403_that_is_not_a_rate_limit():
    with pytest.raises(GitHubError) as error:
        fetch_repo("pallets", "flask", client=fake_github({BASE: httpx.Response(403)}))
    assert error.value.kind == "other"
    assert "403" in str(error.value)


def test_rejected_token():
    with pytest.raises(GitHubError, match="token was rejected"):
        fetch_repo("pallets", "flask", client=fake_github({BASE: httpx.Response(401)}))


def test_rate_limit_halfway_through_is_still_reported():
    routes = dict(FULL_ROUTES)
    routes[f"{BASE}/languages"] = httpx.Response(403, headers={"x-ratelimit-remaining": "0"})
    with pytest.raises(GitHubError) as error:
        fetch_repo("pallets", "flask", client=fake_github(routes))
    assert error.value.kind == "rate_limit"


def test_timeout_and_connection_problems():
    def slow(request):
        raise httpx.ReadTimeout("slow")

    def down(request):
        raise httpx.ConnectError("offline")

    with pytest.raises(GitHubError, match="too long"):
        fetch_repo("pallets", "flask", client=httpx.Client(transport=httpx.MockTransport(slow)))
    with pytest.raises(GitHubError, match="Could not reach"):
        fetch_repo("pallets", "flask", client=httpx.Client(transport=httpx.MockTransport(down)))


def test_unreadable_reply():
    routes = {BASE: httpx.Response(200, text="not json")}
    with pytest.raises(GitHubError, match="could not be read"):
        fetch_repo("pallets", "flask", client=fake_github(routes))