from types import SimpleNamespace

from agent import config
from agent.health import _installed, ollama_problems


class FakeClient:
    def __init__(self, names=None, error=None):
        self.names = names or []
        self.error = error

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(models=[SimpleNamespace(model=n) for n in self.names])


def test_name_without_tag_matches_any_tag():
    assert _installed(["qwen3:latest"], "qwen3")
    assert not _installed(["llama3.2:latest"], "qwen3")


def test_name_with_tag_must_match_exactly():
    assert _installed(["qwen3:8b"], "qwen3:8b")
    assert not _installed(["qwen3:latest"], "qwen3:8b")


def test_no_problems_when_the_model_is_installed():
    assert ollama_problems(FakeClient(names=["qwen3:4b"])) == []


def test_missing_model_is_reported_with_pull_command():
    problems = ollama_problems(FakeClient(names=["llama3.2:latest"]))
    assert problems == [
        "Model 'qwen3:4b' is missing. Run: ollama pull qwen3:4b"
    ]


def test_ollama_not_running_is_reported():
    problems = ollama_problems(FakeClient(error=ConnectionError("refused")))
    assert "not running" in problems[0]


def test_nothing_to_check_for_other_providers(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    assert ollama_problems(FakeClient(error=ConnectionError("refused"))) == []
