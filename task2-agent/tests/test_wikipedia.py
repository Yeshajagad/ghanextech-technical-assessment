import httpx
import pytest

from agent.tools.base import ToolError
from agent.tools.wikipedia import search_wikipedia


def make_client(body, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(dict(request.url.params))
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_returns_title_summary_and_link():
    body = {"query": {"pages": {"1": {"title": "Ada Lovelace", "extract": "Ada was a mathematician."}}}}
    text = search_wikipedia("Ada Lovelace", client=make_client(body))
    assert text.startswith("Ada Lovelace: Ada was a mathematician.")
    assert text.endswith("Source: https://en.wikipedia.org/wiki/Ada_Lovelace")


def test_search_options_are_sent():
    seen = []
    body = {"query": {"pages": {"1": {"title": "X", "extract": "y"}}}}
    search_wikipedia("black holes", client=make_client(body, seen))
    assert seen[0]["gsrsearch"] == "black holes"
    assert seen[0]["exintro"] == "1"


def test_no_article_found_is_not_an_error():
    assert "No Wikipedia article was found" in search_wikipedia("zzzzqqqq", client=make_client({}))


def test_article_without_text():
    body = {"query": {"pages": {"1": {"title": "Empty", "extract": "  "}}}}
    assert "has no summary" in search_wikipedia("Empty", client=make_client(body))


def test_link_is_url_safe():
    body = {"query": {"pages": {"1": {"title": "Café (drink)", "extract": "A drink."}}}}
    text = search_wikipedia("cafe", client=make_client(body))
    assert "wiki/Caf%C3%A9_%28drink%29" in text


def test_empty_query():
    with pytest.raises(ToolError, match="topic"):
        search_wikipedia("")


def test_rate_limit_is_reported():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429)))
    with pytest.raises(ToolError, match="limiting requests"):
        search_wikipedia("anything", client=client)