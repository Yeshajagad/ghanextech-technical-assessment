"""Look up a topic on Wikipedia and return the introduction of the best matching article."""

from urllib.parse import quote

from agent.tools.base import Tool, ToolError, get_json

API_URL = "https://en.wikipedia.org/w/api.php"


def search_wikipedia(query, client=None):
    query = query.strip()
    if not query:
        raise ToolError("Please give a topic to look up.")

    data = get_json(
        API_URL,
        {
            "action": "query",
            "format": "json",
            "generator": "search",  # search, then fetch the text of the top hit
            "gsrsearch": query,
            "gsrlimit": 1,
            "prop": "extracts",
            "exintro": 1,  # only the introduction
            "explaintext": 1,  # plain text, no HTML
            "exchars": 900,
            "redirects": 1,
        },
        client,
    )

    pages = (data.get("query") or {}).get("pages")
    if not pages:
        return f"No Wikipedia article was found for '{query}'."

    page = next(iter(pages.values()))
    title = page.get("title", query)
    extract = (page.get("extract") or "").strip()
    if not extract:
        return f"The Wikipedia article '{title}' has no summary text."

    link = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
    return f"{title}: {extract}\nSource: {link}"


TOOL = Tool(
    name="search_wikipedia",
    description=(
        "Look up a topic on Wikipedia and return the introduction of the best matching article. "
        "Use it for facts about people, places, events, science, history or concepts. "
        "Do not use it for live data such as weather or prices."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The topic to look up, for example 'Ada Lovelace' or 'photosynthesis'.",
            }
        },
        "required": ["query"],
    },
    func=search_wikipedia,
)