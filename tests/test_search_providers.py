"""Official search APIs (Brave, Tavily), the unofficial DuckDuckGo switch, and caching rules."""
import json

import httpx

from conftest import collect, run
from proxmino.config import Config
from proxmino.db import DB
from proxmino.pipeline import TurnRequest
from proxmino.search import BraveProvider, SearchService, TavilyProvider


def cfg(**search):
    return Config(overrides={"search": search})


def test_providers_follow_keys_and_the_unofficial_switch(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    s = SearchService.from_config(cfg(providers=["brave", "tavily", "searxng"], searxng_url="", allow_unofficial=False), None, None)
    assert s.providers == []  # nothing set up: no keys, no SearXNG, DuckDuckGo off
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    s = SearchService.from_config(cfg(providers=["brave", "tavily"], allow_unofficial=True), None, None)
    assert [p.name for p in s.providers] == ["tavily", "ddgs"]  # keyed first, the unofficial one last
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "b-test")
    s = SearchService.from_config(cfg(providers=["brave", "tavily"], allow_unofficial=False), None, None)
    assert [p.name for p in s.providers] == ["brave", "tavily"]


def test_brave_and_tavily_requests_and_parsing():
    seen = []

    def handler(req):
        seen.append(req)
        if "brave" in req.url.host:
            return httpx.Response(200, json={"web": {"results": [
                {"title": "Mars facts", "url": "https://nasa.gov/mars", "description": "Mars is cold.", "page_age": "2026-01-02"}]}})
        return httpx.Response(200, json={"results": [
            {"title": "Mars news", "url": "https://news.example/mars", "content": "Rover update.", "published_date": "2026-10-28", "score": 0.9}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            b = await BraveProvider("bk", c).search("mars", "news", 5)
            t = await TavilyProvider("tk", c).search("mars", "news", 5)
            return b, t
    b, t = run(go())
    assert (b[0].url, b[0].title, b[0].snippet, b[0].published) == ("https://nasa.gov/mars", "Mars facts", "Mars is cold.", "2026-01-02")
    assert (t[0].url, t[0].snippet, t[0].published) == ("https://news.example/mars", "Rover update.", "2026-10-28")
    brave, tavily = seen
    assert brave.headers["x-subscription-token"] == "bk" and brave.url.params["freshness"] == "pw"
    assert tavily.headers["authorization"] == "Bearer tk"
    assert json.loads(tavily.content)["topic"] == "news"


def test_official_api_results_are_not_cached():
    class Counting(TavilyProvider):
        calls = 0

        async def search(self, q, kind="web", n=6):
            Counting.calls += 1
            from proxmino.search import SearchResult
            return [SearchResult("https://a.example", "A", "a")]
    db = DB(":memory:")
    s = SearchService([Counting("k", None)], db=db)
    run(s.search("same question"))
    run(s.search("same question"))
    assert Counting.calls == 2


def test_turn_explains_when_search_is_not_set_up(engine, fake):
    engine.search.providers = []
    evs = run(collect(engine, TurnRequest(message="what happened today in the news?", search=True)))
    steps = " ".join(d["text"] for k, d in evs if k == "step")
    assert "Web search is not set up" in steps


def test_wikipedia_is_added_next_to_web_results():
    import asyncio
    import httpx
    from proxmino.search import FixtureProvider, SearchResult, SearchService, WikipediaProvider

    def handler(req):
        assert req.url.host == "en.wikipedia.org" and "Proxmino" in req.headers["user-agent"]
        assert req.url.params["maxlag"] == "5"
        return httpx.Response(200, json={"query": {"pages": [
            {"index": 2, "title": "Beta", "fullurl": "https://en.wikipedia.org/wiki/Beta", "extract": "Beta is a letter."},
            {"index": 1, "title": "Alpha", "fullurl": "https://en.wikipedia.org/wiki/Alpha", "extract": "Alpha is first."}]}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            web = FixtureProvider([SearchResult("https://example.org/a", "A", "a")])
            both = SearchService([web, WikipediaProvider(c)], None, c)
            alone = SearchService([WikipediaProvider(c)], None, c)
            return await both.search("alpha"), await alone.search("alpha"), await both.search("alpha", "news")

    both, alone, news = asyncio.run(go())
    assert [r.url for r in both] == ["https://en.wikipedia.org/wiki/Alpha", "https://en.wikipedia.org/wiki/Beta", "https://example.org/a"]
    assert both[0].title == "Alpha (Wikipedia)" and both[0].content == "Alpha is first."  # no second fetch needed
    assert [r.title for r in alone] == ["Alpha (Wikipedia)", "Beta (Wikipedia)"]  # works with no search key at all
    assert [r.url for r in news] == ["https://example.org/a"]  # news stays with news providers
