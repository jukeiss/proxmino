"""Web search: providers, cached results, page fetching, extraction and ranking.

Official search APIs (Brave Search, Tavily) take the customer's own key from .env. SearXNG is a
search server the user runs. DuckDuckGo through `ddgs` is unofficial (it automates the website,
and the library says it is for educational purposes only), so it runs only when
`search.allow_unofficial` is on: fine for personal use, not for a product you sell.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import math
import os
import re
import socket
from collections import Counter
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

log = logging.getLogger("proxmino.search")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    published: str = ""
    source: str = ""
    content: str = ""  # full extracted text when fetched

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.")


@dataclass
class Evidence:
    """A numbered source handed to the model and shown to the user."""

    n: int
    url: str
    title: str
    domain: str
    snippet: str
    published: str = ""
    passages: list[str] = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("passages")
        return d


class SearchProvider:
    name = "base"
    cacheable = True  # official APIs set False: providers can restrict storing their results

    async def search(self, q: str, kind: str = "web", n: int = 6) -> list[SearchResult]:
        raise NotImplementedError


class DDGSProvider(SearchProvider):
    name = "ddgs"

    async def search(self, q, kind="web", n=6):
        from ddgs import DDGS

        def run():
            with DDGS() as d:
                if kind == "news":
                    return [SearchResult(r.get("url", ""), r.get("title", ""), r.get("body", ""),
                                         r.get("date", ""), self.name) for r in d.news(q, max_results=n)]
                return [SearchResult(r.get("href", ""), r.get("title", ""), r.get("body", ""), "", self.name)
                        for r in d.text(q, max_results=n)]
        return await asyncio.to_thread(run)


class BraveProvider(SearchProvider):
    """Brave Search API: https://api-dashboard.search.brave.com (official, needs a key)."""

    name = "brave"
    cacheable = False

    def __init__(self, key: str, client: httpx.AsyncClient):
        self.key, self.client = key, client

    async def search(self, q, kind="web", n=6):
        params = {"q": q, "count": max(1, min(n, 20))}
        if kind == "news":
            params["freshness"] = "pw"  # last 7 days
        r = await self.client.get("https://api.search.brave.com/res/v1/web/search", params=params, timeout=10,
                                  headers={"X-Subscription-Token": self.key, "Accept": "application/json"})
        r.raise_for_status()
        return [SearchResult(x.get("url", ""), x.get("title", ""), x.get("description", ""),
                             x.get("page_age") or x.get("age") or "", self.name)
                for x in (r.json().get("web") or {}).get("results", [])[:n]]


class TavilyProvider(SearchProvider):
    """Tavily search API: https://docs.tavily.com (official, needs a key)."""

    name = "tavily"
    cacheable = False

    def __init__(self, key: str, client: httpx.AsyncClient):
        self.key, self.client = key, client

    async def search(self, q, kind="web", n=6):
        body = {"query": q, "max_results": max(1, min(n, 20)), "topic": "news" if kind == "news" else "general",
                "search_depth": "basic"}
        r = await self.client.post("https://api.tavily.com/search", json=body, timeout=15,
                                   headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        return [SearchResult(x.get("url", ""), x.get("title", ""), x.get("content", ""),
                             x.get("published_date") or "", self.name)
                for x in r.json().get("results", [])[:n]]


KEYED = {"brave": (BraveProvider, "BRAVE_SEARCH_API_KEY"), "tavily": (TavilyProvider, "TAVILY_API_KEY")}


class WikipediaProvider(SearchProvider):
    """Wikipedia through the official MediaWiki API: free, no key, a trusted reference added next to web results.
    Follows Wikimedia's API rules: a descriptive User-Agent, one request per search, maxlag. Text is CC BY-SA, so
    every answer shows the article's title and link as its source (the attribution the license asks for)."""

    name = "wikipedia"

    def __init__(self, client: httpx.AsyncClient, lang: str = "en"):
        self.client, self.lang = client, (lang or "en").lower()[:12]

    async def search(self, q, kind="web", n=3):
        r = await self.client.get(f"https://{self.lang}.wikipedia.org/w/api.php", timeout=10, headers={
            "User-Agent": "Proxmino/1.0 (self-hosted personal assistant; https://www.mediawiki.org/wiki/API:Etiquette)"}, params={
            "action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrsearch": q[:300],
            "gsrlimit": max(1, min(n, 10)), "prop": "extracts|info", "exintro": "1", "explaintext": "1",
            "exlimit": max(1, min(n, 10)), "inprop": "url", "maxlag": "5"})
        r.raise_for_status()
        pages = sorted((r.json().get("query") or {}).get("pages", []), key=lambda x: x.get("index", 99))
        return [SearchResult(x.get("fullurl", ""), f"{x.get('title', '')} (Wikipedia)", (x.get("extract") or "")[:300],
                             x.get("touched", "")[:10], self.name, content=x.get("extract") or "")
                for x in pages if x.get("fullurl") and x.get("extract")][:n]


class SearXNGProvider(SearchProvider):
    name = "searxng"

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base, self.client = base_url.rstrip("/"), client

    async def search(self, q, kind="web", n=6):
        r = await self.client.get(f"{self.base}/search", timeout=10, params={
            "q": q, "format": "json", "categories": "news" if kind == "news" else "general"})
        r.raise_for_status()
        return [SearchResult(x.get("url", ""), x.get("title", ""), x.get("content", ""),
                             x.get("publishedDate") or "", self.name)
                for x in r.json().get("results", [])[:n]]


class FixtureProvider(SearchProvider):
    """Offline provider for tests: maps a query substring to results."""

    name = "fixture"

    def __init__(self, results: list[SearchResult] | dict[str, list[SearchResult]]):
        self.results = results
        self.queries: list[str] = []

    async def search(self, q, kind="web", n=6):
        self.queries.append(q)
        if isinstance(self.results, list):
            return self.results[:n]
        for key, res in self.results.items():
            if key.lower() in q.lower():
                return res[:n]
        return []


# ---------- extraction ----------

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "iframe"}
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "tr", "br", "section", "article", "pre", "td"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def extract_text(html: str) -> str:
    try:
        import trafilatura  # optional, better quality
        text = trafilatura.extract(html, include_comments=False)
        if text:
            return text
    except ImportError:
        pass
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    text = "".join(p.parts)
    lines = (re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if len(ln) > 30)


def _is_public_ip(addr: str) -> bool:
    """Only globally routable unicast addresses. Blocks loopback, private, link-local, 0.0.0.0, CGNAT/Tailscale
    (100.64/10), IPv4-mapped IPv6 forms of all of those, multicast and reserved ranges."""
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _is_public_host(host: str) -> bool:
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return False
    return bool(infos) and all(_is_public_ip(info[4][0]) for info in infos)


MAX_REDIRECTS = 5
SPLIT_YEAR = re.compile(r"\b(19|20) (\d{2})\b")
FETCH_DEADLINE = 6.0  # seconds for all page fetches of one search together


async def safe_get(client: httpx.AsyncClient, url: str, resolve=_is_public_host) -> httpx.Response | None:
    """GET a public web page and nothing else. Redirects are followed one hop at a time and every hop must be
    http(s) on a public address; the address actually connected to is checked too (DNS rebinding).
    Returns None when the page is not allowed."""
    for _ in range(MAX_REDIRECTS + 1):
        u = urlparse(url)
        if u.scheme not in ("http", "https") or u.username or u.password:
            return None
        if not await asyncio.to_thread(resolve, u.hostname or ""):
            log.info("blocked non-public address: %s", u.hostname)
            return None
        resp = await client.get(url, timeout=8, follow_redirects=False, headers={"User-Agent": UA})
        stream = resp.extensions.get("network_stream")
        peer = stream.get_extra_info("server_addr") if stream is not None else None
        if peer and not _is_public_ip(str(peer[0])):
            log.info("blocked: %s resolved to a non-public address at connect time", u.hostname)
            return None
        if resp.is_redirect and resp.headers.get("location"):
            url = str(resp.url.join(resp.headers["location"]))
            continue
        return resp
    return None


# ---------- ranking ----------

_WORD = re.compile(r"[a-z0-9]{2,}")
_STOP = set("the and for are but not you all any can had her was one our out has his how its may new now "
            "see two who did get him let say she too use that with have this will your from they what when "
            "make like time than them into just over also some these would there their which about".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def chunk(text: str, size: int = 700) -> list[str]:
    paras, out, cur = [p.strip() for p in text.split("\n") if p.strip()], [], ""
    for para in paras:
        if len(cur) + len(para) > size and cur:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{para}" if cur else para
        while len(cur) > size * 1.6:
            out.append(cur[:size])
            cur = cur[size:]
    if cur:
        out.append(cur)
    return out


def rank(query: str, passages: list[str], k: int) -> list[int]:
    """BM25 over the candidate passages; returns indices of the best k."""
    if not passages:
        return []
    docs = [_tokens(p) for p in passages]
    q = set(_tokens(query))
    avg = sum(len(d) for d in docs) / len(docs) or 1
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append((s, -i, i))
    return [i for _, _, i in sorted(scores, reverse=True)[:k]]


# ---------- service ----------

class SearchService:
    def __init__(self, providers: list[SearchProvider], db=None, client: httpx.AsyncClient | None = None,
                 results: int = 6, fetch_pages: int = 4, cache_minutes: float = 30):
        self.providers, self.db, self.client = providers, db, client
        self.results, self.fetch_pages, self.cache_s = results, fetch_pages, cache_minutes * 60
        self.fetch_sem = asyncio.Semaphore(6)

    @classmethod
    def from_config(cls, cfg, db, client):
        """Providers in the configured order. Keyed APIs without a key, and the unofficial DuckDuckGo
        search unless allow_unofficial is on, are skipped."""
        provs: list[SearchProvider] = []
        for name in cfg.get("search.providers", ["brave", "tavily", "searxng"]):
            if name in KEYED:
                cls_, env = KEYED[name]
                key = os.environ.get(cfg.get(f"search.{name}_key_env", env) or env)
                if key:
                    provs.append(cls_(key, client))
            elif name == "searxng" and (url := cfg.get("search.searxng_url") or os.environ.get("PROXMINO_SEARXNG_URL")):
                provs.append(SearXNGProvider(url, client))  # the server packages run their own SearXNG
        if cfg.get("search.allow_unofficial", False):
            provs.append(DDGSProvider())  # personal use only, always last
        if cfg.get("search.wikipedia", True) and client:
            provs.append(WikipediaProvider(client, cfg.get("search.wikipedia_lang", "en")))  # added to, not instead of, web results
        return cls(provs, db, client, cfg.get("search.results", 6), cfg.get("search.fetch_pages", 4),
                   cfg.get("search.cache_minutes", 30))

    async def search(self, q: str, kind: str = "web") -> list[SearchResult]:
        key = hashlib.sha1(f"{kind}|{q.lower().strip()}".encode()).hexdigest()
        if self.db and (hit := self.db.cache_get(key)) is not None:
            return [SearchResult(**r) for r in hit]
        wiki = next((p for p in self.providers if isinstance(p, WikipediaProvider)), None)
        res, cacheable = [], True
        for prov in self.providers:
            if prov is wiki:
                continue
            try:
                res = [r for r in await prov.search(q, kind, self.results) if r.url.startswith("http")]
            except Exception as e:  # rate limits, network, parsing: try the next provider
                log.warning("search provider %s failed: %s", prov.name, e)
                continue
            if res:
                cacheable = prov.cacheable and not isinstance(prov, FixtureProvider)
                break
        if wiki and kind == "web":  # a trusted reference first, then the web
            try:
                ref = await wiki.search(q, kind, 2)
            except Exception as e:
                log.warning("search provider wikipedia failed: %s", e)
                ref = []
            urls = {r.url for r in ref}
            res = ref + [r for r in res if r.url not in urls]
        if res and self.db and cacheable:
            self.db.cache_put(key, [asdict(r) for r in res], self.cache_s)
        return res

    async def fetch(self, r: SearchResult) -> SearchResult:
        if r.content or not self.client or not self.fetch_pages:
            return r
        async with self.fetch_sem:
            try:
                resp = await safe_get(self.client, r.url)
                ctype = resp.headers.get("content-type", "") if resp else ""
                if resp is not None and resp.status_code < 400 and "html" in ctype and len(resp.content) < 3_000_000:
                    r.content = (await asyncio.to_thread(extract_text, resp.text))[:40_000]
            except httpx.HTTPError as e:
                log.info("fetch failed %s: %s", r.url, e)
        return r

    async def gather(self, queries: list[str], question: str, kind: str = "web",
                     max_sources: int = 8, passages_per_source: int = 2) -> list[Evidence]:
        """Run queries in parallel, read top pages, and return ranked numbered evidence."""
        batches = await asyncio.gather(*(self.search(q, kind) for q in queries))
        seen, results = set(), []
        for batch in zip(*[b + [None] * (max(map(len, batches), default=0) - len(b)) for b in batches]):
            for r in batch:  # interleave so each query contributes its top results
                if r and r.url not in seen:
                    seen.add(r.url)
                    results.append(r)
        results = results[: max_sources + 2]
        to_fetch = results[: self.fetch_pages * max(1, len(queries))]
        # One slow site must not hold the whole answer: use whatever pages have loaded after a few seconds
        tasks = [asyncio.ensure_future(self.fetch(r)) for r in to_fetch]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=FETCH_DEADLINE)
            for t in pending:
                t.cancel()

        evidence: list[Evidence] = []
        for r in results:
            r.title = SPLIT_YEAR.sub(r"\1\2", r.title or "")
            r.snippet = SPLIT_YEAR.sub(r"\1\2", r.snippet or "")  # "20 26" from pages that style digits apart
            r.content = SPLIT_YEAR.sub(r"\1\2", re.sub(r"(\d)\*\*(\d+)\*\*", r"\1\2", r.content)) if r.content else r.content
            passages = chunk(r.content) if r.content else []
            k = 5 if r.source == "wikipedia" else passages_per_source  # a Wikipedia summary is short and dense: keep most of it
            best = [passages[i] for i in sorted(rank(question, passages, k))]
            if not best and not r.snippet:
                continue
            evidence.append(Evidence(len(evidence) + 1, r.url, r.title, r.domain, r.snippet,
                                     r.published, best))
            if len(evidence) >= max_sources:
                break
        return evidence


def format_sources(evidence: list[Evidence], char_budget: int = 9000) -> str:
    """Sources as delimited, untrusted data blocks for the prompt."""
    per = max(400, char_budget // max(1, len(evidence)))
    blocks = []
    for e in evidence:
        body = "\n...\n".join(e.passages) if e.passages else e.snippet
        body = body[:per].replace("</source", "</ source")
        title = e.title.replace('"', "'")
        date = f' published="{e.published}"' if e.published else ""
        blocks.append(f'<source id="{e.n}" url="{e.url}" title="{title}"{date}>\n{body}\n</source>')
    return "\n\n".join(blocks)
