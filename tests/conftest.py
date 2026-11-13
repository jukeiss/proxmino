import asyncio

import pytest

from proxmino import prompts as P
from proxmino.config import ROOT, Config
from proxmino.db import DB
from proxmino.llm import FakeProvider, Router
from proxmino.pipeline import Engine
from proxmino.search import FixtureProvider, SearchResult, SearchService

FIXTURES = [
    SearchResult("https://example.org/a", "Alpha article", "Alpha snippet", "2026-10-01", "fixture",
                 content="The alpha protocol was released in 2026.\nIt improves speed by forty percent over beta."),
    SearchResult("https://news.example.com/b", "Beta report", "Beta snippet", "", "fixture",
                 content="Beta remains widely used.\nAnalysts expect alpha adoption to grow."),
]


def responder(msgs):
    system, user = msgs[0]["content"], msgs[-1]["content"]
    if "web search queries" in system:
        return '{"queries": ["alpha protocol"]}'
    if "extract claims" in system:
        return 'Here you go: {"claims": ["Alpha is faster than beta."]}'
    if "plan research" in system:
        return '```json\n{"questions": ["alpha speed", "beta usage"]}\n```'
    if "review research coverage" in system:
        return '{"gaps": []}'
    if "verify answers" in system:
        return "VERIFIED"
    if "short chat titles" in system:
        return "Alpha protocol questions"
    if "summarize conversations" in system:
        return "Earlier they discussed alpha."
    if "Several specialists" in user:
        return "Merged answer."
    if "<source id=" in user:
        return "Alpha is faster [1] and beta is common [2]. Unknown claim [9]."
    return f"Echo: {user} \U0001F600"


def run(coro):
    return asyncio.run(coro)


async def collect(engine, req):
    return [ev async for ev in engine.turn(req)]


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch, request):
    """Tests never read the real .env or see real keys (and so can never print them)."""
    from proxmino import config, keys
    if "dotenv" not in request.node.name:  # the .env loader's own test uses a temporary file
        monkeypatch.setattr(config, "_load_dotenv", lambda path: None)
    from proxmino import reports
    monkeypatch.setattr(reports, "DEFAULT_VAULT", "")  # tests never write into the real Obsidian vault
    for name in keys.NAMES | {"OLLAMA_URL", "PROXMINO_HOSTS", "PRAXMINO_HOSTS"}:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake():
    return FakeProvider(responder)


@pytest.fixture
def engine(fake, tmp_path):
    cfg = Config(overrides={"product_name": "TestBot", "defaults": {"allow_emoji": False}, "agents_dir": str(tmp_path / "bots"),
                            "limits": {"heavy_agents": ["atlas", "probe", "edge"], "context_chars": 24000}})
    router = Router({"fake": fake}, {t: ["fake:m"] for t in ("fast", "chat", "reason", "code")})
    db = DB(":memory:")
    search = SearchService([FixtureProvider(FIXTURES)], db=db, client=None, fetch_pages=0)
    e = Engine(cfg, db, router, search, P.load_agents(ROOT / "crew"))
    e.mcp.path, e.rules.path = tmp_path / "mcp.json", tmp_path / "approval_rules.json"  # never the real data folder
    return e
