"""Email and X posting: offered only when set up, every send approved, everything logged. No real sends."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxmino import integrations as IN
from proxmino.pipeline import TurnRequest
from proxmino.server import create_app

from conftest import run

X_ENV = {"X_API_KEY": "k", "X_API_SECRET": "s", "X_ACCESS_TOKEN": "t", "X_ACCESS_SECRET": "ts"}
H = {"X-Proxmino": "1"}


@pytest.fixture
def x_env(monkeypatch):
    for k, v in X_ENV.items():
        monkeypatch.setenv(k, v)


def test_post_url_and_pasted_posts_are_parsed():
    assert IN.tweet_id("https://x.com/someone/status/1234567890?s=20") == "1234567890"
    assert IN.tweet_id("https://twitter.com/a/statuses/42") == "42" and IN.tweet_id("99") == "99"
    assert IN.tweet_id("https://example.com/x") is None
    posts = IN.split_posts("https://x.com/a/status/1\nShipping is a habit.\n\nLocal models got good this year.")
    assert posts == [{"post": "Shipping is a habit.", "url": "https://x.com/a/status/1"},
                     {"post": "Local models got good this year.", "url": None}]


def test_replies_lose_greetings_hashtags_and_emojis():
    raw = '"Hey Proxmino fam! Got any tips for a budget launch? \U0001F680 #Proxmino #SaaS"'
    assert IN.clean_reply(raw) == "Got any tips for a budget launch?"
    assert IN.clean_reply("Shipping weekly beats shipping perfect.") == "Shipping weekly beats shipping perfect."


def test_oauth_header_is_signed_and_changes_with_the_secret():
    a = IN.oauth1_header("POST", IN.X_POST_URL, "k", "s", "t", "ts", nonce="n", ts="1")
    b = IN.oauth1_header("POST", IN.X_POST_URL, "k", "other", "t", "ts", nonce="n", ts="1")
    assert a.startswith("OAuth ") and 'oauth_signature_method="HMAC-SHA1"' in a and 'oauth_token="t"' in a
    assert a == IN.oauth1_header("POST", IN.X_POST_URL, "k", "s", "t", "ts", nonce="n", ts="1") and a != b


def test_post_x_sends_a_reply_through_the_official_api(x_env):
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(201, json={"data": {"id": "777", "text": "hi"}})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = run(IN.post_x("Good point.", "https://x.com/a/status/55", client=client))
    assert out.endswith("/777") and seen["url"] == IN.X_POST_URL and seen["auth"].startswith("OAuth ")
    assert seen["body"] == {"text": "Good point.", "reply": {"in_reply_to_tweet_id": "55"}}
    with pytest.raises(ValueError):
        run(IN.post_x("x" * 281, client=client))
    with pytest.raises(ValueError):
        run(IN.post_x("hi", "https://example.com/nope", client=client))
    bad = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403, text="forbidden")))
    with pytest.raises(RuntimeError, match="403"):
        run(IN.post_x("hi", client=bad))


def test_email_is_validated_and_sent_over_starttls(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): sent.append(("tls",))
        def login(self, u, p): sent.append(("login", u))
        def send_message(self, m): sent.append(("msg", m["To"], m["Subject"]))
    monkeypatch.setattr(IN.smtplib, "SMTP", FakeSMTP)
    with pytest.raises(RuntimeError):
        run(IN.send_email("a@b.co", "Hi", "Body"))
    for k, v in {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "me@example.com", "SMTP_PASSWORD": "pw"}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError):
        run(IN.send_email("not-an-address", "Hi", "Body"))
    assert run(IN.send_email("a@b.co, c@d.org", "Hi\nthere", "Body")) == "Sent to a@b.co, c@d.org"
    assert ("tls",) in sent and ("msg", "a@b.co, c@d.org", "Hi there") in sent


async def turn(engine, decision, **kw):
    events = []
    async for k, d in engine.turn(TurnRequest(tools=True, **kw)):
        events.append((k, d))
        if k == "approval" and decision:
            asyncio.get_running_loop().call_soon(engine.resolve_approval, d["id"], decision)
    return events


def test_outward_tools_always_ask_and_log(engine, fake, x_env, monkeypatch):
    calls = []

    async def fake_post(text, reply_to=None, client=None):
        calls.append((text, reply_to))
        return "Posted on X: https://x.com/i/status/1"
    monkeypatch.setattr(IN, "post_x", fake_post)
    names = [t["name"] for t in engine._builtin_catalog()]
    assert "post_x" in names and "send_email" not in names  # email is not set up in this test
    script = iter(['{"tool": "post_x", "args": {"text": "Hello"}}', "Done.", '{"tool": "post_x", "args": {"text": "Again"}}', "Done.",
                   '{"tool": "post_x", "args": {"text": "Routine"}}', "Done."])
    fake.responder = lambda msgs: "A title" if "chat titles" in msgs[0]["content"] else next(script)
    first = run(turn(engine, "always", message="post hello"))
    assert [d["tool"] for k, d in first if k == "approval"] == ["post_x"] and calls == [("Hello", None)]
    second = run(turn(engine, "deny", message="post again"))  # "always" did not stick for outward tools
    assert [d["tool"] for k, d in second if k == "approval"] == ["post_x"] and len(calls) == 1
    third = run(collect_unattended(engine))
    assert not [1 for k, _ in third if k == "approval"] and len(calls) == 1
    log = engine.db.outbox()
    assert [(r["kind"], r["status"], r["text"]) for r in log] == [("post_x", "sent", "Hello")]


async def collect_unattended(engine):
    return [ev async for ev in engine.turn(TurnRequest(tools=True, unattended=True, message="post routine"))]


def test_reply_desk_drafts_one_reply_per_post(engine, fake):
    fake.responder = lambda msgs: '"Have you tried running it on a quantized 3B model first?"' if "<post>" in msgs[-1]["content"] else "t"
    with TestClient(create_app(engine, run_scheduler=False), base_url="http://127.0.0.1") as c:
        r = c.post("/api/replies/draft", json={"text": "https://x.com/a/status/9\nLocal LLMs are too slow.\n\nWhat do you use for evals?",
                                               "platform": "x", "goal": "find beta users", "product": "Proxmino"}, headers=H).json()
        assert len(r) == 2 and r[0]["url"] == "https://x.com/a/status/9" and r[0]["draft"].startswith("Have you tried")
        prompt = fake.calls[-1][-1]["content"]
        assert "Proxmino only if it truly answers" in prompt and "At most 240 characters" in prompt
        assert c.post("/api/replies/draft", json={"text": " ", "platform": "x"}, headers=H).status_code == 400
        assert c.post("/api/replies/draft", json={"text": "hi", "platform": "myspace"}, headers=H).status_code == 400
        assert c.post("/api/replies/post", json={"text": "hi"}, headers=H).status_code == 400  # X not set up
        st = c.get("/api/integrations").json()
        assert st["x"] is False and st["x_posted_24h"] == 0
