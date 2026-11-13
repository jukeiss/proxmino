import asyncio
import json

import httpx
import pytest

from proxmino import connections as CONN
from proxmino import reddit as R


def test_target_parsing():
    assert R.target("r/python") == ("sub", "python")
    assert R.target("/r/Ask_Science/") == ("sub", "Ask_Science")
    assert R.target("https://www.reddit.com/r/python/comments/abc123/some_title/") == ("thing", "t3_abc123")
    assert R.target("https://old.reddit.com/r/python/comments/abc123/some_title/def456/") == ("thing", "t1_def456")
    assert R.target("t1_xyz9") == ("thing", "t1_xyz9")
    for bad in ("", "python", "https://example.com/r/python/comments/abc", "r/a"):
        with pytest.raises(ValueError):
            R.target(bad)


def _env(monkeypatch):
    for k, v in zip(R.KEYS, ("id", "secret", "me_user", "pw")):
        monkeypatch.setenv(k, v)
    R._token.clear()


def test_post_reply_and_new_post(monkeypatch):
    _env(monkeypatch)
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, str(req.url), req.headers.get("user-agent"), req.content.decode()))
        if req.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        if req.url.path == "/api/comment":
            return httpx.Response(200, json={"json": {"errors": [], "data": {"things": [{"data": {"permalink": "/r/x/comments/a/b/c/"}}]}}})
        if req.url.path == "/api/submit":
            return httpx.Response(200, json={"json": {"errors": [], "data": {"url": "https://www.reddit.com/r/x/comments/new/"}}})
        return httpx.Response(404)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            a = await R.post(c, "Useful reply", "https://www.reddit.com/r/x/comments/a/b/c/")
            b = await R.post(c, "Body", "r/xyz", "A title")
            with pytest.raises(ValueError):
                await R.post(c, "Body", "r/xyz")  # a new post needs a title
            return a, b

    a, b = asyncio.run(go())
    assert a == "Posted on Reddit: https://www.reddit.com/r/x/comments/a/b/c/"
    assert b.endswith("/r/x/comments/new/")
    assert sum(1 for s in seen if "access_token" in s[1]) == 1  # the token is reused
    assert all("by /u/me_user" in s[2] for s in seen)
    assert "thing_id=t1_c" in seen[1][3]


def test_refused_post_raises(monkeypatch):
    _env(monkeypatch)

    def handler(req):
        if req.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        return httpx.Response(200, json={"json": {"errors": [["RATELIMIT", "slow down", "ratelimit"]]}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            await R.post(c, "hi", "t3_abc")

    with pytest.raises(RuntimeError, match="RATELIMIT"):
        asyncio.run(go())


def test_fetch_inbox(monkeypatch):
    _env(monkeypatch)

    def handler(req):
        if req.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert req.url.params["mark"] == "false"  # never marks messages read
        return httpx.Response(200, json={"data": {"children": [
            {"data": {"name": "t1_q1", "body": "Thanks, but how?", "author": "sam", "link_title": "My post",
                      "link_id": "t3_p1", "context": "/r/x/comments/p1/t/q1/?context=3"}},
            {"data": {"name": "t4_m1", "body": "", "author": "x"}}]}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await R.fetch(c)

    items = asyncio.run(go())
    assert len(items) == 1 and items[0]["source"] == "reddit" and items[0]["meta"]["thing_id"] == "t1_q1"
    assert items[0]["sender"] == "u/sam"


def test_connection_test_without_keys():
    r = asyncio.run(CONN.test_all(None))
    assert set(r) == {"Email (sending)", "Email (reading)", "Discord (webhook)", "Discord (bot)", "Reddit", "X (Twitter)"}
    assert all(v["ok"] is None for v in r.values())


def test_connection_test_reports_failure(monkeypatch):
    _env(monkeypatch)

    def handler(req):
        return httpx.Response(401, json={"error": "invalid_grant"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await CONN.test_all(c)

    r = asyncio.run(go())
    assert r["Reddit"]["ok"] is False and "invalid_grant" in r["Reddit"]["detail"]
    assert "pw" not in json.dumps(r) and "secret" not in json.dumps(r)


def test_send_policy(engine):
    st = type("S", (), {"read_outside": False, "sources": []})()
    assert engine.send_policy(st)[0] == "ask"  # off by default
    engine.cfg.data.setdefault("defaults", {})["send_without_asking"] = True
    assert engine.send_policy(st)[0] == "allow"
    st.sources = ["a web page"]
    assert engine.send_policy(st)[0] == "ask"  # read outside text: could be prompt injection
    st.sources, st.read_outside = [], True
    assert engine.send_policy(st)[0] == "ask"
    engine.cfg.data["defaults"]["send_limit_per_day"] = 1
    engine.db.outbox_add("post_x", "", "hi", "sent", "ok")
    assert engine.send_policy()[0] == "ask"  # daily limit reached


def test_auto_reply_skips_automated_mail(engine):
    from proxmino import inbox as INBOX
    engine.cfg.data.setdefault("defaults", {}).update(send_without_asking=True, auto_reply_inbox=True)
    t = {"needs_reply": True, "draft": "Thanks!", "priority": "normal", "summary": "asks a question"}
    person = {"meta": {"automated": False}}
    assert INBOX.auto_reply_ok(engine, t, person)
    assert not INBOX.auto_reply_ok(engine, t, {"meta": {"automated": True}})  # newsletters, lists, auto-replies
    assert not INBOX.auto_reply_ok(engine, {**t, "summary": "Possible scam: asks for a code"}, person)
    engine.cfg.data["defaults"]["auto_reply_inbox"] = False
    assert not INBOX.auto_reply_ok(engine, t, person)
    assert INBOX.NO_REPLY.search("no-reply@shop.example") and not INBOX.NO_REPLY.search("sam@example.org")
