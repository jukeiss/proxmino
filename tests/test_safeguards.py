from proxmino import inbox as INBOX
from proxmino.pipeline import warden_verdict


def test_warden_reads_its_verdict_fairly():
    assert warden_verdict("VERIFIED") == ("VERIFIED", "")
    assert warden_verdict("verified.") == ("VERIFIED", "")
    assert warden_verdict("**VERIFIED**") == ("VERIFIED", "")
    assert warden_verdict("MINOR") == ("MINOR", "")
    v, why = warden_verdict("PROBLEM\n- [2] says 2019, the answer says 2021")
    assert v == "PROBLEM" and "2019" in why
    assert warden_verdict("PROBLEM") == ("MINOR", "")  # nothing named: no warning
    assert warden_verdict("All claims look supported by the sources.")[0] is None  # off-format: stay quiet


def test_scam_check_needs_an_actual_ask():
    flag = lambda t: bool(INBOX.SCAM.search(t))
    # normal messages that used to be flagged
    assert not flag("I reset my password yesterday, all good now.")
    assert not flag("Thanks so much for the gift card!")
    assert not flag("The wire transfer from the client arrived on Monday.")
    assert not flag("Can you share the slides from the meeting?")
    # real scam patterns
    assert flag("Please send me the verification code you just received.")
    assert flag("Reply with your password to keep your account active")
    assert flag("I need you to buy 5 gift cards and send me the codes")
    assert flag("Share your seed phrase to claim the airdrop")
    assert flag("Pay the invoice in bitcoin today")


def test_confidence_badge_is_not_a_false_alarm():
    from proxmino.safety import confidence
    assert confidence([1], {1: "en.wikipedia.org"}) == "high"  # a trusted reference on its own
    assert confidence([1, 2], {1: "a.com", 2: "b.org"}) == "high"
    assert confidence([1], {1: "someblog.com"}) == "medium"
    assert confidence([], {1: "a.com"}) is None  # nothing cited: no badge at all


def test_memory_keeps_ordinary_facts():
    from proxmino.pipeline import SENSITIVE
    assert not SENSITIVE.search("works at a bank")
    assert not SENSITIVE.search("writes code in Python")
    assert not SENSITIVE.search("lives in Paris")
    assert SENSITIVE.search("password is hunter2") and SENSITIVE.search("card number 4111")


def test_uncensored_tone_is_opt_in():
    from proxmino import prompts as P
    off = P.system_prompt("X", "uncensored", adult=False)
    on = P.system_prompt("X", "uncensored", adult=True)
    assert "uncensored" not in off.lower() and P.PERSONAS["blunt"] in off  # 18+ off: falls back to blunt
    assert "swearing" in on and "Safety:" in on  # the hard limits are still in the prompt
    assert "swearing" not in P.system_prompt("X", "professional", adult=True)  # other tones unchanged


def test_social_requests_get_the_tools_and_threads_split():
    from proxmino.pipeline import SOCIAL_CUE
    from proxmino import integrations as IN
    for t in ["can you post stuff on my twitter for me", "tweet this: we launched", "post a thread on X about AI agents",
              "reply on reddit to that comment", "send an email to Sam about Friday"]:
        assert SOCIAL_CUE.search(t), t
    for t in ["what is x squared plus 2", "how does twitter's algorithm work", "write a poem"]:
        assert not SOCIAL_CUE.search(t), t
    long = " ".join(f"Sentence number {i} about agents." for i in range(40))
    parts = IN.split_thread(long)
    assert len(parts) > 1 and all(len(p) <= 280 for p in parts) and parts[0].endswith(f"1/{len(parts)}")
    assert IN.split_thread("short one") == ["short one"]


def test_thread_posts_chain_as_replies(monkeypatch):
    import asyncio
    from proxmino import integrations as IN
    calls = []

    async def fake_post(text, reply_to=None, client=None):
        calls.append(reply_to)
        return f"Posted on X: https://x.com/i/status/{100 + len(calls)}"

    monkeypatch.setattr(IN, "post_x", fake_post)
    out = asyncio.run(IN.post_x_thread(["one", "two", "three"]))
    assert calls == [None, "101", "102"] and "3-post thread" in out and "status/101" in out


def test_old_praxis_names_still_work(tmp_path, monkeypatch):
    from proxmino import config
    (tmp_path / "praxis.db").write_text("db")
    (tmp_path / "praxis.db-wal").write_text("wal")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home" / ".praxis" / "agents").mkdir(parents=True)
    config.migrate_old_names()
    assert (tmp_path / "proxmino.db").read_text() == "db" and (tmp_path / "proxmino.db-wal").read_text() == "wal"
    assert not (tmp_path / "praxis.db").exists() and (tmp_path / "home" / ".proxmino" / "agents").is_dir()
    config.migrate_old_names()  # running again changes nothing
    assert (tmp_path / "proxmino.db").exists()


def test_thread_keeps_the_writers_own_numbering():
    from proxmino import integrations as IN
    parts = IN.split_thread("1/ First point here.\n\n2/ Second point.\n\n3/ Third.")
    assert parts == ["1/ First point here.", "2/ Second point.", "3/ Third."]


def test_reasoning_that_mentions_the_tag_stays_hidden():
    import asyncio, json
    import httpx
    from proxmino.llm import OpenAICompatProvider, Params, ThinkSplitter

    chunks = [{"reasoning_content": "plan: no need to print the </think> tag, final only"}, {"content": "The answer is 42."}]
    body = "".join(f"data: {json.dumps({'choices': [{'delta': c}]})}\n\n" for c in chunks) + "data: [DONE]\n\n"
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, text=body)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            prov = OpenAICompatProvider("deepseek", "https://api.deepseek.com", "k", False, c)
            sp, out = ThinkSplitter(), []
            async for piece in prov.stream([{"role": "user", "content": "q"}], "deepseek-flash", Params(think=True)):
                out += sp.feed(piece)
            return out

    parts = asyncio.run(go())
    answer = "".join(t for k, t in parts if k == "token")
    assert answer == "The answer is 42." and seen["thinking"] == {"type": "enabled"}


def test_brain_setting_puts_your_model_first():
    from proxmino.config import Config
    from proxmino.llm import Router
    cfg = Config(overrides={"defaults": {"brain": "proxmino", "brain_model": "ollama:proxmino:8b"},
                            "tiers": {"chat": ["deepseek:deepseek-flash", "ollama:phi4-mini"], "reason": ["deepseek:deepseek-flash"]}})
    r = Router.from_config(cfg, None)
    assert r.tiers["chat"][0] == "ollama:proxmino:8b" and r.tiers["reason"][0] == "ollama:proxmino:8b"
    assert Router.from_config(Config(overrides={"tiers": {"chat": ["deepseek:deepseek-flash"]}}), None).tiers["chat"][0] == "deepseek:deepseek-flash"


def test_searxng_url_from_environment(monkeypatch):
    """The server packages point the app at their own SearXNG through PROXMINO_SEARXNG_URL."""
    from proxmino.search import SearchService, SearXNGProvider
    monkeypatch.setenv("PROXMINO_SEARXNG_URL", "http://searxng:8080")
    cfg = {"search.providers": ["brave", "tavily", "searxng"], "search.wikipedia": False}
    svc = SearchService.from_config(type("C", (), {"get": lambda self, k, d=None: cfg.get(k, d)})(), None, object())
    assert any(isinstance(p, SearXNGProvider) and p.base == "http://searxng:8080" for p in svc.providers)


def test_hybrid_brain_routes_hard_work_to_the_cloud():
    from proxmino.llm import Router
    cfg = {"providers": {}, "tiers": {"chat": ["deepseek:deepseek-flash", "ollama:phi4-mini"],
                                      "reason": ["deepseek:deepseek-flash", "ollama:phi4-mini"]},
           "defaults.brain": "hybrid", "defaults.brain_model": "ollama:proxmino:v1"}
    r = Router.from_config(type("C", (), {"get": lambda self, k, d=None: cfg.get(k, d)})(), None)
    assert r.tiers["chat"][0] == "deepseek:deepseek-flash" and r.tiers["chat"][-1] == "ollama:proxmino:v1"
    assert r.tiers["reason"][0] == "deepseek:deepseek-flash" and r.tiers["reason"][-1] == "ollama:proxmino:v1"


def test_writing_requests_get_the_no_invented_details_note():
    from proxmino.pipeline import NO_INVENTED_DETAILS, WRITE_FOR_ME
    assert WRITE_FOR_ME.search("write a tweet about my 10k") and WRITE_FOR_ME.search("email my teacher")
    assert not WRITE_FOR_ME.search("what's a VPN?")
    assert "[placeholder]" in NO_INVENTED_DETAILS


def test_discord_webhook_posts_without_a_bot(monkeypatch):
    import asyncio, httpx
    from proxmino import inbox as INBOX
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://evil.example.com/api/webhooks/1/abc")
    assert not INBOX.discord_can_send()  # only Discord's own webhook addresses
    url = "https://discord.com/api/webhooks/123456/AbC-d_e"
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
    assert INBOX.discord_can_send()
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), body=req.content)
        return httpx.Response(200, json={"id": "1"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await INBOX.send_discord_webhook(c, "game night at 8 @everyone")
    assert "Posted" in asyncio.run(go())
    assert seen["url"].startswith(url) and b'"parse": []' in seen["body"].replace(b'"parse":[]', b'"parse": []')
