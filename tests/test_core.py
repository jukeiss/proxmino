import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import collect, run
from proxmino import prompts as P
from proxmino.llm import FakeProvider, Params, ProviderError, Router, ThinkSplitter, parse_json
from proxmino.pipeline import TurnRequest
from proxmino.safety import strip_emoji, verify_citations
from proxmino.sandbox import run_python
from proxmino.search import chunk, rank
from proxmino.server import create_app


def kinds(events, kind):
    return [d for k, d in events if k == kind]


def text_of(events):
    return "".join(d["t"] for d in kinds(events, "token"))


# ---------------- units ----------------

def test_think_splitter_handles_split_tags():
    s, out = ThinkSplitter(), []
    for chunk_ in ["Hi <th", "ink>plan a", "nd solve</thi", "nk>Answer"]:
        out += s.feed(chunk_)
    out += s.flush()
    joined = {}
    for k, t in out:
        joined[k] = joined.get(k, "") + t
    assert joined == {"token": "Hi Answer", "thinking": "plan and solve"}


def test_parse_json_from_prose_and_fences():
    assert parse_json('Sure! {"a": [1]} done', {}) == {"a": [1]}
    assert parse_json('```json\n["x"]\n```', []) == ["x"]
    assert parse_json("<think>{bad}</think>no json", {"d": 1}) == {"d": 1}


def test_emoji_filter_and_citation_verifier():
    assert strip_emoji("Done \U0001F680✅ ok") == "Done  ok"
    cleaned, used, invalid = verify_citations("A [1] B [3] C [1]", {1, 2})
    assert cleaned == "A [1] B  C [1]" and used == [1] and invalid == [3]
    text = "See [1][2] and [7]. Code: `arr[7]`, x[7], and\n```py\nprint(a[7])\n```"
    cleaned, used, invalid = verify_citations(text, {1, 2})
    assert used == [1, 2] and invalid == [7]
    assert "`arr[7]`" in cleaned and "x[7]" in cleaned and "a[7]" in cleaned and "and ." in cleaned


def test_bm25_rank_prefers_relevant_chunk():
    passages = chunk("Cats sleep a lot.\n" * 3 + "\n" + "Rust ownership and borrowing rules explained.")
    best = rank("rust borrowing", passages, 1)
    assert "borrowing" in passages[best[0]]


def test_agent_cards_load_and_validate(tmp_path):
    agents = P.load_agents(Path(__file__).parent.parent / "crew")
    assert {"nexus", "forge", "warden", "rogue"} <= set(agents)
    assert "Forge" in agents["forge"].persona_block()
    (tmp_path / "bad.yaml").write_text("name: Bad\nshell: rm -rf /\n")
    (tmp_path / "good.yaml").write_text("name: Auditor\nrole: auditor\ntraits: {bluntness: 0.9}\n")
    custom = P.load_agents(tmp_path)
    assert "good" in custom and "bad" not in custom and not custom["good"].builtin


# ---------------- router ----------------

class Broken(FakeProvider):
    name = "broken"

    async def stream(self, messages, model, p):
        raise ProviderError("broken: ConnectError")
        yield  # pragma: no cover


def test_router_falls_back_then_demo():
    good = FakeProvider(lambda m: "fine")
    good.name = "good"
    r = Router({"broken": Broken(), "good": good}, {"chat": ["broken:x", "good:y"]})
    label, text = run(_complete(r))
    assert label == "good:y" and text == "fine"
    r2 = Router({"broken": Broken()}, {"chat": ["broken:x"]})
    label, text = run(_complete(r2))
    assert label == "demo" and "No language model is reachable" in text


def test_router_privacy_gate_skips_cloud():
    cloud = FakeProvider(lambda m: "cloud")
    cloud.name, cloud.local = "cloud", False
    r = Router({"cloud": cloud}, {"chat": ["cloud:x"]}, allow_cloud=False)
    assert r.chain("chat") == []
    r.allow_cloud = True
    assert len(r.chain("chat")) == 1


async def _complete(router):
    label, it = await router.open("chat", [{"role": "user", "content": "hi"}], Params())
    return label, "".join([c async for c in it])


# ---------------- pipeline ----------------

def test_chat_streams_persists_and_strips_emoji(engine):
    ev = run(collect(engine, TurnRequest(message="hello")))
    meta, done = kinds(ev, "meta")[0], kinds(ev, "done")[0]
    assert text_of(ev).startswith("Echo: hello") and "\U0001F600" not in done["content"]
    msgs = engine.db.messages(meta["chat_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["parent_id"] == msgs[0]["id"] and msgs[1]["meta"]["model"] == "fake:m"


def test_search_cites_only_real_sources(engine):
    ev = run(collect(engine, TurnRequest(message="tell me about alpha", search=True)))
    done = kinds(ev, "done")[0]
    assert len(kinds(ev, "source")) == 2
    assert "[9]" not in done["content"] and done["used_sources"] == [1, 2]
    assert done["confidence"] == "high"  # two distinct domains
    assert not any("citation" in s["text"] for s in kinds(ev, "step"))  # cleaned quietly, no warning chip


def test_auto_routing(engine):
    st = engine.parse(TurnRequest(), "what is the latest news about alpha")
    assert st.route == "search"
    assert engine.parse(TurnRequest(), "https://example.org/x").route == "tldr"
    assert engine.parse(TurnRequest(), "is it true that alpha is faster?").route == "check"
    assert engine.parse(TurnRequest(), "write a haiku").route == "chat"
    # Auto mode: make -> tools, maths/logic -> think, a named fact -> search (only when search is set up)
    st = engine.parse(TurnRequest(), "make a chart of churn by reason and tenure")
    assert (st.route, st.tools) == ("tools", True) and st.auto
    assert engine.parse(TurnRequest(), "what is 17 * 23").route == "think"
    assert engine.parse(TurnRequest(), "why does my code crash").route == "think"
    assert engine.parse(TurnRequest(), "Who is the CEO of Anthropic?").route == "search"
    assert engine.parse(TurnRequest(), "What can Forge do?").route == "chat"
    assert engine.parse(TurnRequest(), "what is photosynthesis").route == "chat"
    assert engine.parse(TurnRequest(), "Who are you?").route == "chat"
    providers, engine.search.providers = engine.search.providers, []
    assert engine.parse(TurnRequest(), "Who founded Anthropic?").route == "chat"
    engine.search.providers = providers
    st = engine.parse(TurnRequest(), "@forge write a parser")
    assert st.agent.id == "forge" and st.text == "write a parser"
    st = engine.parse(TurnRequest(), "/blunt rate my plan")
    assert st.persona == "blunt" and st.route == "chat"


def test_memory_commands_and_injection(engine, fake):
    run(collect(engine, TurnRequest(message="/remember I prefer metric units")))
    ev = run(collect(engine, TurnRequest(message="/memory")))
    assert "metric units" in text_of(ev)
    run(collect(engine, TurnRequest(message="how tall is it")))
    assert "metric units" in fake.calls[-1][0]["content"] or any(
        "metric units" in c[0]["content"] for c in fake.calls)
    run(collect(engine, TurnRequest(message="/forget")))
    assert engine.db.memories() == []


def test_regenerate_and_branching(engine):
    ev = run(collect(engine, TurnRequest(message="first")))
    meta = kinds(ev, "meta")[0]
    ev2 = run(collect(engine, TurnRequest(regenerate=meta["assistant_id"])))
    meta2 = kinds(ev2, "meta")[0]
    a1, a2 = engine.db.message(meta["assistant_id"]), engine.db.message(meta2["assistant_id"])
    assert a1["parent_id"] == a2["parent_id"] == meta["user_id"] and a1["id"] != a2["id"]
    # follow-up on branch sees only its own path
    ev3 = run(collect(engine, TurnRequest(message="second", chat_id=meta["chat_id"], parent_id=a2["id"])))
    path = engine.db.path_to(kinds(ev3, "meta")[0]["assistant_id"])
    assert [m["content"].split(":")[0] for m in path][:1] == ["first"] and len(path) == 4


def test_factcheck_and_deepsearch(engine):
    ev = run(collect(engine, TurnRequest(message="/check Alpha is faster than beta")))
    steps = " ".join(s["text"] for s in kinds(ev, "step"))
    assert "Extracted 1 claim" in steps and "Checked against sources: supported" in steps
    ev = run(collect(engine, TurnRequest(message="/deep how good is alpha")))
    steps = [s for s in kinds(ev, "step")]
    assert steps[0]["agent"] == "Atlas" and "alpha speed" in steps[0]["text"]
    assert kinds(ev, "done")[0]["verified"] is True


def test_heavy_mode_merges_three_drafts(engine):
    ev = run(collect(engine, TurnRequest(message="design a cache", heavy=True)))
    drafted = {s["agent"] for s in kinds(ev, "step") if "finished a draft" in s["text"]}
    assert drafted == {"Atlas", "Probe", "Edge"}
    assert text_of(ev) == "Merged answer."


def test_history_trim_and_background_summary(engine):
    engine.cfg.data["limits"]["context_chars"] = 60
    meta = None
    for i in range(4):
        ev = run(collect(engine, TurnRequest(message=f"message number {i} " + "x" * 30,
                                              chat_id=meta and meta["chat_id"],
                                              parent_id=meta and meta["assistant_id"])))
        meta = kinds(ev, "meta")[0]
    history = engine.db.path_to(meta["assistant_id"])
    assert len(engine._history(history)) < len(history)


def test_help_and_unknown_command(engine):
    assert "/deep" in text_of(run(collect(engine, TurnRequest(message="/help"))))
    assert "Unknown command" in text_of(run(collect(engine, TurnRequest(message="/nope"))))


# ---------------- sandbox ----------------

def test_sandbox_runs_and_limits():
    r = run(run_python("print(sum(range(10)))", 5))
    assert r.stdout.strip() == "45" and r.exit_code == 0
    r = run(run_python("while True: pass", 1))
    assert r.timed_out or r.exit_code != 0


@pytest.mark.skipif(sys.platform != "darwin", reason="network sandbox is macOS only")
def test_sandbox_blocks_network_and_home():
    r = run(run_python("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)\nprint('connected')", 5))
    assert "connected" not in r.stdout and r.sandboxed
    r = run(run_python("import os\nprint(os.listdir('/Users'))\n"
                       "open('/Users/Shared/proxmino-test.txt','w')", 5))
    assert r.exit_code != 0


# ---------------- server ----------------

def test_server_security_and_sse(engine):
    with TestClient(create_app(engine)) as c:
        assert "TestBot" in c.get("/").text
        assert c.post("/api/memory", json={"text": "x"}).status_code == 403  # no X-Praxis header
        assert c.get("/api/chats", headers={"host": "evil.example"}).status_code == 403
        h = {"X-Proxmino": "1"}
        r = c.post("/api/chat", json={"message": "hi there"}, headers=h)
        events = [b.split("\n") for b in r.text.strip().split("\n\n")]
        names = [e[0].removeprefix("event: ") for e in events]
        assert names[0] == "meta" and names[-1] == "done" and "token" in names
        chat_id = json.loads(events[0][1].removeprefix("data: "))["chat_id"]
        assert c.get(f"/api/chats/{chat_id}").json()["messages"][0]["content"] == "hi there"
        assert c.post("/api/run", json={"code": "print(2+2)"}, headers=h).json()["stdout"].strip() == "4"
        assert len(c.get("/api/agents").json()) >= 9


def test_about_block_lists_crew_and_modes():
    from proxmino import prompts as P
    from proxmino.config import ROOT
    agents = P.load_agents(ROOT / "crew")
    about = P.about_block("Proxmino", agents, {"search": "", "deep": "", "professional": ""}, ["atlas", "probe"])
    for name in ("Nexus", "Atlas", "Forge", "Warden", "Scout"):
        assert name in about
    assert "/search" in about and "/professional" not in about.split("Tones:")[0]
    assert "Atlas, Probe draft independently" in about
    sp = P.system_prompt("Proxmino", about=about)
    assert sp.index("About Proxmino") > sp.index("No emojis")


def test_usage_counts_answers_per_model(engine):
    run(collect(engine, TurnRequest(message="hello")))
    rows = engine.db.usage()
    assert rows and rows[0]["answers"] >= 1 and rows[0]["model"]


def test_ollama_retries_without_think_when_model_rejects_it():
    import httpx
    from proxmino.llm import OllamaProvider
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if "think" in body:
            return httpx.Response(400, text='{"error":"model does not support thinking"}')
        return httpx.Response(200, text='{"message":{"content":"0.05"},"done":true}\n')

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            p = OllamaProvider("ollama", "http://x", c)
            return "".join([t async for t in p.stream([{"role": "user", "content": "q"}], "m", Params(think=True))])

    assert run(go()) == "0.05"
    assert "think" in calls[0] and "think" not in calls[1]


def test_think_recovers_answer_when_think_tag_is_never_closed(engine, fake):
    fake.responder = lambda msgs: "<think>Let B be the ball.\n\n2B = 0.10\n\nSo, the ball costs 0.05."
    evs = run(collect(engine, TurnRequest(message="bat and ball?", think=True)))
    done = [d for k, d in evs if k == "done"][0]
    assert done["content"] == "So, the ball costs 0.05."


def test_local_and_cloud_are_separate_opt_ins():
    from proxmino.llm import Router
    local, cloud = FakeProvider(), FakeProvider()
    cloud.name, cloud.local = "cloudy", False
    tiers = {"chat": ["fake:a", "cloudy:b"]}
    r = Router({"fake": local, "cloudy": cloud}, tiers, allow_cloud=False, allow_local=False)
    assert r.chain("chat") == []
    r = Router({"fake": local, "cloudy": cloud}, tiers, allow_cloud=True, allow_local=False)
    assert [p.name for p, _ in r.chain("chat")] == ["cloudy"]
    r = Router({"fake": local, "cloudy": cloud}, tiers, allow_cloud=False, allow_local=True)
    assert [p.name for p, _ in r.chain("chat")] == ["fake"]


def test_disabled_local_is_not_used_as_last_resort():
    from proxmino.llm import Router
    r = Router({"fake": FakeProvider()}, {"chat": []}, allow_local=False, demo_fallback=True)
    model, _ = run(r.open("chat", [{"role": "user", "content": "hi"}]))
    assert model == "demo"


def test_dotenv_reload_adds_and_clears_keys(tmp_path, monkeypatch):
    import os
    from proxmino import config as C
    env = tmp_path / ".env"
    monkeypatch.delenv("PROXMINO_TEST_KEY", raising=False)
    env.write_text("PROXMINO_TEST_KEY=abc\n")
    C._load_dotenv(env)
    assert os.environ["PROXMINO_TEST_KEY"] == "abc"
    env.write_text("PROXMINO_TEST_KEY=\n")
    C._load_dotenv(env)
    assert "PROXMINO_TEST_KEY" not in os.environ


def test_health_reports_cloud_keys_without_values(engine):
    from fastapi.testclient import TestClient as TC
    from proxmino.server import create_app
    app = create_app(engine)
    with TC(app, base_url="http://127.0.0.1") as c:
        h = c.get("/api/health").json()
    assert "ready" in h and "allow_local" in h and isinstance(h["cloud"], list)
    for item in h["cloud"]:
        assert set(item) == {"name", "key_env", "has_key"}


def test_chat_model_is_pinned_and_remembered(engine, fake):
    other = FakeProvider(lambda msgs: "from other")
    other.name = "other"
    engine.router.providers["other"] = other
    evs = run(collect(engine, TurnRequest(message="hi", model="other:big")))
    done = [d for k, d in evs if k == "done"][0]
    chat_id = [d for k, d in evs if k == "meta"][0]["chat_id"]
    assert done["model"] == "other:big" and done["content"] == "from other"
    assert engine.db.chat(chat_id)["model"] == "other:big"
    # later turns in the same chat use it without being told again
    evs = run(collect(engine, TurnRequest(message="again", chat_id=chat_id)))
    assert [d for k, d in evs if k == "done"][0]["model"] == "other:big"
    # an empty string clears the pin
    evs = run(collect(engine, TurnRequest(message="back", chat_id=chat_id, model="")))
    assert [d for k, d in evs if k == "done"][0]["model"] == "fake:m"


def test_pinned_router_falls_back_when_pin_is_unavailable(engine):
    r = engine.router.pinned("missing:x")
    assert [f"{p.name}:{m}" for p, m in r.chain("chat")] == ["fake:m"]
    assert engine.router.tiers["chat"] == ["fake:m"]  # the original is untouched


def test_token_usage_is_recorded_from_provider(engine, fake):
    class Counting(FakeProvider):
        async def stream(self, messages, model, p):
            async for t in super().stream(messages, model, p):
                yield t
            if p.usage is not None:
                p.usage.update({"in": 12, "out": 5})
    c = Counting(lambda m: "hello there")
    c.name = "fake"
    engine.router.providers["fake"] = c
    evs = run(collect(engine, TurnRequest(message="hi")))
    done = [d for k, d in evs if k == "done"][0]
    assert done["tokens_in"] == 12 and done["tokens_out"] == 5
    row = [r for r in engine.db.usage() if r["model"] == "fake:m"][0]
    assert row["tokens_in"] >= 12 and row["tokens_out"] >= 5


def test_openai_compat_reads_usage_chunk():
    import httpx
    from proxmino.llm import OpenAICompatProvider
    sse = ('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
           'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\ndata: [DONE]\n\n')
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, text=sse)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            p = OpenAICompatProvider("cloud", "http://x/v1", "k", False, c)
            prm = Params(usage={})
            text = "".join([t async for t in p.stream([{"role": "user", "content": "q"}], "m", prm)])
            return text, prm.usage

    text, usage = run(go())
    assert text == "hi" and usage == {"in": 7, "out": 2}
    assert sent[0]["stream_options"] == {"include_usage": True}


def test_memory_is_shared_or_per_agent(engine, fake):
    run(collect(engine, TurnRequest(message="/remember I prefer metric units")))
    run(collect(engine, TurnRequest(message="/remember @forge Always add type hints")))
    seen = {}

    def responder(msgs):
        seen.setdefault("prompts", []).append(msgs[0]["content"])
        return "ok"
    fake.responder = responder
    run(collect(engine, TurnRequest(message="hello")))
    run(collect(engine, TurnRequest(message="@forge write a function")))
    plain, forge = [p for p in seen["prompts"] if "Follow these rules" in p]  # skip chat-title prompts
    assert "metric units" in plain and "type hints" not in plain
    assert "metric units" in forge and "type hints" in forge
    mems = engine.db.memories()
    assert [m["agent"] for m in mems] == [None, "forge"]


def test_ollama_cloud_models_need_cloud_switched_on(fake):
    from proxmino.llm import Router, is_cloud_model
    assert is_cloud_model("deepseek-v4.1-flash:cloud") and is_cloud_model("deepseek-v4-pro:0813-cloud")
    assert not is_cloud_model("phi4-mini")
    fake.local = True
    tiers = {"chat": ["fake:deepseek-v4.1-flash:cloud", "fake:phi4-mini"]}
    off = Router({"fake": fake}, tiers, allow_cloud=False, allow_local=True)
    on = Router({"fake": fake}, tiers, allow_cloud=True, allow_local=True)
    assert [m for _, m in off.chain("chat")] == ["phi4-mini"]
    assert [m for _, m in on.chain("chat")] == ["deepseek-v4.1-flash:cloud", "phi4-mini"]


def test_ollama_cloud_provider_sends_the_key_and_counts_as_cloud(monkeypatch):
    import httpx
    from proxmino.config import Config
    from proxmino.llm import Params, Router
    seen = {}

    def handler(req):
        seen["auth"], seen["url"] = req.headers.get("authorization"), str(req.url)
        return httpx.Response(200, text='{"message": {"content": "hi"}, "done": true}\n')
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tiers = {"chat": ["ollama_cloud:deepseek-v4.1-flash"]}
    cfg = Config(overrides={"defaults": {"allow_cloud": True}, "tiers": tiers})
    off_cfg = Config(overrides={"defaults": {"allow_cloud": False}, "tiers": tiers})
    assert "ollama_cloud" not in Router.from_config(cfg, client).providers  # no key: skipped
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-test-key")
    r = Router.from_config(cfg, client)
    prov = r.providers["ollama_cloud"]
    assert prov.local is False and prov.base == "https://ollama.com"
    out = run(r.complete("chat", [{"role": "user", "content": "x"}], Params()))
    assert out == "hi" and seen["auth"] == "Bearer secret-test-key" and seen["url"] == "https://ollama.com/api/chat"
    assert prov.num_ctx == 8192
    off = Router.from_config(off_cfg, client)
    assert all(p.name != "ollama_cloud" for p, _ in off.chain("chat"))


def test_auto_routes_real_questions_to_sources_when_search_first(engine):
    engine.cfg.data.setdefault("defaults", {})["search_first"] = True
    assert engine.parse(TurnRequest(), "what are the best budget headphones?").route == "search"  # products, picks
    assert engine.parse(TurnRequest(), "is it worth upgrading from an iphone 13?").route == "search"
    assert engine.parse(TurnRequest(), "who is the governor of Lagos?").route == "search"  # a named place
    assert engine.parse(TurnRequest(), "Explain photosynthesis for a 10 year old").route == "chat"  # a concept: answered directly
    assert engine.parse(TurnRequest(), "what's 17% of 2,340").route == "chat"
    assert engine.parse(TurnRequest(), "write a poem about rain?").route == "chat"
    assert engine.parse(TurnRequest(), "what can you do?").route == "chat"  # about the assistant itself
    assert engine.parse(TurnRequest(), "hi").route == "chat"
    assert engine.parse(TurnRequest(study=True), "photosynthesis").route == "study"
    engine.cfg.data["defaults"]["search_first"] = False
    assert engine.parse(TurnRequest(), "what are the best budget headphones?").route == "chat"


def test_study_mode_teaches_from_sources_and_quizzes(engine, fake):
    evs = run(collect(engine, TurnRequest(message="/study alpha protocol")))
    prompt = next(m for m in fake.calls if "Check yourself" in m[-1]["content"])[-1]["content"]
    assert "<source id=" in prompt and "Key terms" in prompt
    assert any(k == "source" for k, _ in evs)


def test_brain_keeps_lasting_facts_but_never_sensitive_ones(engine, fake):
    fake.responder = lambda msgs: ('{"facts": ["Building an AI startup called Proxmino", "Bank PIN is 4821 and password hunter2"]}'
                                   if "lasting facts" in msgs[0]["content"] else "ok")
    saved = run(engine._learn("I'm building an AI startup called Proxmino, my bank PIN is 4821"))
    assert saved == ["Building an AI startup called Proxmino"]
    assert run(engine._learn("I'm building an AI startup called Proxmino")) == []  # no duplicates
    assert run(engine._learn("what's the weather")) == []  # nothing about the user: the model is not even asked
    engine.cfg.data.setdefault("defaults", {})["auto_memory"] = False
    assert run(engine._learn("I love sailing")) == []
    assert [m["text"] for m in engine.db.memories()] == ["Building an AI startup called Proxmino"]


def test_plain_definitions_skip_the_web_but_named_facts_do_not(engine):
    engine.cfg.data.setdefault("defaults", {})["search_first"] = True
    assert engine.parse(TurnRequest(), "what's a VPN? one sentence").route == "chat"
    assert engine.parse(TurnRequest(), "what does NASA stand for").route == "chat"
    assert engine.parse(TurnRequest(), "what is the capital of Peru?").route == "chat"  # settled fact: no search


def test_simple_facts_answer_straight_away_but_changing_facts_search(engine):
    for q in ["what is the capital of France", "who wrote Hamlet", "how tall is Mount Everest", "when did World War 2 end",
              "who painted the Mona Lisa", "how many legs does a spider have"]:
        st = engine.parse(TurnRequest(), q)
        assert (st.route, st.simple) == ("chat", True), q
    for q in ["what's the population of Japan", "who is the president of the US", "who won the super bowl"]:
        assert not engine.parse(TurnRequest(), q).simple, q


def test_post_page_links_fill_in_the_post():
    from proxmino.pipeline import composer_link
    x = composer_link("X", "Just shipped my app & it works")
    assert x["url"] == "https://x.com/intent/post?text=Just%20shipped%20my%20app%20%26%20it%20works"
    r = composer_link("reddit", "body", title="Is 650W enough?", where="r/buildapc")
    assert r["url"].startswith("https://www.reddit.com/r/buildapc/submit?title=Is%20650W%20enough%3F&text=body")
    assert composer_link("reddit", "b", where="r/../../evil")["url"].startswith("https://www.reddit.com/submit?")
    assert composer_link("email", "hi", to="sam@example.com", subject="Lunch")["url"] == "mailto:sam@example.com?subject=Lunch&body=hi"
    assert composer_link("instagram", "hi") is None and composer_link("x", "  ") is None


def test_draft_is_pulled_out_of_the_chatter_for_the_post_button():
    from proxmino.pipeline import draft_from_reply
    reply = ("I've written the post and set up a button.\n\n---\n\nI'm starting a new job.\n\nHere's to that.\n\n---\n\n"
             "Press the button to post it, or connect LinkedIn in Settings > API keys.")
    assert draft_from_reply(reply) == ("I'm starting a new job.\n\nHere's to that.", "")
    reddit = "Here's the post:\n\n**Title:** Is 650W enough?\n\n**Body:**\n\nThinking of a 4070.\n\nPress the button to post it."
    assert draft_from_reply(reddit) == ("Thinking of a 4070.", "Is 650W enough?")
    assert draft_from_reply("Draft:\n\n> Just shipped it.\n\nPress the button.")[0] == "Just shipped it."


def test_concept_questions_skip_search_and_notes_stay_out_of_search(engine):
    engine.cfg.data.setdefault("defaults", {})["search_first"] = True
    for q in ["explain a for loop in python", "how do I center a div in CSS", "why does Python use indentation"]:
        assert engine.parse(TurnRequest(), q).route != "search", q
    st = engine.parse(TurnRequest(), "who is the current UK prime minister")
    assert st.route == "search"
    msgs = engine._messages(st, "x", [], "sources...\n\nAnswer the question using the sources above.")
    assert "[placeholder]" not in msgs[-1]["content"]


def test_untrusted_tags_never_reach_the_user():
    from proxmino.pipeline import UNTRUSTED_TAG
    assert UNTRUSTED_TAG.sub("", "Lima &lt;untrusted&gt;is</untrusted> the capital") == "Lima is the capital"


def test_shopping_questions_always_search(engine):
    engine.cfg.data.setdefault("defaults", {})["search_first"] = True
    for q in ["best budget wireless earbuds", "top 5 gaming laptops under $1000", "is the steam deck worth it"]:
        assert engine.parse(TurnRequest(), q).route == "search", q


def test_maths_leftovers_are_tidied_outside_maths():
    from proxmino.pipeline import tidy_answer
    assert tidy_answer("CMF buds are good.\n\n<answer>CMF</answer>\n\nAnswer: CMF buds", "search") == "CMF buds are good.\n\nCMF"
    assert tidy_answer("Released in 20**26**.", "chat") == "Released in 2026."
    assert tidy_answer("The ball costs 5 cents, since 2x = 0.10.\n\nAnswer: 5", "think").endswith("since 2x = 0.10.")
    assert tidy_answer("Answer: 4", "think") == "Answer: 4"


def test_app_draws_charts_from_numbers():
    from proxmino.charts import svg_chart
    svg = svg_chart("Sales", ["Jan", "Feb", "Mar"], ["10", 15, "$7"], "bar")
    assert svg.startswith("<svg") and "Jan" in svg and ">15<" in svg
    assert "<path" in svg_chart("Share", ["A", "B"], [1, 3], "pie")
    assert "polyline" in svg_chart("Trend", ["1", "2"], [1, 2], "line")


def test_round_two_fixes():
    from proxmino.pipeline import (clock_sum, strip_leaks, draft_from_reply, relevant, similar, TRANSIENT,
                                   WRITING_TASK, SOCIAL_CUE, tidy_answer)
    from proxmino.search import Evidence
    assert clock_sum("A train leaves at 14:20 and the trip takes 2 h 47 min. What time does it arrive?").endswith("= 17:07")
    assert "10:15 pm" in clock_sum("The movie starts at 7:45 pm and lasts 2 hours 30 minutes. When does it end?")
    assert clock_sum("I woke at 6:10, 45 minutes before the bus. When is the bus?") is None
    out = strip_leaks('Fact &lt;source id="3"&gt;source&lt;/source&gt; [1]. <post>hi</post> <a href="x">link</a> '
                      '![i](https://s3.amazonaws.com/x.png)\n```html\n<button>ok</button>\n```')
    assert "source" not in out and "<post>" not in out and "s3." not in out and "link" in out and "<button>ok</button>" in out
    body, _ = draft_from_reply("Here is your draft:\n\n&lt;post&gt;Shipped my app today. [proxmino.com](https://proxmino.com)"
                               "&lt;/post&gt;\n\n[Open on Reddit](https://reddit.com/r/Python/comments/xxxxxx)\n\nPress the button.")
    assert body == "Shipped my app today. proxmino.com"
    q = "What changed in Python 3.14?"
    assert relevant(Evidence(1, "u", "What's New In Python 3.14", "python.org", "Python 3.14 adds ..."), q)
    assert not relevant(Evidence(2, "u", "Monty Python", "wikipedia.org", "British comedy group"), q)
    assert not relevant(Evidence(3, "u", "Good Morning, Miami", "wikipedia.org", "sitcom"), "weather in Lisbon this week")
    assert similar("wants to budget better each month", "wants to budget better every month")
    assert TRANSIENT.search("Wants to post a gym selfie on Instagram")
    assert WRITING_TASK.search("Say good morning and ask how the weather is")
    assert SOCIAL_CUE.search("Draft an email to nobody@example.com about the meeting")
    assert SOCIAL_CUE.search("Write a LinkedIn post about my internship")
    assert tidy_answer("Sure. &lt;action&gt;post&lt;/action&gt; Done", "chat") == "Sure. post Done"


def test_round_three_fixes():
    from proxmino.pipeline import typed_values, tidy_answer, draft_from_reply, CHAT_SEARCH
    assert typed_values("Make a pie chart of rent 1200, food 400, fun 200", ["Rent", "Food", "Fun"], [12000, 400, 200]) == [1200, 400, 200]
    assert tidy_answer('Saved chart [ab12] "Budget". The user sees it as a preview card above your reply: do not add an image '
                       'link. Exact figures (use these, do not recompute): Rent 1,200 (67%); total 1,800. Rent is largest.', "tools") == "Rent is largest."
    assert tidy_answer(r"So \( x = 50 \) dollars.", "think") == "So x = 50 dollars."
    assert draft_from_reply("Here is a short Bluesky post:\nFinished my project.\n\nThe button below opens the site.")[0] == "Finished my project."
    assert CHAT_SEARCH.match("Search my earlier chats for boxes").group("q") == "boxes"


def test_chat_search_route(engine):
    run(collect(engine, TurnRequest(message="I keep my tools in blue boxes")))
    ev = run(collect(engine, TurnRequest(message="Search my earlier chats for boxes")))
    done = kinds(ev, "done")[0]
    assert done["route"] == "chatsearch" and "boxes" in done["content"].lower() and "Found" in done["content"]


def test_split_parts_and_cues():
    from proxmino.pipeline import split_parts, CODE_TASK, NUMERIC_TASK, clock_sum
    text = ("Act as a professor. Solve these.\n\n=====\nSTAGE 1: MATHS\n=====\nCalculate the probability p = 0.02.\n\n"
            "=====\nSTAGE 2: LOGIC\n=====\nWho has the red hat?\n\nSTAGE 3: CODE\nWrite a Python function named f.")
    intro, parts = split_parts(text)
    assert intro == "Act as a professor. Solve these." and len(parts) == 3
    assert parts[0].startswith("STAGE 1") and "0.02" in parts[0] and "STAGE 2" not in parts[0]
    assert NUMERIC_TASK.search(parts[0]) and not NUMERIC_TASK.search(parts[1]) and CODE_TASK.search(parts[2])
    assert split_parts("just one question about stage 1 of a rocket")[1] == []
    assert clock_sum("p = 0.02 and each attempt takes 5 tokens") is None


def test_program_numbers_agree():
    from proxmino.pipeline import program_numbers
    assert program_numbers("p = 7203/3125000, EV 4") == program_numbers("0.00230496 and 4.0")
    assert program_numbers("2401/781250 and 580") != program_numbers("7203/3125000 and 4")


def test_cut_off_and_hedge():
    from proxmino.pipeline import cut_off, drop_wrong_hedge, HEDGE
    from proxmino.search import Evidence
    assert cut_off("Argentina won the 20") and not cut_off("Argentina won the 2022 FIFA World Cup.")
    assert not cut_off("Here is code:\n```python\nprint(1)\n```") and not cut_off("Done (see above)")
    src = [Evidence(1, "u", "2022 FIFA World Cup final", "wikipedia.org", "Argentina beat France on penalties in Qatar")]
    out = drop_wrong_hedge(HEDGE + " Argentina beat France in Qatar.", src)
    assert out == "Argentina beat France in Qatar [1]."
    assert drop_wrong_hedge(HEDGE + " Brazil won it in Tokyo.", src).startswith(HEDGE)


def _office(files: dict) -> bytes:
    import io, zipfile
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, t in files.items():
            z.writestr(n, t)
    return b.getvalue()


def test_extract_office_files():
    from proxmino.extract import extract_text, Unsupported
    W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    docx = _office({"word/document.xml": f'<w:document {W}><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
                    '<w:r><w:t>Budget</w:t></w:r></w:p><w:p><w:r><w:t>Rent is 1200.</w:t></w:r></w:p></w:body></w:document>'})
    assert extract_text("plan.docx", docx) == "# Budget\n\nRent is 1200."
    S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    xlsx = _office({"xl/sharedStrings.xml": f'<sst {S}><si><t>item</t></si><si><t>rent</t></si></sst>',
                    "xl/workbook.xml": f'<workbook {S}><sheets><sheet name="May"/></sheets></workbook>',
                    "xl/worksheets/sheet1.xml": f'<worksheet {S}><sheetData><row><c r="A1" t="s"><v>0</v></c><c r="B1"><v>7</v></c></row>'
                    '<row><c r="A2" t="s"><v>1</v></c><c r="B2"><v>1200</v></c></row></sheetData></worksheet>'})
    assert extract_text("money.xlsx", xlsx) == "## May\nitem,7\nrent,1200"
    A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    pptx = _office({"ppt/slides/slide1.xml": f'<p:sld xmlns:p="x" {A}><a:p><a:r><a:t>Hello deck</a:t></a:r></a:p></p:sld>'})
    assert extract_text("deck.pptx", pptx) == "[slide 1]\nHello deck"
    assert extract_text("notes.md", b"# hi") == "# hi"
    for name in ("photo.png", "old.doc"):
        try:
            extract_text(name, b"x")
            assert False
        except Unsupported:
            pass


def test_attached_file_routing(engine):
    from proxmino.pipeline import TurnRequest as TR
    msg = '<file name="news.txt">\nThe latest election results today were close.\n</file>\n\nSummarize this in one line.'
    st = engine.parse(TR(message=msg), msg)
    assert st.text == "Summarize this in one line." and st.files and st.files[0][0] == "news.txt"
    assert st.route != "search"  # words inside the file must not trigger a web search
    msgs = engine._messages(st, "c1", [], st.text)
    assert "The latest election results" in msgs[-1]["content"] and "Summarize this" in msgs[-1]["content"]


def test_word_problems_pages_and_plain_maths(engine):
    from proxmino.pipeline import TurnRequest as TR, plain_maths, tidy_answer
    for q in ("A jacket costs $40. It is 25% off, then 8% sales tax is added to the discounted price. What do I pay?",
              "Maria is 3 times as old as her son. In 12 years she will be twice as old as him. How old is Maria now?"):
        assert engine.parse(TR(message=q), q).route == "think", q  # every number is in the question: no web search
    q = "What is the price of gold today?"
    assert engine.parse(TR(message=q), q).route != "think"
    q = ("Build a single-file HTML page: a pricing section with three plans (Free, Pro, Team) and a monthly/yearly "
         "toggle that changes the prices.")
    st = engine.parse(TR(message=q), q)
    assert st.route == "chat" and st.page and not st.tools  # a page, not a chart of the prices
    assert plain_maths(r"Probability = \frac{5}{36}, about 12\% and 3 \times 4") == "Probability = 5/36, about 12% and 3 × 4"
    assert plain_maths("```python\nx = r'\\frac{a}{b}'\n```") == "```python\nx = r'\\frac{a}{b}'\n```"  # code untouched
    assert tidy_answer(r"Final answer: \frac{5}{36}", "think").endswith("5/36")


def test_hard_maths_routing_and_imports(engine):
    from proxmino.pipeline import TurnRequest as TR, add_missing_imports, plain_maths
    for q in ("What is the remainder when 3^2024 is divided by 1000?",
              "What is the smallest positive integer n such that n! ends in at least 100 zeros?",
              "You roll a fair die until you get two sixes in a row. What is the expected number of rolls?"):
        assert engine.parse(TR(message=q), q).route == "think", q
    q = "Who won the 2024 election?"
    assert engine.parse(TR(message=q), q).route != "think"
    fixed = add_missing_imports("```python\ndef f(a, x):\n    return bisect.bisect_left(a, x)\n```")
    assert fixed.startswith("```python\nimport bisect\n")
    assert add_missing_imports("```python\nimport bisect\nbisect.bisect(a, 1)\n```").count("import bisect") == 1
    assert plain_maths(r"3^{10} \equiv 49 \pmod{125} \implies x") == "3^10 ≡ 49 (mod 125) ⇒ x"
