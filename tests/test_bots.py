"""Custom bots (builder), the marketplace, chats that belong to a bot, and group chats."""
from fastapi.testclient import TestClient

from proxmino import bots as B
from proxmino.pipeline import TurnRequest
from proxmino.server import create_app

from conftest import collect, run

H = {"X-Proxmino": "1"}


def client(engine):
    return TestClient(create_app(engine, run_scheduler=False), base_url="http://127.0.0.1")


def test_builder_creates_edits_and_deletes_a_bot(engine, tmp_path):
    with client(engine) as c:
        made = c.post("/api/agents", json={"name": "Night Owl", "color": "#3366ff", "shape": 4,
                                           "description": "Works late", "instructions": "Be brief."}, headers=H).json()
        assert made["id"] == "night-owl" and made["shape"] == 4 and made["builtin"] is False
        assert (tmp_path / "bots" / "night-owl.yaml").exists()
        assert "Be brief." in engine.agents["night-owl"].persona_block()
        again = c.post("/api/agents", json={"name": "Night Owl"}, headers=H).json()
        assert again["id"] == "night-owl-2"  # names can repeat, ids stay unique
        edited = c.put("/api/agents/night-owl", json={"name": "Night Owl", "color": "#ff0000", "temperature": 0.9}, headers=H).json()
        assert edited["color"] == "#ff0000" and edited["temperature"] == 0.9
        for bad in ({"name": ""}, {"name": "X", "color": "red"}, {"name": "X", "shape": 12}, {"name": "X", "tier": "huge"},
                    {"name": "X", "traits": {"evil": 1}}):
            assert c.post("/api/agents", json=bad, headers=H).status_code == 400
        assert c.put("/api/agents/forge", json={"name": "Mine now"}, headers=H).status_code == 400  # built-ins are fixed
        assert c.delete("/api/agents/forge", headers=H).status_code == 400
        assert c.delete("/api/agents/night-owl", headers=H).json() == {"ok": True}
        assert "night-owl" not in engine.agents and not (tmp_path / "bots" / "night-owl.yaml").exists()


def test_every_marketplace_bot_is_valid_and_installs(engine):
    items = B.market(engine.agents)
    assert len(items) >= 12 and any(i["featured"] for i in items)
    assert {i["category"] for i in items} >= {"Productivity", "Writing", "Business", "Engineering"}
    with client(engine) as c:
        for i in items:
            got = c.post(f"/api/market/{i['id']}/install", headers=H).json()
            assert got["id"] == i["id"] and got["builtin"] is False
        assert all(i["installed"] for i in c.get("/api/market").json())
        assert c.post("/api/market/..%2Fsecrets/install", headers=H).status_code in (400, 404)


def test_chat_that_belongs_to_a_bot_uses_it(engine):
    with client(engine) as c:
        chat = c.post("/api/chats", json={"agent": "forge"}, headers=H).json()
        assert chat["agent"] == "forge" and chat["title"] == "Forge"
    evs = run(collect(engine, TurnRequest(message="hello", chat_id=chat["id"])))
    meta = dict(evs)["meta"]
    assert meta["agent"]["id"] == "forge"
    row = engine.db.chats()[0]
    assert row["last"].startswith("Echo") and row["last_agent"] == "forge"


def test_group_chat_lead_replies_and_hands_off(engine, fake):
    with client(engine) as c:
        assert c.post("/api/chats", json={"members": ["forge"]}, headers=H).status_code == 400
        chat = c.post("/api/chats", json={"title": "Startup", "members": ["atlas", "forge", "probe"]}, headers=H).json()
    seen = []

    def team(msgs):
        system = msgs[0]["content"]
        if "short chat titles" in system or "summarize" in system:
            return "Title"
        seen.append(msgs)
        if "You are Atlas" in system:
            return "[Atlas] Plan: three steps. @forge can you build step one?"
        if "You are Forge" in system:
            return "Forge: Built it."
        return "Probe here."
    fake.responder = team
    evs = run(collect(engine, TurnRequest(message="Ship the login page", chat_id=chat["id"])))
    kinds = [k for k, _ in evs]
    assert kinds.count("done") == 2 and kinds.count("speaker") == 1  # Atlas leads, hands off to Forge; Probe is not asked
    speaker = dict(evs)["speaker"]
    assert speaker["agent"]["id"] == "forge"
    msgs = engine.db.messages(chat["id"])
    a, f = [m for m in msgs if m["role"] == "assistant"]
    assert a["content"].startswith("Plan:") and f["content"] == "Built it."  # name prefixes stripped
    assert f["parent_id"] == a["id"] and a["meta"]["agent"]["id"] == "atlas" and f["meta"]["agent"]["id"] == "forge"
    forge_prompt = "\n".join(m["content"] for m in seen[1])
    assert "group chat called \"Startup\"" in forge_prompt and "[Atlas] Plan" in forge_prompt
    # @mentions pick who answers; @all asks everyone, in member order
    evs = run(collect(engine, TurnRequest(message="@probe test it", chat_id=chat["id"], parent_id=f["id"])))
    assert dict(evs)["meta"]["agent"]["id"] == "probe" and [k for k, _ in evs].count("done") == 1
    later = [m for m in engine.db.messages(chat["id"]) if m["role"] == "assistant"][-1]
    probe_prompt = "\n".join(m["content"] for m in seen[-1])
    assert later["meta"]["agent"]["id"] == "probe" and "[Forge] Built it." in probe_prompt
    evs = run(collect(engine, TurnRequest(message="@all status?", chat_id=chat["id"], parent_id=later["id"])))
    assert [k for k, _ in evs].count("done") == 3
