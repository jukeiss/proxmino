"""MCP client, approval rules (ported from Rakazo), and the chat tool loop. No real model runs."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from conftest import collect, run
from proxmino import approval as AP
from proxmino import mcp as M
from proxmino.pipeline import TurnRequest

SERVER = str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")


def notes_cfg(**kw):
    return M.ServerConfig(name="Notes", command="python3", args=[SERVER], **kw)


# ---------------------------------------------------------------- approval rules

@pytest.mark.parametrize("tool,ro,needs", [
    ("list_issues", None, False), ("get_file", None, False), ("search", None, False), ("readFile", None, False),
    ("send_email", None, True), ("create_issue", None, True), ("delete-repo", None, True),
    ("get_and_delete", None, True),      # compound actions ask
    ("summarize", None, True),           # unknown verbs ask
    ("list_notes", False, True),         # a declared write always asks
    ("send_message", True, True),        # a declared read never relaxes the name check
])
def test_name_rules(tool, ro, needs):
    assert AP.tool_requires_approval(tool, ro) is needs


def test_user_rules_override(tmp_path):
    rules = AP.Rules(tmp_path / "r.json")
    assert AP.decide("gh__create_issue", None, rules.all())[0] == "ask"
    rules.add(AP.Rule("always_allow", "tool", "gh__create_issue"))
    assert AP.decide("gh__create_issue", None, rules.all())[0] == "allow"
    rules.add(AP.Rule("deny", "server", "gh"))
    assert AP.decide("gh__list_issues", None, rules.all())[0] == "deny"
    assert AP.decide("gh__create_issue", None, rules.all())[0] == "allow"  # the tool rule beats the server rule
    with pytest.raises(ValueError):
        rules.add(AP.Rule("maybe", "tool", "x"))


# ---------------------------------------------------------------- client

def test_stdio_server_lists_and_calls_tools(tmp_path):
    async def go():
        m = M.Manager(tmp_path / "mcp.json", None)
        m.upsert(notes_cfg())
        await m.refresh()
        names = [t["name"] for t in m.catalog()]
        added = await m.call("notes__add_note", {"text": "buy milk"})
        listed = await m.call("notes__list_notes", {})
        await m.close()
        return names, added, listed, m.errors
    names, added, listed, errors = run(go())
    assert names == ["notes__list_notes", "notes__add_note"] and not errors
    assert added == "saved (1 notes)" and listed == "buy milk"


def test_launch_commands_are_allowlisted(tmp_path):
    m = M.Manager(tmp_path / "mcp.json", None)
    with pytest.raises(M.MCPError, match="not an allowed launch command"):
        m.upsert(M.ServerConfig(name="evil", command="bash", args=["-c", "rm -rf ~"]))
    assert m.servers() == []


def test_remote_urls_must_be_https():
    for bad in ["http://example.com/mcp", "https://user:pw@example.com/mcp", "https://example.com/mcp#x", "ftp://x"]:
        with pytest.raises(M.MCPError):
            M.validate_url(bad)
    assert M.validate_url("http://localhost:8931/mcp")
    assert M.validate_url("https://mcp.example.com/mcp")


def test_failed_server_reports_its_error_output(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nsys.stderr.write('missing API token\\n')\nsys.exit(3)\n")
    async def go():
        m = M.Manager(tmp_path / "mcp.json", None, timeout=5)
        m.upsert(M.ServerConfig(name="Bad", command="python3", args=[str(bad)]))
        await m.refresh()
        return m.errors
    errors = run(go())
    assert "missing API token" in errors["bad"]


def test_http_server_json_and_sse(tmp_path):
    seen = []

    def handler(req):
        msg = json.loads(req.content) if req.content else {}
        seen.append(dict(req.headers))
        if req.method == "DELETE" or "id" not in msg:
            return httpx.Response(202)
        m = msg["method"]
        if m == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": {"serverInfo": {"name": "r"}}},
                                  headers={"mcp-session-id": "s1"})
        if m == "tools/list":
            body = {"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [{"name": "get_weather", "inputSchema": {}}]}}
            return httpx.Response(200, text=f"event: message\ndata: {json.dumps(body)}\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": "sunny"}]}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            m = M.Manager(tmp_path / "mcp.json", c)
            m.upsert(M.ServerConfig(name="Remote", transport="http", url="https://mcp.example.com/mcp",
                                    headers={"Authorization": "Bearer t"}))
            await m.refresh()
            out = await m.call("remote__get_weather", {"city": "Oslo"})
            await m.close()
            return [t["name"] for t in m.catalog()], out
    names, out = run(go())
    assert names == ["remote__get_weather"] and out == "sunny"
    assert seen[-2].get("mcp-session-id") == "s1" and seen[-2].get("authorization") == "Bearer t"


def test_public_view_hides_secret_values(tmp_path):
    cfg = M.ServerConfig(name="GH", transport="http", url="https://x.example/mcp", headers={"Authorization": "Bearer secret"})
    assert "secret" not in json.dumps(cfg.public()) and cfg.public()["header_keys"] == ["Authorization"]


# ---------------------------------------------------------------- chat tool loop
# Each test runs in one event loop: a server process belongs to the loop that started it.

def scripted(fake, replies):
    it = iter(replies)
    # Background chat titles also call the model; answer those separately so the script stays in order.
    fake.responder = lambda msgs: "A title" if "chat titles" in msgs[0]["content"] else next(it)


async def turn(engine, decision=None, **kw):
    events = []
    async for k, d in engine.turn(TurnRequest(tools=True, **kw)):
        events.append((k, d))
        if k == "approval" and decision:
            asyncio.get_running_loop().call_soon(engine.resolve_approval, d["id"], decision)
    return events


def steps(evs):
    return " | ".join(d["text"] for k, d in evs if k == "step")


def test_tool_loop_reads_without_asking(engine, fake):
    engine.mcp.upsert(notes_cfg())
    scripted(fake, ['{"tool": "notes__list_notes", "args": {}}', "You have no notes yet."])

    async def go():
        evs = await turn(engine, message="what notes do I have?")
        await engine.mcp.close()
        return evs
    evs = run(go())
    assert [d for k, d in evs if k == "done"][0]["content"] == "You have no notes yet."
    assert "Called notes__list_notes" in steps(evs) and not [1 for k, _ in evs if k == "approval"]


def test_tool_loop_asks_before_a_write_and_respects_the_answer(engine, fake):
    engine.mcp.upsert(notes_cfg())
    scripted(fake, ['{"tool": "notes__add_note", "args": {"text": "call mom"}}', "Saved it.",
                    '{"tool": "notes__add_note", "args": {"text": "second"}}', "Okay, not saved."])

    async def go():
        first = await turn(engine, "allow", message="remember to call mom")
        second = await turn(engine, "deny", message="also remember something else")
        listed = await engine.mcp.call("notes__list_notes", {})
        await engine.mcp.close()
        return first, second, listed
    first, second, listed = run(go())
    approval = [d for k, d in first if k == "approval"][0]
    assert approval["tool"] == "notes__add_note" and approval["args"] == {"text": "call mom"}
    assert "Called notes__add_note" in steps(first)
    assert "Did not run notes__add_note: you declined" in steps(second)
    assert listed == "call mom"  # only the approved call ran


def test_always_allow_saves_a_rule(engine, fake):
    engine.mcp.upsert(notes_cfg())
    scripted(fake, ['{"tool": "notes__add_note", "args": {"text": "x"}}', "Done.",
                    '{"tool": "notes__add_note", "args": {"text": "y"}}', "Done again."])

    async def go():
        first = await turn(engine, "always", message="save x")
        second = await turn(engine, None, message="save y")  # allowed by the rule: no card
        await engine.mcp.close()
        return first, second
    first, second = run(go())
    assert [r.value for r in engine.rules.all()] == ["notes__add_note"]
    assert not [1 for k, _ in second if k == "approval"] and "Called notes__add_note" in steps(second)


def test_unattended_runs_refuse_tools_that_need_approval(engine, fake):
    engine.mcp.upsert(notes_cfg())
    scripted(fake, ['{"tool": "notes__add_note", "args": {"text": "x"}}', "I could not save it."])

    async def go():
        evs = await turn(engine, message="save x", unattended=True)
        await engine.mcp.close()
        return evs
    evs = run(go())
    assert not [1 for k, _ in evs if k == "approval"] and "nobody can approve" in steps(evs)


def test_plain_answer_streams_when_no_tool_is_needed(engine, fake):
    engine.mcp.upsert(notes_cfg())
    scripted(fake, ["Paris is the capital of France."])

    async def go():
        evs = await turn(engine, message="capital of France?")
        await engine.mcp.close()
        return evs
    tokens = [d["t"] for k, d in run(go()) if k == "token"]
    assert "".join(tokens) == "Paris is the capital of France." and len(tokens) > 1


def test_without_servers_the_built_in_tools_remain(engine, fake):
    scripted(fake, ["Hello there."])
    evs = run(collect(engine, TurnRequest(message="hello", tools=True)))
    assert [d for k, d in evs if k == "meta"][0]["route"] == "tools"
    assert [d for k, d in evs if k == "done"][0]["content"] == "Hello there."


def test_without_the_tools_pill_it_is_a_normal_chat(engine, fake):
    evs = run(collect(engine, TurnRequest(message="hello")))
    assert [d for k, d in evs if k == "meta"][0]["route"] == "chat"


# ---------------------------------------------------------------- built-in tools (Rakazo scratchpad, search_history)

def test_scratchpad_tools_belong_to_the_agent_and_need_no_approval(engine, fake):
    scripted(fake, ['{"tool": "scratchpad_add", "args": {"title": "Write parser tests", "notes": "edge cases"}}', "Noted.",
                    '{"tool": "scratchpad_list", "args": {}}', "You have one item."])
    evs = run(collect(engine, TurnRequest(message="@forge note that I still need parser tests", tools=True)))
    assert not [1 for k, _ in evs if k == "approval"]
    items = engine.db.scratch_list("forge")
    assert [(i["title"], i["status"], i["notes"]) for i in items] == [("Write parser tests", "open", "edge cases")]
    assert engine.db.scratch_list("proxmino") == []  # another agent's pad is separate
    prompts = []

    def record(msgs):
        prompts.append(msgs[0]["content"])
        return "ok"
    fake.responder = record
    run(collect(engine, TurnRequest(message="@forge what is open?")))
    assert "Your scratchpad" in prompts[0] and "Write parser tests" in prompts[0]


def test_scratchpad_complete_and_ownership(engine, fake):
    mine = engine.db.scratch_add("proxmino", "Mine")
    theirs = engine.db.scratch_add("forge", "Theirs")
    scripted(fake, [json.dumps({"tool": "scratchpad_complete", "args": {"id": theirs["id"]}}), "Could not.",
                    json.dumps({"tool": "scratchpad_complete", "args": {"id": mine["id"]}}), "Done."])
    run(collect(engine, TurnRequest(message="finish theirs", tools=True)))
    run(collect(engine, TurnRequest(message="finish mine", tools=True)))
    assert engine.db.scratch_get(theirs["id"])["status"] == "open"
    assert engine.db.scratch_get(mine["id"])["status"] == "done"


def test_search_chats_tool_and_api(engine, fake):
    run(collect(engine, TurnRequest(message="Plan a hiking trip to Norway")))
    scripted(fake, ['{"tool": "search_chats", "args": {"query": "norway"}}', "You asked about Norway before."])
    evs = run(collect(engine, TurnRequest(message="did we talk about norway?", tools=True)))
    assert "Called search_chats" in steps(evs)
    from fastapi.testclient import TestClient
    from proxmino.server import create_app
    with TestClient(create_app(engine, run_scheduler=False), base_url="http://127.0.0.1") as c:
        hits = c.get("/api/search", params={"q": "hiking norw"}).json()
    assert hits and "[[hiking]]" in hits[0]["snip"].lower() and hits[0]["chat_id"]


# ---------------------------------------------------------------- artifacts and knowledge (Rakazo)

def test_create_and_update_artifact_tools(engine, fake):
    html = "<h1>Budget</h1><p>Total: 42</p>"
    scripted(fake, [json.dumps({"tool": "create_artifact", "args": {"title": "Budget page", "kind": "html", "content": html}}),
                    "Here is your budget page."])
    evs = run(collect(engine, TurnRequest(message="make me a budget page", tools=True)))
    made = [d for k, d in evs if k == "artifact"]
    assert made and made[0]["title"] == "Budget page" and made[0]["kind"] == "html"
    done = [d for k, d in evs if k == "done"][0]
    assert done["artifacts"][0]["id"] == made[0]["id"]
    a = engine.db.artifact(made[0]["id"])
    assert a["content"] == html and a["chat_id"] == [d for k, d in evs if k == "meta"][0]["chat_id"]
    scripted(fake, [json.dumps({"tool": "update_artifact", "args": {"id": a["id"], "content": "<h1>Budget v2</h1>"}}), "Updated."])
    run(collect(engine, TurnRequest(message="update it", tools=True)))
    a2 = engine.db.artifact(a["id"])
    assert a2["version"] == 2 and a2["content"] == "<h1>Budget v2</h1>" and a2["title"] == "Budget page"


def test_knowledge_chunks_search_and_tool(engine, fake):
    from proxmino.db import chunk_text
    doc = ("The Lumen lamp ships with a 2-year warranty.\n\n" + "Filler sentence about lamps. " * 60 +
           "\n\nReturns are accepted within 30 days of delivery.")
    chunks = chunk_text(doc)
    assert len(chunks) >= 2 and all(len(c) <= 1300 for c in chunks)
    d = engine.db.knowledge_add("lumen-manual.md", doc, "markdown")
    assert d["chunks"] == len(chunks)
    hits = engine.db.knowledge_search("how many days for returns")
    assert hits and "30 days" in hits[0]["text"] and hits[0]["name"] == "lumen-manual.md"
    scripted(fake, ['{"tool": "search_knowledge", "args": {"query": "warranty"}}', "The warranty is 2 years (lumen-manual.md)."])
    evs = run(collect(engine, TurnRequest(message="what is the warranty?", tools=True)))
    assert "Called search_knowledge" in steps(evs)
    engine.db.knowledge_delete(d["id"])
    assert engine.db.knowledge_search("warranty") == [] and engine.db.knowledge_docs() == []
