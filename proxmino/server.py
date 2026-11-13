"""FastAPI app: SSE chat plus the small JSON API the web UI uses."""
from __future__ import annotations

import asyncio
import os
import json
import logging
import time
from html import escape as html_escape
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import prompts as P
from .config import DATA_DIR, ENV_FILE, ROOT, Config, lock_down
from . import approval as AP
from . import mcp as MCPmod
from . import auth as AU
from . import inbox as INBOX
from . import integrations as IN
from . import reports as REPORTS
from . import keys as KEYS
from . import connections as CONN
from . import reddit as REDDIT
from . import lab as LAB
from . import stats as ST
from . import bots as BOTS
from . import routines as RT
from .llm import Params
from .pipeline import COMMANDS, Engine, TurnRequest, agents_dir
from .scheduler import Scheduler, schedule
from .sandbox import run_python

log = logging.getLogger("proxmino.server")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "testserver"}
# Sent with every response. No outside scripts, styles, fonts or images; the page only talks to this app;
# frames (artifact previews) inherit this, so model-written HTML cannot load or send anything either.
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; media-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
       "frame-src 'self' blob: data: about:; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; "
       "form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer", "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), geolocation=(), payment=(), usb=(), microphone=(self)",
}
MAX_BODY = 25 * 1024 * 1024  # largest request accepted (attachments and Knowledge uploads included)


def secure(response):
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response

REMOTE_HOSTS = [".trycloudflare.com", ".ts.net"]  # tunnel addresses allowed (with a login); more in config remote.hosts


def remote_hosts(cfg) -> list[str]:
    """Addresses other devices may use (always with the password): tunnels, plus config remote.hosts and
    PROXMINO_HOSTS (comma-separated, for example the home server's LAN address 192.168.1.50)."""
    raw = ",".join(os.environ.get(k, "") for k in ("PROXMINO_HOSTS", "PRAXMINO_HOSTS"))  # old name still works
    extra = [h.strip().lower() for h in raw.split(",") if h.strip()]
    return list(cfg.get("remote.hosts") or REMOTE_HOSTS) + extra


class ChatIn(BaseModel):
    message: str = ""
    chat_id: str | None = None
    parent_id: str | None = None
    regenerate: str | None = None
    search: bool = False
    deep: bool = False
    think: bool = False
    heavy: bool = False
    study: bool = False
    persona: str | None = None
    agent: str | None = None
    lang: str | None = None
    model: str | None = None
    tools: bool = False


class NewChatIn(BaseModel):
    title: str = ""
    agent: str | None = None          # a chat with one bot
    members: list[str] = []           # a group chat with several bots


class BotIn(BaseModel):
    name: str
    color: str = "#8b8c93"
    shape: int | None = None
    role: str = ""
    description: str = ""
    voice: str = ""
    instructions: str = ""
    tier: str = "chat"
    temperature: float = 0.5
    traits: dict = {}
    working_style: list[str] = []
    never: list[str] = []


class McpIn(BaseModel):
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    url: str = ""
    headers: dict[str, str] = {}
    enabled: bool = True


class ArtifactIn(BaseModel):
    title: str
    kind: str = "html"
    content: str
    chat_id: str | None = None


class KnowledgeIn(BaseModel):
    name: str
    text: str
    kind: str = "text"


class ScratchIn(BaseModel):
    title: str = ""
    agent: str | None = None
    status: str | None = None
    notes: str | None = None


class RuleIn(BaseModel):
    effect: str
    kind: str
    value: str


class ApprovalIn(BaseModel):
    decision: str


class MemoryIn(BaseModel):
    text: str
    agent: str | None = None


class RunIn(BaseModel):
    code: str


class FeedbackIn(BaseModel):
    message_id: str
    kind: str
    note: str = ""


class ReplyDraftIn(BaseModel):
    text: str
    platform: str = "x"
    goal: str = ""
    voice: str = ""
    product: str = ""


class ReplyPostIn(BaseModel):
    text: str
    reply_to: str | None = None
    platform: str = "x"


class KeysIn(BaseModel):
    values: dict[str, str]


class LoginIn(BaseModel):
    password: str


class TitleIn(BaseModel):
    title: str


class RoutineIn(BaseModel):
    name: str
    prompt: str
    freq: str = "Every day"
    n: int = 3
    unit: str = "minutes"
    time: str = "9:00 AM"
    cron: str = ""
    enabled: bool = True
    options: dict = {}


class PresetIn(BaseModel):
    freq: str = "Every day"
    n: int = 3
    unit: str = "minutes"
    time: str = "9:00 AM"
    cron: str = ""


def create_app(engine: Engine | None = None, cfg: Config | None = None, run_scheduler: bool = True,
               auth_path: Path | None = None, env_file: Path | None = None) -> FastAPI:
    state: dict = {"engine": engine}
    auth = AU.Auth(auth_path or DATA_DIR / "auth.json")
    env_path = env_file or ENV_FILE

    @asynccontextmanager
    async def lifespan(app):
        if state["engine"] is None:
            lock_down()
            state["engine"] = Engine.create(cfg or Config())
        state["scheduler"] = Scheduler(state["engine"])
        if run_scheduler:
            state["scheduler"].start()
        yield
        await state["scheduler"].stop()
        await state["engine"].close()

    app = FastAPI(title="Proxmino", lifespan=lifespan, docs_url=None, redoc_url=None)

    def eng() -> Engine:
        return state["engine"]

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Blocks DNS-rebinding and cross-site requests from other web pages: the API can run code.
        # Requests through a tunnel (Cloudflare, Tailscale) need a password login; this computer does not.
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
        remote = AU.is_remote(request.headers, host, LOCAL_HOSTS)
        if host not in LOCAL_HOSTS and not any(host == h.lstrip(".") or host.endswith(h if h.startswith(".") else "." + h)
                                               for h in (remote_hosts(eng().cfg))):
            return secure(JSONResponse({"detail": "Host not allowed"}, status_code=403))
        if request.method not in ("GET", "HEAD", "OPTIONS") and "1" not in (request.headers.get("x-proxmino"), request.headers.get("x-praxis")):
            return secure(JSONResponse({"detail": "Missing X-Proxmino header"}, status_code=403))  # X-Praxis: the old name, still accepted
        try:
            too_big = int(request.headers.get("content-length") or 0) > MAX_BODY
        except ValueError:
            too_big = True
        if too_big:
            return secure(JSONResponse({"detail": "Request too large"}, status_code=413))
        if remote and not auth.valid(request.cookies.get(AU.COOKIE)):
            if not auth.enabled:
                return secure(JSONResponse({"detail": "Remote access is off. On the computer running it: python -m proxmino password"}, status_code=403))
            if request.url.path == "/" and request.method == "GET":
                return secure(HTMLResponse(AU.LOGIN_PAGE.format(product=html_escape(eng().cfg.product_name))))
            if request.url.path != "/api/login":
                return secure(JSONResponse({"detail": "Sign in first"}, status_code=401))
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if remote:
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.post("/api/login")
    async def login(body: LoginIn, request: Request):
        who = (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
               or (request.client.host if request.client else "?"))
        if auth.limited(who):
            raise HTTPException(429, "Too many attempts. Try again in 10 minutes.")
        if not auth.check(body.password):
            auth.failed(who)
            raise HTTPException(401, "Wrong password")
        r = JSONResponse({"ok": True})
        # Secure only over HTTPS: browsers drop Secure cookies on http://192.168.x.x, so a home server on the
        # LAN could never keep you signed in. Tailscale and tunnels are HTTPS and keep the flag.
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
        r.set_cookie(AU.COOKIE, auth.new_session(), max_age=AU.SESSION_DAYS * 86400, httponly=True, secure=https,
                     samesite="strict", path="/")
        return r

    @app.post("/api/logout")
    async def logout(request: Request):
        auth.end_session(request.cookies.get(AU.COOKIE))
        r = JSONResponse({"ok": True})
        r.delete_cookie(AU.COOKIE, path="/")
        return r

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = (ROOT / "web" / "index.html").read_text()
        return html.replace("{{PRODUCT}}", eng().cfg.product_name)

    @app.post("/api/chat")
    async def chat(body: ChatIn, request: Request):
        req = TurnRequest(**body.model_dump())

        async def events():
            agen = eng().turn(req)
            try:
                async for kind, data in agen:
                    yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                    if await request.is_disconnected():
                        break
            finally:
                await agen.aclose()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/chats")
    async def chats():
        return eng().db.chats()

    @app.post("/api/chats")
    async def new_chat(body: NewChatIn):
        e = eng()
        members = list(dict.fromkeys(m for m in body.members if m in e.agents))
        if body.agent and body.agent not in e.agents:
            raise HTTPException(400, "Unknown bot")
        if body.members and len(members) < 2:
            raise HTTPException(400, "A group needs at least two bots")
        title = body.title.strip()[:80] or (e.agents[body.agent].name if body.agent else "Group")
        cid = e.db.create_chat(title, body.agent if not members else None, members or None)
        return e.db.chat(cid)

    @app.get("/api/chats/{cid}")
    async def chat_detail(cid: str):
        chat = eng().db.chat(cid)
        if not chat:
            raise HTTPException(404, "Chat not found")
        return {"chat": chat, "messages": eng().db.messages(cid)}

    @app.patch("/api/chats/{cid}")
    async def rename(cid: str, body: TitleIn):
        eng().db.rename_chat(cid, body.title.strip()[:80])
        return {"ok": True}

    @app.delete("/api/chats/{cid}")
    async def delete_chat(cid: str):
        eng().db.delete_chat(cid)
        return {"ok": True}

    @app.get("/api/memory")
    async def memory():
        return eng().db.memories()

    @app.post("/api/memory")
    async def add_memory(body: MemoryIn):
        if not body.text.strip():
            raise HTTPException(400, "Memory text is empty")
        if body.agent and body.agent not in eng().agents:
            raise HTTPException(400, "Unknown agent")
        return {"id": eng().db.add_memory(body.text, body.agent)}

    @app.delete("/api/memory/{mid}")
    async def delete_memory(mid: int):
        eng().db.delete_memory(mid)
        return {"ok": True}

    @app.delete("/api/memory")
    async def forget_all():
        eng().db.delete_memory()
        return {"ok": True}

    @app.post("/api/run")
    async def run(body: RunIn):
        return asdict(await run_python(body.code, eng().cfg.get("limits.sandbox_seconds", 10)))

    @app.post("/api/feedback")
    async def feedback(body: FeedbackIn):
        eng().db.add_feedback(body.message_id, body.kind, body.note)
        return {"ok": True}

    # ------------------------------------------------------------ search across chats, scratchpad

    @app.get("/api/search-providers")
    async def search_providers():
        """Which web search providers are set up (key names only, never the keys)."""
        from .search import KEYED
        e = eng()
        order = e.cfg.get("search.providers", []) or []
        out = []
        for name, (_, env) in KEYED.items():
            env = e.cfg.get(f"search.{name}_key_env", env) or env
            out.append({"name": name, "official": True, "key_env": env, "has_key": bool(os.environ.get(env)), "enabled": name in order})
        out.append({"name": "searxng", "official": True, "url": e.cfg.get("search.searxng_url") or "", "enabled": "searxng" in order})
        out.append({"name": "ddgs", "official": False, "enabled": bool(e.cfg.get("search.allow_unofficial", False))})
        return {"providers": out, "active": [p.name for p in e.search.providers]}

    @app.get("/api/search")
    async def search_chats(q: str = "", limit: int = 30):
        return eng().db.search_messages(q, max(1, min(limit, 100)))

    @app.get("/api/scratchpad")
    async def scratch_list(agent: str | None = None, include_done: bool = False):
        return eng().db.scratch_list(agent, include_done)

    @app.post("/api/scratchpad")
    async def scratch_add(body: ScratchIn):
        try:
            return eng().db.scratch_add(body.agent or "proxmino", body.title, body.status or "open", body.notes or "")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.patch("/api/scratchpad/{iid}")
    async def scratch_update(iid: str, body: ScratchIn):
        try:
            return eng().db.scratch_update(iid, body.title or None, body.status, body.notes)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/scratchpad/{iid}")
    async def scratch_delete(iid: str):
        return {"ok": eng().db.scratch_remove(iid)}

    # ------------------------------------------------------------ artifacts and knowledge

    @app.get("/api/artifacts")
    async def artifacts_list(chat_id: str | None = None):
        return eng().db.artifacts(chat_id)

    @app.get("/api/artifacts/{aid}")
    async def artifact_get(aid: str):
        a = eng().db.artifact(aid)
        if not a:
            raise HTTPException(404, "Artifact not found")
        return a

    @app.post("/api/artifacts")
    async def artifact_create(body: ArtifactIn):
        try:
            return eng().db.artifact_save(body.title, body.kind, body.content, body.chat_id)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/artifacts/{aid}")
    async def artifact_delete(aid: str):
        eng().db.artifact_delete(aid)
        return {"ok": True}

    @app.get("/api/knowledge")
    async def knowledge_list():
        return eng().db.knowledge_docs()

    @app.post("/api/knowledge")
    async def knowledge_add(body: KnowledgeIn):
        if len(body.text) > 5_000_000:
            raise HTTPException(400, "Documents can be at most 5 MB of text.")
        try:
            return eng().db.knowledge_add(body.name, body.text, body.kind)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/extract")
    async def extract(request: Request, name: str = "file"):
        """Text from an attached PDF, Word, Excel or PowerPoint file (read here, on this computer)."""
        from .extract import Unsupported, extract_text
        data = await request.body()
        if not data:
            raise HTTPException(400, "The file is empty.")
        try:
            text = await asyncio.to_thread(extract_text, name, data)
        except Unsupported as e:
            raise HTTPException(415, str(e))
        if not text.strip():
            raise HTTPException(415, "No text was found in this file.")
        return {"name": name, "text": text}

    @app.delete("/api/knowledge/{did}")
    async def knowledge_delete(did: str):
        eng().db.knowledge_delete(did)
        return {"ok": True}

    @app.get("/api/knowledge/search")
    async def knowledge_search(q: str = ""):
        return eng().db.knowledge_search(q, 8)

    # ------------------------------------------------------------ MCP tools and approvals

    def _mcp_view() -> dict:
        m = eng().mcp
        tools = {}
        for t in m.catalog():
            tools.setdefault(t["server"], []).append({"name": t["tool"], "full": t["name"], "description": t["description"],
                                                      "needs_approval": AP.decide(t["name"], t["annotations"].get("readOnlyHint"), [])[0] == "ask"})
        return {"servers": [{**s.public(), "tools": tools.get(s.id, []), "error": m.errors.get(s.id)} for s in m.servers()],
                "allowed_commands": m.allowed, "rules": [r.__dict__ for r in eng().rules.all()]}

    @app.get("/api/mcp")
    async def mcp_list():
        return _mcp_view()

    @app.post("/api/mcp")
    async def mcp_save(body: McpIn):
        try:
            eng().mcp.upsert(MCPmod.ServerConfig(**body.model_dump()))
        except MCPmod.MCPError as e:
            raise HTTPException(400, str(e))
        await eng().mcp.refresh()
        eng()._mcp_ready = True
        return _mcp_view()

    @app.delete("/api/mcp/{sid}")
    async def mcp_delete(sid: str):
        eng().mcp.remove(sid)
        await eng().mcp.refresh()
        return _mcp_view()

    @app.post("/api/mcp/refresh")
    async def mcp_refresh():
        await eng().mcp.refresh()
        eng()._mcp_ready = True
        return _mcp_view()

    @app.post("/api/approval-rules")
    async def rule_add(body: RuleIn):
        try:
            eng().rules.add(AP.Rule(body.effect, body.kind, body.value))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _mcp_view()

    @app.delete("/api/approval-rules/{kind}/{value}")
    async def rule_delete(kind: str, value: str):
        eng().rules.remove(kind, value)
        return _mcp_view()

    @app.post("/api/approvals/{aid}")
    async def approve(aid: str, body: ApprovalIn):
        if not eng().resolve_approval(aid, body.decision):
            raise HTTPException(404, "That request is no longer waiting.")
        return {"ok": True}

    # ------------------------------------------------------------ routines

    def _routine_view(r: dict) -> dict:
        runs = eng().db.routine_runs(r["id"], 1)
        return {**r, "label": RT.format_cron(r["cron"]), "preset": asdict(RT.preset_from_cron(r["cron"])),
                "last_run": runs[0] if runs else None}

    def _cron_of(body) -> str:
        cron = RT.cron_from_preset(RT.Preset(body.freq, body.n, body.unit, body.time, body.cron))
        try:
            if cron != RT.ONCE:
                RT.parse_cron(cron)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return cron

    @app.post("/api/routines/preview")
    async def routine_preview(body: PresetIn):
        cron = _cron_of(body)
        nxt, _ = schedule(cron)
        return {"cron": cron, "label": RT.format_cron(cron), "next_run": nxt}

    @app.get("/api/routines")
    async def routines_list():
        return [_routine_view(r) for r in eng().db.routines()]

    @app.post("/api/routines")
    async def routine_create(body: RoutineIn):
        cron = _cron_of(body)
        nxt, anchor = schedule(cron)
        rid = eng().db.save_routine({"name": body.name.strip()[:80] or "Routine", "prompt": body.prompt.strip(),
                                     "cron": cron, "enabled": body.enabled, "options": body.options,
                                     "anchor": anchor, "next_run": nxt if body.enabled else None})
        return _routine_view(eng().db.routine(rid))

    @app.put("/api/routines/{rid}")
    async def routine_update(rid: str, body: RoutineIn):
        r = eng().db.routine(rid)
        if not r:
            raise HTTPException(404, "Routine not found")
        cron = _cron_of(body)
        if cron != r["cron"]:
            r["next_run"], r["anchor"] = schedule(cron)
        elif body.enabled and not r["next_run"]:
            r["next_run"], r["anchor"] = schedule(cron, anchor=r.get("anchor"))
        r.update({"name": body.name.strip()[:80] or r["name"], "prompt": body.prompt.strip(), "cron": cron,
                  "enabled": body.enabled, "options": body.options})
        if not body.enabled:
            r["next_run"] = None
        eng().db.save_routine(r)
        return _routine_view(eng().db.routine(rid))

    @app.delete("/api/routines/{rid}")
    async def routine_delete(rid: str):
        eng().db.delete_routine(rid)
        return {"ok": True}

    @app.post("/api/routines/{rid}/run")
    async def routine_run_now(rid: str):
        r = eng().db.routine(rid)
        if not r:
            raise HTTPException(404, "Routine not found")
        return await state["scheduler"].run(r, manual=True)

    @app.get("/api/routines/{rid}/runs")
    async def routine_runs(rid: str):
        return eng().db.routine_runs(rid, 50)

    @app.get("/api/stats")
    async def stats():
        e = eng()
        lab = ST.lab_dir(e.cfg)
        ollama = e.router.providers.get("ollama")
        return {"generated": time.time(),
                "models": await ST.ollama_models(ollama.base, e.client) if ollama and e.client else {"installed": [], "loaded": []},
                "tiers": e.cfg.get("tiers", {}), "cloud_on": bool(e.cfg.get("defaults.allow_cloud")),
                "speed": ST.speed(e.db), "usage": e.db.usage(30), "training": ST.training_runs(lab),
                "tests": ST.tests(lab), "spending": ST.spending(lab), "machine": ST.machine(DATA_DIR),
                "app": ST.app_counts(e.db, e.agents)}

    @app.get("/api/lab")
    async def lab_data():
        return LAB.build(ST.lab_dir(eng().cfg), DATA_DIR)

    @app.get("/lab", response_class=HTMLResponse)
    async def lab_page():
        return (ROOT / "web" / "lab.html").read_text().replace("{{PRODUCT}}", html_escape(eng().cfg.product_name))

    @app.get("/api/usage")
    async def usage(days: int = 30):
        return {"days": days, "models": eng().db.usage(max(1, min(days, 365)))}

    @app.get("/api/agents")
    async def agents():
        return [a.public() for a in eng().agents.values()]

    def bot_call(fn, *args):
        e = eng()
        try:
            bid = fn(agents_dir(e.cfg), *args, e.agents)
        except BOTS.BotError as err:
            raise HTTPException(400, str(err)) from None
        e.load_agents()
        return e.agents[bid].public() if bid in e.agents else {"ok": True}

    @app.post("/api/agents")
    async def create_bot(body: BotIn):
        return bot_call(BOTS.save, body.model_dump())

    @app.put("/api/agents/{bid}")
    async def edit_bot(bid: str, body: BotIn):
        return bot_call(lambda folder, spec, agents: BOTS.save(folder, spec, agents, bid), body.model_dump())

    @app.delete("/api/agents/{bid}")
    async def delete_bot(bid: str):
        return bot_call(lambda folder, agents: BOTS.delete(folder, bid, agents) or bid)

    @app.get("/api/inbox")
    async def inbox_list(status: str | None = None):
        e = eng()
        return {"items": e.db.inbox(status), "sources": INBOX.status(), "last_check": e.db.kv_get("inbox.last_check")}

    @app.post("/api/inbox/check")
    async def inbox_check():
        try:
            return await INBOX.check(eng())
        except Exception as err:
            raise HTTPException(502, f"Could not check: {err}") from None

    def inbox_item(iid: str) -> dict:
        item = eng().db.inbox_get(iid)
        if not item:
            raise HTTPException(404, "No such message")
        return item

    @app.post("/api/inbox/{iid}/draft")
    async def inbox_redraft(iid: str):
        e, item = eng(), inbox_item(iid)
        t = await INBOX.triage(e.router, item, e.cfg.get("inbox.voice", ""))
        e.db.inbox_update(iid, draft=t["draft"], summary=t["summary"], priority=t["priority"], status="drafted")
        return e.db.inbox_get(iid)

    @app.post("/api/inbox/{iid}/send")
    async def inbox_send(iid: str, body: ReplyPostIn):
        """Send the reply you approved (your click is the approval)."""
        e, item = eng(), inbox_item(iid)
        if item["status"] == "sent":
            raise HTTPException(409, "Already answered")
        try:
            res = await INBOX.send_reply(e, item, body.text)
        except (ValueError, RuntimeError, KeyError) as err:
            raise HTTPException(400, str(err)) from None
        return {"result": res}

    @app.post("/api/inbox/{iid}/ignore")
    async def inbox_ignore(iid: str):
        inbox_item(iid)
        eng().db.inbox_update(iid, status="ignored")
        return {"ok": True}

    @app.get("/api/reports")
    async def reports_latest():
        e = eng()
        folder = REPORTS.vault_dir(e.cfg)
        return {"daily": e.db.kv_get("report.daily"), "weekly": e.db.kv_get("report.weekly"),
                "vault": str(folder) if folder else None, "email": bool(e.cfg.get("reports.email", False))}

    @app.post("/api/reports/{kind}")
    async def reports_make(kind: str):
        if kind not in ("daily", "weekly"):
            raise HTTPException(400, "daily or weekly")
        return await REPORTS.make(eng(), kind)

    @app.get("/api/integrations")
    async def integrations():
        day = time.time() - 86400
        return {**IN.status(), "reddit": REDDIT.ready(), "discord": INBOX.discord_can_send(),
                "x_posted_24h": eng().db.outbox_count_since("post_x", day),
                "reddit_posted_24h": eng().db.outbox_count_since("post_reddit", day),
                "emails_24h": eng().db.outbox_count_since("send_email", day)}

    @app.get("/api/outbox")
    async def outbox():
        return eng().db.outbox()

    @app.post("/api/replies/draft")
    async def draft_replies(body: ReplyDraftIn):
        e = eng()
        if body.platform not in IN.LENGTHS:
            raise HTTPException(400, "Platform must be x, linkedin or reddit")
        posts = IN.split_posts(body.text)
        if not posts:
            raise HTTPException(400, "Paste at least one post")
        out = []
        for item in posts:  # one at a time: a local model answers one request at a time
            prompt = IN.reply_prompt(item["post"], body.platform, body.goal.strip()[:300], body.voice.strip()[:200], body.product.strip()[:60])
            text = await e.router.complete("chat", [{"role": "system", "content": "You write short, genuine social media replies."},
                                                    {"role": "user", "content": prompt}], Params(temperature=0.7, max_tokens=220))
            text = IN.clean_reply(text)
            out.append({**item, "draft": text, "chars": len(text)})
        return out

    @app.post("/api/replies/post")
    async def post_reply(body: ReplyPostIn):
        try:
            if body.platform == "reddit":
                if not body.reply_to:
                    raise ValueError("Add the link of the Reddit post or comment to reply to")
                return {"result": await eng().outward("post_reddit", {"where": body.reply_to, "text": body.text})}
            return {"result": await eng().outward("post_x", {"text": body.text, "reply_to": body.reply_to})}
        except (ValueError, RuntimeError) as err:
            raise HTTPException(400, str(err)) from None

    @app.get("/api/market")
    async def market():
        return BOTS.market(eng().agents)

    @app.post("/api/market/{item}/install")
    async def install_bot(item: str):
        return bot_call(lambda folder, agents: BOTS.install(folder, item, agents))

    @app.get("/api/config")
    async def config():
        e = eng()
        return {"product_name": e.cfg.product_name, "personas": list(P.PERSONAS) + (list(P.ADULT_PERSONAS) if e.cfg.get("defaults.adult_mode", False) else []), "languages": P.LANGS,
                "commands": COMMANDS, "defaults": e.cfg.get("defaults"), "tiers": e.cfg.get("tiers"),
                "search": e.cfg.get("search"), "agents": [a.public() for a in e.agents.values()]}

    @app.post("/api/settings")
    async def settings(patch: dict):
        e = eng()
        e.cfg.save_settings(patch)
        e.reload()
        return {"ok": True}

    def this_computer(request: Request) -> None:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
        if AU.is_remote(request.headers, host, LOCAL_HOSTS) and not (
                os.environ.get("PROXMINO_REMOTE_KEYS") == "1" and auth.valid(request.cookies.get(AU.COOKIE))):
            # a home server has no screen of its own: PROXMINO_REMOTE_KEYS=1 lets you change keys after signing in
            raise HTTPException(403, "API keys can only be changed on the computer running Proxmino")

    @app.get("/api/keys")
    async def keys(request: Request):
        this_computer(request)
        return KEYS.status(env_path)

    @app.post("/api/keys")
    async def save_keys(body: KeysIn, request: Request):
        this_computer(request)
        try:
            changed = KEYS.save(env_path, body.values)
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        eng().reload()  # re-reads .env, so new keys work without a restart
        return {"changed": changed, "keys": KEYS.status(env_path)}

    @app.post("/api/keys/test")
    async def test_keys(request: Request):
        """Sign in to each service that has keys, without sending or changing anything."""
        this_computer(request)
        return await CONN.test_all(eng().client)

    @app.post("/api/reload")
    async def reload():
        """Re-read config.yaml, settings and .env (for example after adding an API key)."""
        eng().reload()
        return {"ok": True}

    @app.get("/api/health")
    async def health():
        e = eng()

        async def probe(p):
            try:
                models = await asyncio.wait_for(p.models(), 4)
                return {"name": p.name, "local": p.local, "ok": True, "models": models[:50]}
            except Exception as ex:
                return {"name": p.name, "local": p.local, "ok": False, "error": type(ex).__name__}

        provs = await asyncio.gather(*(probe(p) for p in e.router.providers.values()))
        for p in provs:
            p["enabled"] = e.router.allowed(e.router.providers[p["name"]])
        # Keyed providers, including ones skipped because their key is not set yet (never the key itself).
        cloud = [{"name": n, "key_env": pc["key_env"], "has_key": bool(os.environ.get(pc["key_env"]))}
                 for n, pc in (e.cfg.get("providers") or {}).items() if pc.get("key_env")]
        in_chat = {prov.name for prov, _ in e.router.chain("chat")}
        # Ready when an enabled, reachable provider can answer chat: named in the chat tier, or local with
        # an installed model (the router falls back to any installed local model).
        ready = any(p["ok"] and p["enabled"] and (p["name"] in in_chat or (p["local"] and p["models"])) for p in provs)
        return {"providers": provs, "cloud": cloud, "ready": ready,
                "allow_cloud": e.router.allow_cloud, "allow_local": e.router.allow_local,
                "tiers": {t: [f"{p.name}:{m}" for p, m in e.router.chain(t)] for t in e.router.tiers}}

    return app


app = create_app()
