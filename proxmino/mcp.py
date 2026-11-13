"""MCP client: connects Proxmino to Model Context Protocol servers (local programs or remote HTTP).

Safety rules follow Rakazo's packages/adapters/src/mcp-transport.ts (Apache-2.0, see
third_party/rakazo/NOTICE.md): local servers start only from an allowlist of launch commands,
remote URLs must be HTTPS (plain HTTP only on this computer) with no credentials or fragment,
every request has a timeout, and the tail of a server's error output is kept for messages.
The protocol client itself is written here (JSON-RPC 2.0), with no extra dependency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger("proxmino.mcp")
PROTOCOL = "2025-06-18"
CLIENT = {"name": "proxmino", "version": "0.1"}
DEFAULT_ALLOWED = ["npx", "uvx", "node", "python3", "python", "deno", "bunx"]
STDERR_TAIL = 4096
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class MCPError(Exception):
    pass


def slug(name: str) -> str:
    """Server name as a tool prefix (Rakazo's deriveMcpSlug)."""
    s = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("_-")[:64]
    return s or "mcp"


@dataclass
class ServerConfig:
    name: str
    transport: str = "stdio"  # stdio | http
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @property
    def id(self) -> str:
        return slug(self.name)

    def public(self) -> dict:
        """For the UI: header and env values are hidden (they often hold tokens)."""
        return {"name": self.name, "id": self.id, "transport": self.transport, "command": self.command, "args": self.args,
                "url": self.url, "enabled": self.enabled, "env_keys": sorted(self.env), "header_keys": sorted(self.headers)}


def validate_url(raw: str) -> str:
    u = urlparse(raw.strip())
    if len(raw) > 2048:
        raise MCPError("The URL is longer than 2048 characters.")
    if u.username or u.password or u.fragment:
        raise MCPError("The URL must not contain a user name, password or #fragment. Put tokens in headers.")
    local = u.hostname in ("localhost", "127.0.0.1", "::1")
    if u.scheme != "https" and not (u.scheme == "http" and local):
        raise MCPError("Remote MCP servers must use https:// (plain http:// only for this computer).")
    return raw.strip()


def validate_command(cmd: str, allowed: list[str]) -> str:
    base = os.path.basename(cmd.strip())
    if base not in allowed:
        raise MCPError(f"'{base}' is not an allowed launch command. Allowed: {', '.join(allowed)} (Settings > Tools).")
    path = shutil.which(cmd.strip()) or (cmd if os.path.isabs(cmd) and os.access(cmd, os.X_OK) else None)
    if not path:
        raise MCPError(f"'{cmd}' was not found on this computer.")
    return path


class _Client:
    def __init__(self, cfg: ServerConfig, timeout: float):
        self.cfg, self.timeout = cfg, timeout
        self.next_id = 0
        self.info: dict = {}

    def _req(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        return {"jsonrpc": "2.0", "id": self.next_id, "method": method, **({"params": params} if params is not None else {})}

    async def initialize(self) -> None:
        res = await self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": CLIENT})
        self.info = res.get("serverInfo", {})
        await self.notify("notifications/initialized")

    async def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        for _ in range(20):  # pages
            res = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += res.get("tools", [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(self, name: str, args: dict) -> dict:
        return await self.request("tools/call", {"name": name, "arguments": args})


class StdioClient(_Client):
    """A local server: newline-delimited JSON-RPC over the process's stdin and stdout."""

    def __init__(self, cfg: ServerConfig, allowed: list[str], timeout: float = 30):
        super().__init__(cfg, timeout)
        self.allowed = allowed
        self.proc: asyncio.subprocess.Process | None = None
        self.stderr = ""
        self.pending: dict[int, asyncio.Future] = {}
        self.reader: asyncio.Task | None = None
        self.err_reader: asyncio.Task | None = None

    async def start(self) -> None:
        exe = validate_command(self.cfg.command, self.allowed)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", ""), **self.cfg.env}
        self.proc = await asyncio.create_subprocess_exec(
            exe, *self.cfg.args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env, limit=8 * 1024 * 1024, start_new_session=True)
        self.reader = asyncio.ensure_future(self._read())
        self.err_reader = asyncio.ensure_future(self._read_err())
        try:
            await self.initialize()
        except Exception as e:
            await self.close()
            raise MCPError(f"{self.cfg.name} did not start: {e}" + (f"\n{self.tail()}" if self.tail() else "")) from None

    def tail(self) -> str:
        return ANSI.sub("", self.stderr).strip()[-STDERR_TAIL:]

    async def _read(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # servers sometimes print logs to stdout
            fut = self.pending.pop(msg.get("id"), None) if isinstance(msg, dict) else None
            if fut and not fut.done():
                fut.set_result(msg)
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(MCPError(f"{self.cfg.name} exited." + (f"\n{self.tail()}" if self.tail() else "")))
        self.pending.clear()

    async def _read_err(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            chunk = await self.proc.stderr.read(4096)
            if not chunk:
                break
            self.stderr = (self.stderr + chunk.decode(errors="replace"))[-STDERR_TAIL * 2:]

    async def request(self, method: str, params: dict | None = None) -> dict:
        if not self.proc or self.proc.returncode is not None:
            raise MCPError(f"{self.cfg.name} is not running.")
        msg = self._req(method, params)
        fut = asyncio.get_running_loop().create_future()
        self.pending[msg["id"]] = fut
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self.proc.stdin.drain()
        try:
            res = await asyncio.wait_for(fut, self.timeout)
        except asyncio.TimeoutError:
            self.pending.pop(msg["id"], None)
            raise MCPError(f"{self.cfg.name} did not answer {method} within {self.timeout:.0f}s.") from None
        if "error" in res:
            raise MCPError(f"{self.cfg.name}: {res['error'].get('message', res['error'])}")
        return res.get("result", {})

    async def notify(self, method: str) -> None:
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": method}) + "\n").encode())
        await self.proc.stdin.drain()

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), 3)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
        for t in (self.reader, self.err_reader):
            if t:
                t.cancel()


class HttpClient(_Client):
    """A remote server over Streamable HTTP: POST JSON-RPC, answer as JSON or as an SSE stream."""

    def __init__(self, cfg: ServerConfig, client: httpx.AsyncClient, timeout: float = 30):
        super().__init__(cfg, timeout)
        self.url = validate_url(cfg.url)
        self.http = client
        self.session: str | None = None

    async def start(self) -> None:
        try:
            await self.initialize()
        except httpx.HTTPError as e:
            raise MCPError(f"{self.cfg.name}: {type(e).__name__}: {e}") from None

    def _headers(self) -> dict:
        h = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
             "MCP-Protocol-Version": PROTOCOL, **self.cfg.headers}
        if self.session:
            h["Mcp-Session-Id"] = self.session
        return h

    async def request(self, method: str, params: dict | None = None) -> dict:
        msg = self._req(method, params)
        r = await self.http.post(self.url, json=msg, headers=self._headers(), timeout=self.timeout, follow_redirects=False)
        if r.status_code >= 400:
            raise MCPError(f"{self.cfg.name}: HTTP {r.status_code} {r.text[:200]}")
        self.session = r.headers.get("mcp-session-id", self.session)
        res = None
        if "text/event-stream" in r.headers.get("content-type", ""):
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    try:
                        d = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if isinstance(d, dict) and d.get("id") == msg["id"]:
                        res = d
        else:
            res = r.json()
        if not isinstance(res, dict):
            raise MCPError(f"{self.cfg.name} sent no answer to {method}.")
        if "error" in res:
            raise MCPError(f"{self.cfg.name}: {res['error'].get('message', res['error'])}")
        return res.get("result", {})

    async def notify(self, method: str) -> None:
        await self.http.post(self.url, json={"jsonrpc": "2.0", "method": method}, headers=self._headers(), timeout=self.timeout)

    async def close(self) -> None:
        if self.session:
            try:
                await self.http.delete(self.url, headers=self._headers(), timeout=5)
            except httpx.HTTPError:
                pass


def result_text(res: dict, limit: int = 12000) -> str:
    """A tool result as text for the model: text parts kept, other parts named."""
    parts = []
    for c in res.get("content", []):
        if c.get("type") == "text":
            parts.append(c.get("text", ""))
        elif c.get("type") == "resource":
            r = c.get("resource", {})
            parts.append(r.get("text") or f"[resource {r.get('uri', '')}]")
        else:
            parts.append(f"[{c.get('type', 'content')} omitted]")
    if not parts and "structuredContent" in res:
        parts.append(json.dumps(res["structuredContent"])[:limit])
    text = "\n".join(parts).strip()
    if res.get("isError"):
        text = "Tool error: " + text
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters)"


class Manager:
    """Server settings (data/mcp.json), live connections and the combined tool list."""

    def __init__(self, path: Path, http: httpx.AsyncClient | None, allowed: list[str] | None = None, timeout: float = 30):
        self.path, self.http, self.timeout = path, http, timeout
        self.allowed = allowed or list(DEFAULT_ALLOWED)
        self.clients: dict[str, _Client] = {}
        self.tools: dict[str, list[dict]] = {}
        self.errors: dict[str, str] = {}
        self.lock = asyncio.Lock()

    def servers(self) -> list[ServerConfig]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return []
        return [ServerConfig(**s) for s in raw.get("servers", [])]

    def save(self, servers: list[ServerConfig]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"servers": [s.__dict__ for s in servers]}, indent=2))
        try:
            os.chmod(self.path, 0o600)  # may hold tokens in headers or env
        except OSError:
            pass

    def upsert(self, cfg: ServerConfig) -> None:
        if cfg.transport == "http":
            validate_url(cfg.url)
        elif cfg.transport == "stdio":
            if os.path.basename(cfg.command.strip()) not in self.allowed:
                raise MCPError(f"'{cfg.command}' is not an allowed launch command. Allowed: {', '.join(self.allowed)}.")
        else:
            raise MCPError("transport must be stdio or http")
        servers = [s for s in self.servers() if s.id != cfg.id]
        old = next((s for s in self.servers() if s.id == cfg.id), None)
        if old:  # keep hidden values the UI did not resend
            cfg.env = {**old.env, **cfg.env}
            cfg.headers = {**old.headers, **cfg.headers}
        self.save(servers + [cfg])

    def remove(self, sid: str) -> None:
        self.save([s for s in self.servers() if s.id != sid])

    async def connect(self, cfg: ServerConfig) -> _Client:
        async with self.lock:
            c = self.clients.get(cfg.id)
            if c:
                return c
            c = StdioClient(cfg, self.allowed, self.timeout) if cfg.transport == "stdio" else HttpClient(cfg, self.http, self.timeout)
            await c.start()
            self.clients[cfg.id] = c
            return c

    async def refresh(self) -> dict[str, list[dict]]:
        """Connect to every enabled server and list its tools. Failures are kept per server."""
        await self.close()
        self.tools, self.errors = {}, {}
        for cfg in self.servers():
            if not cfg.enabled:
                continue
            try:
                c = await self.connect(cfg)
                self.tools[cfg.id] = await c.list_tools()
            except Exception as e:
                self.errors[cfg.id] = str(e)
                log.warning("mcp %s: %s", cfg.name, e)
        return self.tools

    def catalog(self) -> list[dict]:
        """Every tool as {"name": "<server>__<tool>", ...}, the name the model calls."""
        out = []
        for sid, tools in self.tools.items():
            for t in tools:
                out.append({"name": f"{sid}__{t['name']}", "server": sid, "tool": t["name"],
                            "description": (t.get("description") or "").strip()[:300],
                            "schema": t.get("inputSchema") or {}, "annotations": t.get("annotations") or {}})
        return out

    async def call(self, full_name: str, args: dict) -> str:
        sid, _, tool = full_name.partition("__")
        cfg = next((s for s in self.servers() if s.id == sid and s.enabled), None)
        if not cfg or not tool:
            raise MCPError(f"Unknown tool {full_name}.")
        c = await self.connect(cfg)
        return result_text(await c.call_tool(tool, args))

    async def close(self) -> None:
        for c in list(self.clients.values()):
            await c.close()
        self.clients.clear()
