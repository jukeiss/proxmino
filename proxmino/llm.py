"""LLM providers, the tier router with fallback chains, and stream helpers."""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx

log = logging.getLogger("proxmino.llm")

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class ProviderError(Exception):
    pass


@dataclass
class Params:
    temperature: float = 0.6
    max_tokens: int | None = None
    think: bool = False
    usage: dict | None = None  # filled by the provider when it reports tokens: {"in": n, "out": n}


class Provider:
    name = "base"
    local = True

    def stream(self, messages: list[Message], model: str, p: Params) -> AsyncIterator[str]:
        raise NotImplementedError

    async def models(self) -> list[str]:
        return []


class OpenAICompatProvider(Provider):
    """Any server exposing /v1/chat/completions (LM Studio, OpenAI, OpenRouter, Groq...)."""

    def __init__(self, name: str, base_url: str, key: str | None, local: bool, client: httpx.AsyncClient):
        self.name, self.base, self.key, self.local, self.client = name, base_url.rstrip("/"), key, local, client

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    async def stream(self, messages, model, p):
        body = {"model": model, "messages": messages, "stream": True, "temperature": p.temperature}
        if p.max_tokens:
            body["max_tokens"] = p.max_tokens
        if not self.local:  # OpenAI, OpenRouter and Groq report token counts in a final chunk when asked
            body["stream_options"] = {"include_usage": True}
        if "deepseek.com" in self.base:  # DeepSeek thinks by default, which made every quick reply slow: think only when asked
            body["thinking"] = {"type": "enabled" if p.think else "disabled"}
            if p.think and not p.max_tokens:  # thinking counts against the output limit: leave room for long code after it
                body["max_tokens"] = 32000
        try:
            # a thinking model can work silently for minutes on a hard proof: give it 15 minutes instead of 2
            wait = httpx.Timeout(900, connect=10) if p.think else httpx.USE_CLIENT_DEFAULT
            async with self.client.stream("POST", f"{self.base}/chat/completions",
                                          json=body, headers=self._headers(), timeout=wait) as r:
                if r.status_code >= 400:
                    raise ProviderError(f"{self.name} HTTP {r.status_code}: {(await r.aread())[:300]!r}")
                in_reasoning = False
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    u = obj.get("usage") if isinstance(obj, dict) else None
                    if u and p.usage is not None:
                        p.usage.update({"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)})
                    try:
                        delta = obj["choices"][0].get("delta", {})
                    except (KeyError, IndexError, TypeError):
                        continue
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        if not in_reasoning:
                            in_reasoning = True
                            yield THINK_OPEN
                        # the model may write "</think>" inside its own reasoning; that must not end the thinking early
                        yield reasoning.replace(THINK_OPEN, "‹think›").replace(THINK_CLOSE, "‹/think›")
                    if delta.get("content"):
                        if in_reasoning:
                            in_reasoning = False
                            yield THINK_CLOSE
                        yield delta["content"]
                if in_reasoning:
                    yield THINK_CLOSE
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}: {e}") from e

    async def models(self):
        r = await self.client.get(f"{self.base}/models", headers=self._headers(), timeout=5)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]


class OllamaProvider(Provider):
    """Native Ollama API: lets us keep models loaded and switch thinking on or off."""

    def __init__(self, name: str, base_url: str, client: httpx.AsyncClient, keep_alive: str = "30m",
                 key: str | None = None, local: bool = True, num_ctx: int = 8192):
        self.name, self.base, self.client, self.keep_alive = name, base_url.rstrip("/"), client, keep_alive
        self.local = local  # False for ollama.com: prompts leave this computer
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.num_ctx = num_ctx  # context window: memory for it is reserved up front, so smaller is faster and lighter
        self.no_think: set[str] = set()  # models that reject the `think` field

    async def stream(self, messages, model, p):
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": self.keep_alive,
                "options": {"temperature": p.temperature, "num_ctx": self.num_ctx}}  # no repeat penalty: it bends repeated numbers ("2022" stopped at "20", $1,200 became $12,000)
        if p.max_tokens:
            body["options"]["num_predict"] = p.max_tokens
        if model not in self.no_think:
            body["think"] = p.think
        retry = False
        try:
            async with self.client.stream("POST", f"{self.base}/api/chat", json=body, headers=self.headers) as r:
                if r.status_code >= 400:
                    err = (await r.aread()).decode(errors="replace")
                    if "think" not in err or "think" not in body:
                        raise ProviderError(f"{self.name} HTTP {r.status_code}: {err[:300]}")
                    self.no_think.add(model)
                    retry = True
                in_thinking = False
                async for line in _no_lines() if retry else r.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise ProviderError(f"{self.name}: {data['error']}")
                    msg = data.get("message", {})
                    if msg.get("thinking"):
                        if not in_thinking:
                            in_thinking = True
                            yield THINK_OPEN
                        yield msg["thinking"]
                    if msg.get("content"):
                        if in_thinking:
                            in_thinking = False
                            yield THINK_CLOSE
                        yield msg["content"]
                    if data.get("done") and p.usage is not None:
                        ns = 1e6  # Ollama reports durations in nanoseconds
                        p.usage.update({"in": data.get("prompt_eval_count", 0), "out": data.get("eval_count", 0),
                                        "load_ms": round((data.get("load_duration") or 0) / ns),
                                        "prompt_ms": round((data.get("prompt_eval_duration") or 0) / ns),
                                        "gen_ms": round((data.get("eval_duration") or 0) / ns)})
                    if data.get("done"):
                        break
                if in_thinking:
                    yield THINK_CLOSE
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}: {e}") from e
        if retry:
            async for chunk in self.stream(messages, model, p):
                yield chunk

    async def models(self):
        r = await self.client.get(f"{self.base}/api/tags", timeout=5, headers=self.headers)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


class FakeProvider(Provider):
    """Deterministic provider for tests. `responder(messages) -> str`."""

    name = "fake"

    def __init__(self, responder: Callable[[list[Message]], str] | None = None):
        self.responder = responder or (lambda m: f"Echo: {m[-1]['content']}")
        self.calls: list[list[Message]] = []

    async def stream(self, messages, model, p):
        self.calls.append(messages)
        text = self.responder(messages)
        for part in re.findall(r"\S+\s*|\s+", text):
            yield part

    async def models(self):
        return ["fake-model"]


class DemoProvider(Provider):
    """Used when no real model is reachable, so the app still explains itself."""

    name = "demo"

    async def stream(self, messages, model, p):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if "Return JSON" in system:
            yield "{}"
            return
        text = ("No language model is reachable, so this is a placeholder reply.\n\n"
                "To get real answers, install Ollama from https://ollama.com and pull a model, "
                "for example `ollama pull phi4-mini`. Or add an API key in `.env` and enable cloud "
                "providers in Settings.")
        sources = re.findall(r'<source id="(\d+)" url="([^"]+)" title="([^"]*)"', "\n".join(
            m["content"] for m in messages))
        if sources:
            text += "\n\nWeb search still works without a model. Sources found:\n" + "\n".join(
                f"- {title or url} [{n}]" for n, url, title in sources)
        for part in re.findall(r"\S+\s*|\s+", text):
            yield part


def is_cloud_model(model: str) -> bool:
    """Ollama cloud models (name:cloud, name:<tag>-cloud) run on ollama.com, not this computer."""
    return model.endswith(":cloud") or model.endswith("-cloud")


class Router:
    """Resolves a tier to the first working provider:model in its chain."""

    def __init__(self, providers: dict[str, Provider], tiers: dict[str, list[str]],
                 allow_cloud: bool = False, demo_fallback: bool = True, allow_local: bool = True):
        self.providers, self.tiers = providers, tiers
        self.allow_cloud, self.demo_fallback, self.allow_local = allow_cloud, demo_fallback, allow_local
        self.down_until: dict[str, float] = {}
        self.missing: dict[tuple[str, str], float] = {}  # (provider, model) -> retry after
        self._installed: dict[str, tuple[float, list[str]]] = {}

    @classmethod
    def from_config(cls, cfg, client: httpx.AsyncClient) -> "Router":
        providers: dict[str, Provider] = {}
        for name, pc in (cfg.get("providers") or {}).items():
            if pc.get("type") == "ollama":
                key = os.environ.get(pc["key_env"]) if pc.get("key_env") else None
                if pc.get("key_env") and not key:
                    continue  # ollama.com without a key: skip silently
                local = pc.get("local", True)
                url = (os.environ.get("OLLAMA_URL") if local else None) or pc["base_url"]  # Docker: http://ollama:11434
                providers[name] = OllamaProvider(name, url, client, str(pc.get("keep_alive", "2m")), key, local,
                                                 int(pc.get("num_ctx", 8192)))
            else:
                key = os.environ.get(pc["key_env"]) if pc.get("key_env") else None
                if pc.get("key_env") and not key:
                    continue  # keyed provider without a key: skip silently
                providers[name] = OpenAICompatProvider(name, pc["base_url"], key, pc.get("local", False), client)
        tiers = dict(cfg.get("tiers", {}) or {})
        mode = cfg.get("defaults.brain", "cloud")
        if mode in ("proxmino", "hybrid"):
            # proxmino: your own model answers everything. hybrid: the fast cloud model answers everything you see
            # (chat, emails, posts, inbox drafts, hard work) and your model does the small background jobs (titles,
            # search wording); when the cloud fails or runs out of credit, your model answers instead.
            own = cfg.get("defaults.brain_model", "ollama:proxmino:8b")
            first = set(tiers) if mode == "proxmino" else {"fast"}
            tiers = {t: ([own] + [r for r in chain if r != own]) if t in first else ([r for r in chain if r != own] + [own])
                     for t, chain in tiers.items()}
        return cls(providers, tiers, cfg.get("defaults.allow_cloud", False),
                   cfg.get("defaults.demo_fallback", True), cfg.get("defaults.allow_local", True))

    def chain(self, tier: str) -> list[tuple[Provider, str]]:
        out = []
        for ref in self.tiers.get(tier) or self.tiers.get("chat") or []:
            pname, _, model = ref.partition(":")
            prov = self.providers.get(pname)
            if prov and self.allowed(prov) and (self.allow_cloud or not is_cloud_model(model)):
                out.append((prov, model))
        return out

    def pinned(self, ref: str) -> "Router":
        """A view of this router that tries `provider:model` first in every tier, then the usual chain.
        Shares providers and failure caches with the original."""
        view = copy.copy(self)
        view.tiers = {t: [ref] + [r for r in chain if r != ref] for t, chain in self.tiers.items()}
        for t in ("fast", "chat", "reason", "code"):
            view.tiers.setdefault(t, [ref])
        return view

    def allowed(self, prov: Provider) -> bool:
        """Local providers use this computer's CPU/GPU; cloud providers send prompts off the machine. Each is opt-in."""
        return self.allow_local if prov.local else self.allow_cloud

    async def installed(self, prov: Provider) -> list[str]:
        hit = self._installed.get(prov.name)
        if hit and time.time() - hit[0] < 60:
            return hit[1]
        try:
            models = await prov.models()
        except Exception:
            models = []
        self._installed[prov.name] = (time.time(), models)
        return models

    async def _candidates(self, tier: str) -> AsyncIterator[tuple[Provider, str]]:
        tried = set()
        for prov, model in self.chain(tier):
            tried.add((prov.name, model))
            yield prov, model
        # Last resort: any chat model installed on a local provider.
        for prov in self.providers.values():
            if not prov.local or not self.allow_local or self.down_until.get(prov.name, 0) > time.time():
                continue
            for model in await self.installed(prov):
                if (prov.name, model) not in tried and "embed" not in model:
                    yield prov, model
                    break

    async def open(self, tier: str, messages: list[Message], p: Params | None = None) -> tuple[str, AsyncIterator[str]]:
        """Start a stream on the first working candidate. Returns (model_label, chunks)."""
        p = p or Params()
        errors = []
        async for prov, model in self._candidates(tier):
            if self.down_until.get(prov.name, 0) > time.time() or self.missing.get((prov.name, model), 0) > time.time():
                continue
            it = prov.stream(messages, model, p).__aiter__()
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                return f"{prov.name}:{model}", _empty()
            except ProviderError as e:
                errors.append(str(e))
                if "Connect" in str(e) or ("Timeout" in str(e) and not prov.local):  # a busy local model is not down
                    self.down_until[prov.name] = time.time() + 20
                elif "402" in str(e) or "401" in str(e) or "credits" in str(e):
                    self.down_until[prov.name] = time.time() + 600  # no credits or a bad key: skip it for 10 minutes
                elif "404" in str(e) or "not found" in str(e):
                    self.missing[(prov.name, model)] = time.time() + 300
                continue
            return f"{prov.name}:{model}", _prepend(first, it)
        if errors:
            log.warning("all providers failed for tier %s: %s", tier, errors)
        if not self.demo_fallback:
            raise ProviderError("No model is reachable. " + "; ".join(errors[-3:]))
        return "demo", DemoProvider().stream(messages, "demo", p)

    async def complete(self, tier: str, messages: list[Message], p: Params | None = None) -> str:
        _, it = await self.open(tier, messages, p)
        return strip_think("".join([c async for c in it]))


async def _prepend(first: str, it: AsyncIterator[str]) -> AsyncIterator[str]:
    yield first
    async for c in it:
        yield c


async def _no_lines():
    return
    yield  # pragma: no cover


async def _empty():
    return
    yield  # pragma: no cover


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


class ThinkSplitter:
    """Splits a token stream into ("thinking", text) and ("token", text) parts.

    Handles <think> tags split across chunk boundaries.
    """

    def __init__(self):
        self.buf, self.thinking = "", False

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        self.buf += chunk
        out: list[tuple[str, str]] = []
        while self.buf:
            tag = THINK_CLOSE if self.thinking else THINK_OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    out.append(("thinking" if self.thinking else "token", self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.thinking = not self.thinking
                continue
            # keep a possible partial tag at the end of the buffer
            keep = 0
            for k in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-k:]):
                    keep = k
                    break
            emit = self.buf[:-keep] if keep else self.buf
            if emit:
                out.append(("thinking" if self.thinking else "token", emit))
            self.buf = self.buf[-keep:] if keep else ""
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        out = [("thinking" if self.thinking else "token", self.buf)] if self.buf else []
        self.buf = ""
        return out


def parse_json(text: str, default):
    """Pull the first JSON object or array out of model output; weak models wrap it in prose."""
    text = strip_think(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return default
