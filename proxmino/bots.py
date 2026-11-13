"""Custom bots and the bot marketplace.

A custom bot is a YAML file in the user's agents folder (~/.proxmino/agents), the same format as the
built-in crew, so the agent loader picks it up. The marketplace is a catalog of ready-made bots in
market/*.yaml (written for Proxmino); "Get" copies one into the agents folder, where it can be edited.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from . import prompts as P
from .config import ROOT

MARKET = ROOT / "market"
MARKET_ONLY = {"category", "tagline", "featured"}
COLOR = re.compile(r"#[0-9a-fA-F]{6}")
TIERS = ("fast", "chat", "reason", "code")
SHAPES = 10  # avatar body families in the UI


class BotError(ValueError):
    pass


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:32].strip("-") or "bot"


def clean(spec: dict) -> dict:
    """Validated bot fields from the builder form or a marketplace entry."""
    name = str(spec.get("name") or "").strip()
    if not 1 <= len(name) <= 40:
        raise BotError("Give the bot a name (up to 40 characters).")
    out = {"name": name}
    color = str(spec.get("color") or "#8b8c93")
    if not COLOR.fullmatch(color):
        raise BotError("Color must look like #a1b2c3.")
    out["color"] = color.lower()
    shape = spec.get("shape")
    if shape is not None:
        if not isinstance(shape, int) or not 0 <= shape < SHAPES:
            raise BotError(f"Shape must be a number from 0 to {SHAPES - 1}.")
        out["shape"] = shape
    for key, limit in (("role", 60), ("description", 240), ("voice", 120), ("instructions", 4000)):
        v = str(spec.get(key) or "").strip()
        if len(v) > limit:
            raise BotError(f"{key.title()} is too long (up to {limit} characters).")
        if v:
            out[key] = v
    tier = spec.get("tier") or "chat"
    if tier not in TIERS:
        raise BotError("Tier must be fast, chat, reason or code.")
    out["tier"] = tier
    try:
        temp = float(spec.get("temperature", 0.5))
    except (TypeError, ValueError):
        raise BotError("Temperature must be a number.") from None
    if not 0 <= temp <= 1.2:
        raise BotError("Temperature must be between 0 and 1.2.")
    out["temperature"] = round(temp, 2)
    traits = spec.get("traits") or {}
    if not isinstance(traits, dict) or any(k not in P._TRAITS or not isinstance(v, (int, float)) or not 0 <= v <= 1
                                           for k, v in traits.items()):
        raise BotError("Traits are warmth, humor, bluntness, verbosity, curiosity or caution, each from 0 to 1.")
    if traits:
        out["traits"] = {k: round(float(v), 2) for k, v in traits.items()}
    for key in ("working_style", "never"):
        items = spec.get(key) or []
        if not isinstance(items, list) or len(items) > 8 or any(not isinstance(i, str) or len(i) > 160 for i in items):
            raise BotError(f"{key.replace('_', ' ').title()} must be up to 8 short lines.")
        if items:
            out[key] = [i.strip() for i in items if i.strip()]
    return out


def save(folder: Path, spec: dict, agents: dict, bot_id: str | None = None) -> str:
    """Create (bot_id None) or replace a custom bot; returns its id. Built-in bots cannot be changed."""
    data = clean(spec)
    if bot_id is None:
        base = slug(data["name"])
        bot_id, n = base, 2
        while bot_id in agents or (folder / f"{bot_id}.yaml").exists():
            bot_id, n = f"{base}-{n}", n + 1
    elif bot_id not in agents or agents[bot_id].builtin:
        raise BotError("Only your own bots can be edited.")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{bot_id}.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return bot_id


def delete(folder: Path, bot_id: str, agents: dict) -> None:
    if bot_id not in agents or agents[bot_id].builtin:
        raise BotError("Only your own bots can be deleted.")
    for f in folder.glob(f"{bot_id}.y*ml"):
        f.unlink()


def market(agents: dict) -> list[dict]:
    items = []
    for f in sorted(MARKET.glob("*.yaml")):
        spec = yaml.safe_load(f.read_text()) or {}
        bot = clean({k: v for k, v in spec.items() if k not in MARKET_ONLY})  # the full spec, so the builder can start from it
        items.append({**bot, "id": f.stem, "category": spec.get("category", "Other"), "tagline": spec.get("tagline", ""),
                      "featured": bool(spec.get("featured")), "installed": f.stem in agents})
    return items


def install(folder: Path, item_id: str, agents: dict) -> str:
    f = MARKET / f"{item_id}.yaml"
    if not re.fullmatch(r"[a-z0-9-]+", item_id) or not f.exists():
        raise BotError("No such bot in the marketplace.")
    if item_id in agents:
        return item_id
    spec = {k: v for k, v in (yaml.safe_load(f.read_text()) or {}).items() if k not in MARKET_ONLY}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{item_id}.yaml").write_text(yaml.safe_dump(clean(spec), sort_keys=False, allow_unicode=True))
    return item_id
