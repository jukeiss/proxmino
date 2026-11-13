"""When a tool call needs the user's approval.

Translated from Rakazo's packages/core/src/action-approval.ts (Apache-2.0, see
third_party/rakazo/NOTICE.md): connector tools are judged by their name (reads such as get, list,
search, find and read are free; anything that sends, deletes, pays, writes or does several things
asks), a declared read-only hint never relaxes the name check because the provider chooses both,
a declared write always asks, and user rules can always allow, ask or deny a tool or a whole server.
Changes: Python, and only the tool and server rule kinds (Proxmino has no sites or categories yet).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

READ_ONLY = re.compile(r"(^|_)(get|list|search|find|read)(_|$)", re.I)
MUTATING = re.compile(
    r"(^|_)(accept|add|approve|archive|assign|buy|cancel|charge|checkout|close|commit|copy|create|delete|deploy|disable|"
    r"enable|execute|forward|grant|invite|link|mark|merge|modify|move|patch|pay|post|publish|purchase|put|register|reject|"
    r"remove|rename|replace|reply|reset|revoke|run|schedule|send|set|share|start|stop|submit|subscribe|trigger|unassign|"
    r"unlink|unsubscribe|update|upload|upsert|write)(_|$)", re.I)
COMPOUND = re.compile(r"_(and|or|then)_", re.I)
EFFECTS = ("always_allow", "ask", "deny")


def words(name: str) -> str:
    """camelCase and dashes to snake_case, so readFile and read-file match like read_file."""
    return re.sub(r"[-\s.]+", "_", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)).lower()


def tool_requires_approval(tool: str, read_only: bool | None = None) -> bool:
    name = words(tool)
    if read_only is False:
        return True
    if MUTATING.search(name) or COMPOUND.search(name):
        return True
    return not READ_ONLY.search(name)


@dataclass
class Rule:
    effect: str  # always_allow | ask | deny
    kind: str    # tool | server
    value: str   # "<server>__<tool>" or "<server>"


def decide(full_name: str, read_only: bool | None, rules: list[Rule]) -> tuple[str, str]:
    """("allow" | "ask" | "deny", reason). A tool rule beats a server rule; deny beats the rest."""
    server = full_name.partition("__")[0]
    for kind, value in (("tool", full_name), ("server", server)):
        hits = [r for r in rules if r.kind == kind and r.value == value]
        if any(r.effect == "deny" for r in hits):
            return "deny", f"blocked by your rule for {value}"
        if any(r.effect == "ask" for r in hits):
            return "ask", f"your rule for {value} asks first"
        if any(r.effect == "always_allow" for r in hits):
            return "allow", f"always allowed for {value}"
    if tool_requires_approval(full_name.partition("__")[2] or full_name, read_only):
        return "ask", "this tool can change data or send something"
    return "allow", "read-only tool"


class Rules:
    def __init__(self, path: Path):
        self.path = path

    def all(self) -> list[Rule]:
        if not self.path.exists():
            return []
        try:
            return [Rule(**r) for r in json.loads(self.path.read_text())]
        except (json.JSONDecodeError, TypeError):
            return []

    def add(self, rule: Rule) -> None:
        if rule.effect not in EFFECTS or rule.kind not in ("tool", "server") or not rule.value:
            raise ValueError("A rule needs an effect (always_allow, ask, deny), a kind (tool, server) and a value.")
        rules = [r for r in self.all() if not (r.kind == rule.kind and r.value == rule.value)] + [rule]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in rules], indent=2))

    def remove(self, kind: str, value: str) -> None:
        rules = [r for r in self.all() if not (r.kind == kind and r.value == value)]
        self.path.write_text(json.dumps([asdict(r) for r in rules], indent=2))
