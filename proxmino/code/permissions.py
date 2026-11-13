"""Permission policy for coding tools. Enforced in code, not in the prompt.

Modes:
  plan       read-only; writes and commands are denied
  ask        reads are free; every write and command needs approval
  auto-edit  reads, edits and the test command are free; other commands need approval
  auto       reads, edits and allowlisted commands are free; other commands need approval
The denylist is checked first in every mode, and commands that touch the network, install
packages, delete files or push always ask (the sandbox also blocks network and outside writes).
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

MODES = ("plan", "ask", "auto-edit", "auto")
ALLOW, ASK, DENY = "allow", "ask", "deny"

DEFAULT_ALLOW = [
    r"(python3?|\.venv/bin/python) -m (pytest|unittest|mypy|ruff|pyflakes|compileall)\b.*",
    r"(pytest|ruff|mypy|flake8|black --check|eslint|tsc --noEmit|prettier --check)\b.*",
    r"npm (test|run (test|lint|build|typecheck))\b.*", r"(go (test|vet|build)|cargo (test|check|build|clippy))\b.*",
    r"git (status|diff|log|show|branch|blame)\b.*",
    r"(ls|cat|head|tail|wc|pwd|tree|file|which|echo)\b[^|;&`$><]*",
    r"(python3?|node) --version",
]
DEFAULT_DENY = [
    r"\bsudo\b", r"\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME|\*|\.\.?/?)(\s|$)", r"\bmkfs\b", r"\bdd\s+if=",
    r":\(\)\s*\{", r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b", r"\bchmod\s+-R\s+777\s+/", r">\s*/dev/(disk|sd)",
    r"\bgit\s+push\s+.*--force\b", r"\blaunchctl\b", r"\bcrontab\b", r"\bosascript\b", r"\bsecurity\s+(find|dump)",
]
ALWAYS_ASK = [
    r"\brm\b", r"\bgit\s+(push|reset\s+--hard|clean|checkout\s+--|rebase)\b", r"\b(pip3?|uv pip|npm|pnpm|yarn|brew|cargo)\s+(install|add|i)\b",
    r"\b(curl|wget|ssh|scp|rsync|nc)\b", r"\bmv\b", r"\bchmod\b",
]
CLASSES = {
    "read_file": "read", "list_dir": "read", "glob": "read", "grep": "read", "repo_map": "read",
    "edit_file": "write", "write_file": "write",
    "run": "exec", "run_tests": "exec",
    "todo": "meta", "ask_user": "meta",
}


@dataclass
class Decision:
    action: str
    reason: str


@dataclass
class Policy:
    mode: str = "ask"
    allow: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOW))
    deny: list[str] = field(default_factory=lambda: list(DEFAULT_DENY))
    remembered: set[str] = field(default_factory=set)  # commands the user approved for this session

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")

    @classmethod
    def from_config(cls, cfg, mode: str | None = None) -> "Policy":
        return cls(mode=mode or cfg.get("code.mode", "ask"),
                   allow=DEFAULT_ALLOW + list(cfg.get("code.allow", []) or []),
                   deny=DEFAULT_DENY + list(cfg.get("code.deny", []) or []))

    def _match(self, patterns: list[str], command: str) -> str | None:
        for p in patterns:
            if re.search(p, command):
                return p
        return None

    def _allowlisted(self, command: str) -> bool:
        # Every part of a compound command must be allowlisted on its own.
        parts = [p.strip() for p in re.split(r"&&|\|\||;|\|", command) if p.strip()]
        return bool(parts) and all(any(re.fullmatch(a, p) for a in self.allow) for p in parts)

    def check(self, tool: str, args: dict) -> Decision:
        cls = CLASSES.get(tool)
        if cls is None:
            return Decision(DENY, f"unknown tool {tool}")
        if cls in ("read", "meta"):
            return Decision(ALLOW, "read-only")
        if self.mode == "plan":
            return Decision(DENY, "plan mode is read-only; finish the plan instead")
        if cls == "write":
            return Decision(ASK if self.mode == "ask" else ALLOW, f"{self.mode} mode")
        command = "run_tests" if tool == "run_tests" else str(args.get("command", ""))
        if tool == "run":
            try:
                shlex.split(command)
            except ValueError:
                return Decision(DENY, "command does not parse (unbalanced quotes)")
            hit = self._match(self.deny, command)
            if hit:
                return Decision(DENY, f"blocked by deny rule {hit!r}")
            if self._match(ALWAYS_ASK, command):
                return Decision(ALLOW if command in self.remembered else ASK, "deletes, moves, installs, pushes or uses the network")
        if self.mode == "ask":
            return Decision(ALLOW if command in self.remembered else ASK, "ask mode")
        if tool == "run_tests" or command in self.remembered:
            return Decision(ALLOW, "tests" if tool == "run_tests" else "approved earlier")
        if self.mode == "auto" and self._allowlisted(command):
            return Decision(ALLOW, "allowlisted")
        return Decision(ASK, "not on the allowlist")
