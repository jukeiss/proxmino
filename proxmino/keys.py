"""API keys in .env, edited from Settings > API keys. Values are written to .env (owner-only permissions) and never
sent back to the browser: the page only learns whether each secret is set."""
from __future__ import annotations

import os
import re
from pathlib import Path

# (name, label, group, secret)
KEYS = [
    ("DEEPSEEK_API_KEY", "DeepSeek", "Models", True),
    ("OLLAMA_API_KEY", "Ollama cloud", "Models", True),
    ("OPENAI_API_KEY", "OpenAI", "Models", True),
    ("OPENROUTER_API_KEY", "OpenRouter", "Models", True),
    ("GROQ_API_KEY", "Groq", "Models", True),
    ("BRAVE_SEARCH_API_KEY", "Brave Search", "Web search", True),
    ("TAVILY_API_KEY", "Tavily", "Web search", True),
    ("SMTP_HOST", "SMTP server", "Email", False),
    ("SMTP_PORT", "SMTP port", "Email", False),
    ("SMTP_USER", "Email login", "Email", False),
    ("SMTP_PASSWORD", "Email password (app password)", "Email", True),
    ("SMTP_FROM", "From address", "Email", False),
    ("IMAP_HOST", "IMAP server (reading mail)", "Email", False),
    ("IMAP_PORT", "IMAP port", "Email", False),
    ("IMAP_USER", "Mailbox login", "Email", False),
    ("IMAP_PASSWORD", "Mailbox password (app password)", "Email", True),
    ("DISCORD_WEBHOOK_URL", "Webhook URL (easiest: posts to one channel)", "Discord", True),
    ("DISCORD_BOT_TOKEN", "Bot token (optional: also reads channels into the Inbox)", "Discord", True),
    ("DISCORD_CHANNEL_IDS", "Channel IDs to watch (comma-separated)", "Discord", False),
    ("REDDIT_CLIENT_ID", "App client ID (under the app name)", "Reddit", False),
    ("REDDIT_CLIENT_SECRET", "App secret", "Reddit", True),
    ("REDDIT_USERNAME", "Your Reddit username", "Reddit", False),
    ("REDDIT_PASSWORD", "Your Reddit password", "Reddit", True),
    ("X_API_KEY", "API key", "X (Twitter)", True),
    ("X_API_SECRET", "API secret", "X (Twitter)", True),
    ("X_ACCESS_TOKEN", "Access token", "X (Twitter)", True),
    ("X_ACCESS_SECRET", "Access token secret", "X (Twitter)", True),
]
NAMES = {k[0] for k in KEYS}


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def current(path: Path) -> dict[str, str]:
    out = {}
    for line in _lines(path):
        m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def status(path: Path) -> list[dict]:
    have = current(path)
    return [{"name": n, "label": label, "group": g, "secret": secret, "set": bool(have.get(n) or os.environ.get(n)),
             **({} if secret else {"value": have.get(n, "")})} for n, label, g, secret in KEYS]


def save(path: Path, values: dict[str, str]) -> list[str]:
    """Set (or clear, with "") the given keys in .env. Returns the names changed."""
    clean = {}
    for name, value in values.items():
        if name not in NAMES:
            raise ValueError(f"Unknown key {name}")
        value = (value or "").strip().strip('"').strip("'")
        if any(c in value for c in "\r\n\0") or len(value) > 500 or " " in value and name != "SMTP_FROM":
            raise ValueError(f"{name}: paste the key alone, on one line, without spaces")
        clean[name] = value
    lines, done = _lines(path), set()
    for i, line in enumerate(lines):
        m = re.match(r"\s*#?\s*([A-Z_][A-Z0-9_]*)\s*=", line)
        if m and m.group(1) in clean and m.group(1) not in done:
            lines[i] = f"{m.group(1)}={clean[m.group(1)]}"
            done.add(m.group(1))
    lines += [f"{n}={v}" for n, v in clean.items() if n not in done]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)
    for n, v in clean.items():  # cleared keys must stop working now, not at the next restart
        if not v:
            os.environ.pop(n, None)
    return sorted(clean)
