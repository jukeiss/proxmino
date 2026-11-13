"""Settings > API keys > Test: signs in to each connected service to show whether its keys work.
Read-only: nothing is sent, posted or marked read. Error text never includes a key."""
from __future__ import annotations

import asyncio
import imaplib
import os
import smtplib

import httpx

from . import inbox as INBOX
from . import integrations as IN
from . import reddit as REDDIT


def _smtp() -> str:
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=20) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
    return f"Signed in as {os.environ['SMTP_USER']}"


def _imap() -> str:
    m = imaplib.IMAP4_SSL(os.environ["IMAP_HOST"], int(os.environ.get("IMAP_PORT", "993")), timeout=20)
    try:
        m.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"])
        m.select("INBOX", readonly=True)
    finally:
        try:
            m.logout()
        except Exception:
            pass
    return f"Mailbox {os.environ['IMAP_USER']} opened read-only"


async def _discord(client: httpx.AsyncClient) -> str:
    h = INBOX._discord_headers()
    r = await client.get(f"{INBOX.DISCORD}/users/@me", headers=h, timeout=15)
    if r.status_code >= 300:
        raise RuntimeError(f"Discord said {r.status_code}: check the bot token")
    bad = []
    for cid in [c.strip() for c in os.environ["DISCORD_CHANNEL_IDS"].split(",") if c.strip()]:
        c = await client.get(f"{INBOX.DISCORD}/channels/{cid}", headers=h, timeout=15)
        if c.status_code >= 300:
            bad.append(cid)
    name = r.json().get("username", "bot")
    if bad:
        raise RuntimeError(f"Signed in as {name}, but it cannot see channel {', '.join(bad)}: invite the bot to that server "
                           "and give it View Channel, Read Message History and Send Messages")
    return f"Signed in as {name}"


async def _discord_webhook(client: httpx.AsyncClient) -> str:
    if not INBOX.webhook_url():
        raise RuntimeError("That is not a Discord webhook URL: it should start with https://discord.com/api/webhooks/")
    r = await client.get(INBOX.webhook_url(), timeout=15)  # reads the webhook's details; posts nothing
    if r.status_code >= 300:
        raise RuntimeError(f"Discord said {r.status_code}: the webhook URL is wrong or was deleted")
    return f"Webhook \"{r.json().get('name', 'webhook')}\" works"


async def _x(client: httpx.AsyncClient) -> str:
    url = "https://api.x.com/2/users/me"
    auth = IN.oauth1_header("GET", url, os.environ["X_API_KEY"], os.environ["X_API_SECRET"],
                            os.environ["X_ACCESS_TOKEN"], os.environ["X_ACCESS_SECRET"])
    r = await client.get(url, headers={"Authorization": auth}, timeout=15)
    if r.status_code >= 300:
        hint = " (the app needs Read and Write permission, then regenerate the access token)" if r.status_code in (401, 403) else ""
        raise RuntimeError(f"X said {r.status_code}{hint}")
    return f"Signed in as @{(r.json().get('data') or {}).get('username', '?')}"


async def _reddit(client: httpx.AsyncClient) -> str:
    return f"Signed in as u/{await REDDIT.me(client)}"


async def test_all(client: httpx.AsyncClient | None) -> dict[str, dict]:
    own = client is None
    client = client or httpx.AsyncClient(timeout=20)
    checks = {
        "Email (sending)": (IN.email_ready(), lambda: asyncio.to_thread(_smtp)),
        "Email (reading)": (INBOX.email_ready(), lambda: asyncio.to_thread(_imap)),
        "Discord (webhook)": (bool(os.environ.get("DISCORD_WEBHOOK_URL")), lambda: _discord_webhook(client)),
        "Discord (bot)": (INBOX.discord_ready(), lambda: _discord(client)),
        "Reddit": (REDDIT.ready(), lambda: _reddit(client)),
        "X (Twitter)": (IN.x_ready(), lambda: _x(client)),
    }

    async def one(ok, run):
        if not ok:
            return {"ok": None, "detail": "Not set up"}
        try:
            return {"ok": True, "detail": await asyncio.wait_for(run(), 30)}
        except Exception as e:  # report, never raise: one broken service must not hide the others
            return {"ok": False, "detail": str(e)[:300] or type(e).__name__}

    try:
        results = await asyncio.gather(*(one(ok, run) for ok, run in checks.values()))
    finally:
        if own:
            await client.aclose()
    return dict(zip(checks, results))
