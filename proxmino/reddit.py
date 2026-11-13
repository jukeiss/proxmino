"""Reddit through its official API (a "script" app you make at https://www.reddit.com/prefs/apps for your own
account). Reads your unread inbox (replies, mentions, messages) without marking anything read, and posts or
replies only when you approve it. Follows Reddit's API terms: a real User-Agent, OAuth, and its rate limits.

Keys live only in .env:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
"""
from __future__ import annotations

import os
import re
import time

import httpx

AUTH = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"
KEYS = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USERNAME", "REDDIT_PASSWORD")
LIMIT = 10_000  # Reddit's limit for a comment or self post
SUB = re.compile(r"^/?r/([A-Za-z0-9_]{2,21})/?$")
PERMALINK = re.compile(r"reddit\.com/r/[^/]+/comments/([a-z0-9]+)(?:/[^/?#]*/([a-z0-9]+))?", re.I)
FULLNAME = re.compile(r"^t[134]_[a-z0-9]+$", re.I)
_token: dict = {}


def ready() -> bool:
    return all(os.environ.get(k) for k in KEYS)


def user_agent() -> str:
    return f"desktop:proxmino:1.0 (by /u/{os.environ.get('REDDIT_USERNAME', 'unknown')})"


async def _auth(client: httpx.AsyncClient) -> dict:
    """A bearer token for the user's own account, reused until a minute before it expires."""
    who = os.environ["REDDIT_USERNAME"]
    if _token.get("who") == who and _token.get("until", 0) > time.time():
        return {"Authorization": f"bearer {_token['value']}", "User-Agent": user_agent()}
    r = await client.post(AUTH, auth=(os.environ["REDDIT_CLIENT_ID"], os.environ["REDDIT_CLIENT_SECRET"]),
                          data={"grant_type": "password", "username": who, "password": os.environ["REDDIT_PASSWORD"]},
                          headers={"User-Agent": user_agent()}, timeout=20)
    d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code >= 300 or "access_token" not in d:
        raise RuntimeError(f"Reddit sign-in failed ({r.status_code}): {d.get('error') or r.text[:120]}. "
                           "Check the app is a 'script' app and the username and password are right (no 2FA on script apps).")
    _token.update(who=who, value=d["access_token"], until=time.time() + int(d.get("expires_in", 3600)) - 60)
    return {"Authorization": f"bearer {d['access_token']}", "User-Agent": user_agent()}


async def me(client: httpx.AsyncClient) -> str:
    r = await client.get(f"{API}/api/v1/me", headers=await _auth(client), timeout=15)
    r.raise_for_status()
    return r.json().get("name", "")


def target(ref: str) -> tuple[str, str]:
    """("sub", name) for r/name, or ("thing", fullname) for a post or comment URL or fullname."""
    ref = (ref or "").strip()
    if m := SUB.match(ref):
        return "sub", m.group(1)
    if FULLNAME.match(ref):
        return "thing", ref.lower()
    if m := PERMALINK.search(ref):
        return "thing", f"t1_{m.group(2)}" if m.group(2) else f"t3_{m.group(1)}"
    raise ValueError("Give r/subreddit for a new post, or a Reddit post or comment link to reply to")


async def post(client: httpx.AsyncClient, text: str, where: str, title: str = "") -> str:
    if not ready():
        raise RuntimeError("Reddit is not set up: add the four REDDIT_ keys in Settings > API keys")
    text = text.strip()
    if not text or len(text) > LIMIT:
        raise ValueError(f"A Reddit post or reply must be 1 to {LIMIT:,} characters")
    kind, name = target(where)
    h = await _auth(client)
    if kind == "sub":
        title = " ".join(title.split())[:300]
        if not title:
            raise ValueError("A new Reddit post needs a title")
        data = {"sr": name, "kind": "self", "title": title, "text": text, "api_type": "json", "resubmit": "true"}
        r = await client.post(f"{API}/api/submit", data=data, headers=h, timeout=20)
    else:
        r = await client.post(f"{API}/api/comment", data={"thing_id": name, "text": text, "api_type": "json"}, headers=h, timeout=20)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    errors = (body.get("json") or {}).get("errors") or []
    if r.status_code >= 300 or errors:
        raise RuntimeError(f"Reddit refused it ({r.status_code}): {errors or r.text[:200]}")
    data = (body.get("json") or {}).get("data") or {}
    url = data.get("url") or next((t["data"].get("permalink") for t in data.get("things", []) if t.get("data")), "")
    url = f"https://www.reddit.com{url}" if url.startswith("/") else url
    return f"Posted on Reddit{': ' + url if url else ''}"


async def fetch(client: httpx.AsyncClient, limit: int = 25) -> list[dict]:
    """Unread replies, mentions and messages, as inbox items. Nothing is marked read."""
    if not ready():
        return []
    r = await client.get(f"{API}/message/unread", params={"limit": min(limit, 100), "mark": "false"},
                         headers=await _auth(client), timeout=20)
    r.raise_for_status()
    out = []
    for child in (r.json().get("data") or {}).get("children", []):
        m = child.get("data") or {}
        if not m.get("name") or not m.get("body"):
            continue
        ctx = m.get("context") or ""
        out.append({"source": "reddit", "external_id": m["name"], "thread": m.get("link_id") or m["name"],
                    "sender": f"u/{m.get('author') or 'unknown'}", "subject": m.get("link_title") or m.get("subject") or "",
                    "text": m["body"][:4000],
                    "meta": {"thing_id": m["name"], "url": f"https://www.reddit.com{ctx}" if ctx else ""}})
    return out
