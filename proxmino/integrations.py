"""Things that leave this computer: email (SMTP) and posts on X (official API). Every send needs the
user's approval, and each one is logged in the outbox table.

Keys live only in .env (never in the app's data or the vault):
  SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM
  X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET   (an X developer app with read and write access)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import re
import secrets
import smtplib
import time
import urllib.parse
from email.message import EmailMessage

import httpx

X_POST_URL = "https://api.x.com/2/tweets"
X_LIMIT = 280
EMAIL = re.compile(r"[^@\s<>,;]+@[^@\s<>,;]+\.[A-Za-z]{2,}")
TWEET_ID = re.compile(r"(?:x|twitter)\.com/[^/\s]+/status(?:es)?/(\d+)")


def email_ready() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def x_ready() -> bool:
    return all(os.environ.get(k) for k in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"))


def status() -> dict:
    return {"email": email_ready(), "x": x_ready()}


MESSAGE_ID = re.compile(r"<[^<>\s\r\n]{1,300}>")


def _send_email_sync(to: str, subject: str, body: str, in_reply_to: str | None = None, auto: bool = False) -> None:
    msg = EmailMessage()
    msg["From"] = os.environ.get("SMTP_FROM") or os.environ["SMTP_USER"]
    msg["To"], msg["Subject"] = to, subject
    if in_reply_to:  # keeps the reply in the same thread
        msg["In-Reply-To"] = msg["References"] = in_reply_to
    if auto:  # RFC 3834: other auto-responders must not answer this
        msg["Auto-Submitted"] = "auto-replied"
    msg.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=30) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        s.send_message(msg)


async def send_email(to: str, subject: str, body: str, in_reply_to: str | None = None, auto: bool = False) -> str:
    to, subject = to.strip(), " ".join(subject.split())[:200]
    if not email_ready():
        raise RuntimeError("Email is not set up: add SMTP_HOST, SMTP_USER and SMTP_PASSWORD to .env")
    addrs = [a.strip() for a in re.split(r"[,;]", to) if a.strip()]
    if not addrs or len(addrs) > 10 or not all(EMAIL.fullmatch(a) for a in addrs):
        raise ValueError("Give one to ten valid email addresses")
    if not body.strip():
        raise ValueError("The email is empty")
    if in_reply_to and not MESSAGE_ID.fullmatch(in_reply_to.strip()):
        in_reply_to = None  # never put an unchecked value into a mail header
    await asyncio.to_thread(_send_email_sync, ", ".join(addrs), subject or "(no subject)", body,
                            in_reply_to.strip() if in_reply_to else None, auto)
    return f"Sent to {', '.join(addrs)}"


def _pct(s: str) -> str:
    return urllib.parse.quote(s, safe="~-._")


def oauth1_header(method: str, url: str, key: str, secret: str, token: str, token_secret: str,
                  nonce: str | None = None, ts: str | None = None) -> str:
    """OAuth 1.0a (HMAC-SHA1) for a JSON body: only the oauth_* parameters are signed."""
    params = {"oauth_consumer_key": key, "oauth_nonce": nonce or secrets.token_hex(16),
              "oauth_signature_method": "HMAC-SHA1", "oauth_timestamp": ts or str(int(time.time())),
              "oauth_token": token, "oauth_version": "1.0"}
    base_params = "&".join(f"{_pct(k)}={_pct(v)}" for k, v in sorted(params.items()))
    base = "&".join([method.upper(), _pct(url), _pct(base_params)])
    sig = base64.b64encode(hmac.new(f"{_pct(secret)}&{_pct(token_secret)}".encode(), base.encode(), hashlib.sha1).digest()).decode()
    params["oauth_signature"] = sig
    return "OAuth " + ", ".join(f'{_pct(k)}="{_pct(v)}"' for k, v in sorted(params.items()))


def tweet_id(ref: str | None) -> str | None:
    if not ref:
        return None
    ref = ref.strip()
    if ref.isdigit():
        return ref
    m = TWEET_ID.search(ref)
    return m.group(1) if m else None


async def post_x(text: str, reply_to: str | None = None, client: httpx.AsyncClient | None = None) -> str:
    text = text.strip()
    if not x_ready():
        raise RuntimeError("X is not set up: add X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN and X_ACCESS_SECRET to .env")
    if not text or len(text) > X_LIMIT:
        raise ValueError(f"A post must be 1 to {X_LIMIT} characters (this one is {len(text)})")
    body: dict = {"text": text}
    rid = tweet_id(reply_to)
    if reply_to and not rid:
        raise ValueError("Reply target must be a post URL like https://x.com/name/status/123 or its id")
    if rid:
        body["reply"] = {"in_reply_to_tweet_id": rid}
    auth = oauth1_header("POST", X_POST_URL, os.environ["X_API_KEY"], os.environ["X_API_SECRET"],
                         os.environ["X_ACCESS_TOKEN"], os.environ["X_ACCESS_SECRET"])
    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        r = await client.post(X_POST_URL, json=body, headers={"Authorization": auth})
    finally:
        if own:
            await client.aclose()
    if r.status_code >= 300:
        detail = r.text[:300]
        raise RuntimeError(f"X refused the post ({r.status_code}): {detail}")
    pid = (r.json().get("data") or {}).get("id", "")
    return f"Posted on X: https://x.com/i/status/{pid}" if pid else "Posted on X"


THREAD_MAX = 25


def split_thread(text: str) -> list[str]:
    """Long text -> posts of at most 280 characters, split at paragraph, then sentence, then word boundaries."""
    parts: list[str] = []
    for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        chunk = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            for word in sent.split(" "):
                cand = f"{chunk} {word}".strip() if chunk else word
                if len(cand) <= X_LIMIT - 6:  # room for " 1/12"
                    chunk = cand
                else:
                    parts.append(chunk)
                    chunk = word[: X_LIMIT - 6]
        if chunk:
            parts.append(chunk)
    numbered = all(re.match(r"^\(?\d{1,2}\s*/\s*\d{0,2}\)?", p) for p in parts)  # the writer numbered them already
    if len(parts) > 1 and not numbered:
        parts = [f"{p} {i}/{len(parts)}" for i, p in enumerate(parts, 1)]
    return parts


async def post_x_thread(posts: list[str] | str, client: httpx.AsyncClient | None = None) -> str:
    """Post a thread: each post replies to the one before. Stops at the first failure and says how far it got."""
    posts = split_thread(posts) if isinstance(posts, str) else [str(p).strip() for p in posts if str(p).strip()]
    if not posts or len(posts) > THREAD_MAX:
        raise ValueError(f"A thread must have 1 to {THREAD_MAX} posts")
    links, prev = [], None
    for i, text in enumerate(posts, 1):
        try:
            res = await post_x(text, prev, client)
        except Exception as e:
            done = f" Posted {i - 1} of {len(posts)}: {' '.join(links)}" if links else ""
            raise RuntimeError(f"Thread stopped at post {i}: {e}.{done}") from None
        m = re.search(r"status/(\d+)", res)
        prev = m.group(1) if m else None
        links.append(res.split(": ", 1)[-1])
        if not prev and i < len(posts):
            raise RuntimeError(f"X did not return the id of post {i}, so the thread cannot continue. Posted: {' '.join(links)}")
    return f"Posted a {len(posts)}-post thread on X: {links[0]}"


REPLY_PROMPT = """Someone else wrote the post below on {platform}. Write the user's reply to that person.
The user's goal on this platform: {goal}
The user's voice: {voice}

Rules:
- Add something real: a specific insight, a useful question, or a short concrete example. Never "Great post", flattery or filler.
- Start with the point. No greeting, no "Hey", no sign-off.
- Sound like a person: plain words, no hashtags, no emojis, no links{product_rule}.
- Never invent experiences, numbers or facts about the user.
- {length}
- Reply with the text of the reply only.

<post>
{post}
</post>"""

LENGTHS = {"x": "At most 240 characters.", "linkedin": "Two to four short sentences.", "reddit": "One short paragraph, helpful first."}


def reply_prompt(post: str, platform: str, goal: str, voice: str, mention_product: str) -> str:
    product_rule = (f"; mention {mention_product} only if it truly answers the post, at most once, without a sales pitch"
                    if mention_product else "")
    return REPLY_PROMPT.format(platform={"x": "X (Twitter)", "linkedin": "LinkedIn", "reddit": "Reddit"}.get(platform, platform),
                               goal=goal or "be useful and get noticed by the right people",
                               voice=voice or "friendly, direct, a little informal", product_rule=product_rule,
                               length=LENGTHS.get(platform, LENGTHS["x"]), post=post[:4000])


GREETING = re.compile(r"^\s*(hey|hi|hello|yo)\b[^.!?\n]{0,40}[.!?,]\s*", re.I)


def clean_reply(text: str) -> str:
    """Enforce the rules small models forget: no wrapping quotes, greetings, hashtags or emojis."""
    from .safety import strip_emoji
    text = strip_emoji(text.strip().strip('"').strip())
    text = re.sub(r"(?:^|\s)#\w+", "", text)
    text = GREETING.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text[:1].upper() + text[1:] if text else text


def split_posts(text: str) -> list[dict]:
    """Posts pasted one per block (blank line between). A post URL in a block is kept as the reply target."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        url = next((u for u in re.findall(r"https?://\S+", block) if TWEET_ID.search(u)), None)
        body = block.replace(url, "").strip() if url else block
        out.append({"post": body or block, "url": url})
    return out[:50]
