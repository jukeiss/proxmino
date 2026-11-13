"""Password login for remote access (a tunnel or Tailscale). The app on this computer needs no login.

The password is stored as an scrypt hash in data/auth.json; sessions are random tokens kept as SHA-256
hashes in the same file, valid for 30 days. Set or change the password with `python -m proxmino password`.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path

SESSION_DAYS = 30
COOKIE = "proxmino_session"
# Headers a tunnel or reverse proxy adds; a request carrying any of them came from outside this computer,
# whatever its Host header says.
PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host", "forwarded",
                 "tailscale-user-login", "x-real-ip")


def _scrypt(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32).hex()


class Auth:
    def __init__(self, path: Path):
        self.path = path
        self.failures: dict[str, list[float]] = {}

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.chmod(0o600)
        tmp.replace(self.path)

    @property
    def enabled(self) -> bool:
        return "hash" in self._load()

    def set_password(self, password: str) -> None:
        if len(password) < 10:
            raise ValueError("Use at least 10 characters.")
        salt = secrets.token_bytes(16)
        self._save({"salt": salt.hex(), "hash": _scrypt(password, salt), "sessions": {}})  # signs everyone out

    def check(self, password: str) -> bool:
        d = self._load()
        if "hash" not in d:
            return False
        return hmac.compare_digest(_scrypt(password, bytes.fromhex(d["salt"])), d["hash"])

    def limited(self, who: str, limit: int = 8, window: float = 600) -> bool:
        now = time.time()
        recent = [t for t in self.failures.get(who, []) if now - t < window]
        self.failures[who] = recent
        return len(recent) >= limit

    def failed(self, who: str) -> None:
        self.failures.setdefault(who, []).append(time.time())

    def new_session(self) -> str:
        token = secrets.token_urlsafe(32)
        d = self._load()
        now = time.time()
        sessions = {k: v for k, v in d.get("sessions", {}).items() if v > now}
        sessions[hashlib.sha256(token.encode()).hexdigest()] = now + SESSION_DAYS * 86400
        d["sessions"] = sessions
        self._save(d)
        return token

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        exp = self._load().get("sessions", {}).get(hashlib.sha256(token.encode()).hexdigest())
        return bool(exp and exp > time.time())

    def end_session(self, token: str | None) -> None:
        if not token:
            return
        d = self._load()
        d.get("sessions", {}).pop(hashlib.sha256(token.encode()).hexdigest(), None)
        self._save(d)


def is_remote(headers, host: str, local_hosts: set[str]) -> bool:
    return host not in local_hosts or any(h in headers for h in PROXY_HEADERS)


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{product}: sign in</title><style>
:root{{color-scheme:dark;--bg:#0b0b0c;--s:#141416;--b:#2a2b30;--t:#ececee;--m:#8b8c93;--e:#ff6b6b}}
@media(prefers-color-scheme:light){{:root{{color-scheme:light;--bg:#fbfbfa;--s:#f3f3f1;--b:#dedfdb;--t:#18181b;--m:#5f6068}}}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--t);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:16px}}
form{{width:100%;max-width:340px;display:flex;flex-direction:column;gap:12px;text-align:center}}
h1{{font-size:24px;margin:8px 0 0}}p{{color:var(--m);margin:0 0 6px}}
input{{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:12px 14px;color:var(--t);font-size:16px;outline:0}}
button{{background:var(--t);color:var(--bg);border:0;border-radius:12px;padding:12px;font-size:15px;font-weight:600;cursor:pointer}}
#err{{color:var(--e);min-height:20px;font-size:14px}}</style></head><body>
<form id="f"><svg viewBox="-60 -60 120 120" width="72" height="72" style="margin:0 auto" aria-hidden="true"><path d="M-40 10C-52 -30 -10 -50 20 -40S55 0 40 30-30 50-40 10Z" fill="#fff"/><rect x="-14" y="-12" width="7" height="24" rx="3.5" fill="#141414"/><rect x="7" y="-12" width="7" height="24" rx="3.5" fill="#141414"/></svg>
<h1>{product}</h1><p>Enter your password to continue.</p>
<input id="pw" type="password" autocomplete="current-password" aria-label="Password" placeholder="Password" autofocus required>
<button>Sign in</button><div id="err" role="alert"></div></form>
<script>document.getElementById("f").onsubmit=async e=>{{e.preventDefault();
const r=await fetch("/api/login",{{method:"POST",headers:{{"Content-Type":"application/json","X-Proxmino":"1"}},body:JSON.stringify({{password:document.getElementById("pw").value}})}});
if(r.ok)location.reload();else document.getElementById("err").textContent=(await r.json().catch(()=>({{}}))).detail||"Sign-in failed";}};</script></body></html>"""
