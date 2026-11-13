"""Remote access: this computer needs no login; tunnel requests need the password."""
from fastapi.testclient import TestClient

from proxmino.server import create_app

H = {"X-Proxmino": "1"}
TUNNEL = {"host": "blue-cat.trycloudflare.com", "cf-connecting-ip": "203.0.113.9"}


def test_remote_needs_password_local_does_not(engine, tmp_path):
    app = create_app(engine, run_scheduler=False, auth_path=tmp_path / "auth.json")
    with TestClient(app, base_url="http://127.0.0.1") as local, \
            TestClient(app, base_url="https://blue-cat.trycloudflare.com") as remote:
        assert local.get("/api/chats").status_code == 200
        off = remote.get("/api/chats", headers=TUNNEL)
        assert off.status_code == 403 and "proxmino password" in off.json()["detail"]  # no password set: remote is off
        # a proxied request claiming a local Host is still remote
        assert local.get("/api/chats", headers={"cf-connecting-ip": "203.0.113.9"}).status_code == 403
        assert remote.get("/api/chats", headers={"host": "evil.example.com"}).status_code == 403
        from proxmino.auth import Auth
        Auth(tmp_path / "auth.json").set_password("correct horse battery")
        assert remote.get("/api/chats", headers=TUNNEL).status_code == 401
        page = remote.get("/", headers=TUNNEL)
        assert page.status_code == 200 and "Sign in" in page.text and "{{PRODUCT}}" not in page.text
        assert remote.post("/api/login", json={"password": "nope"}, headers={**TUNNEL, **H}).status_code == 401
        ok = remote.post("/api/login", json={"password": "correct horse battery"}, headers={**TUNNEL, **H})
        assert ok.status_code == 200
        cookie = ok.headers["set-cookie"].lower()
        assert "httponly" in cookie and "secure" in cookie and "samesite=strict" in cookie
        got = remote.get("/api/chats", headers=TUNNEL)
        assert got.status_code == 200 and got.headers["x-frame-options"] == "DENY"
        assert remote.post("/api/logout", headers={**TUNNEL, **H}).status_code == 200
        remote.cookies.clear()
        assert remote.get("/api/chats", headers=TUNNEL).status_code == 401
        assert local.get("/api/chats").status_code == 200


def test_login_is_rate_limited(engine, tmp_path):
    from proxmino.auth import Auth
    Auth(tmp_path / "auth.json").set_password("correct horse battery")
    app = create_app(engine, run_scheduler=False, auth_path=tmp_path / "auth.json")
    with TestClient(app, base_url="https://blue-cat.trycloudflare.com") as remote:
        codes = [remote.post("/api/login", json={"password": "x"}, headers={**TUNNEL, **H}).status_code for _ in range(9)]
        assert codes[:8] == [401] * 8 and codes[8] == 429
        assert remote.post("/api/login", json={"password": "correct horse battery"}, headers={**TUNNEL, **H}).status_code == 429


def test_short_password_refused(tmp_path):
    import pytest
    from proxmino.auth import Auth
    with pytest.raises(ValueError):
        Auth(tmp_path / "a.json").set_password("short")
