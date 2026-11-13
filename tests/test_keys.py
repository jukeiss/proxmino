"""Settings > API keys: writes .env, never returns secret values, only on this computer."""
import os
import stat

from fastapi.testclient import TestClient

from proxmino.server import create_app

H = {"X-Proxmino": "1"}


def test_keys_are_saved_but_never_shown(engine, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# DeepSeek\nDEEPSEEK_API_KEY=\n# OLLAMA_API_KEY=old-commented\nOTHER=keep\n")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = create_app(engine, run_scheduler=False, auth_path=tmp_path / "auth.json", env_file=env)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        got = c.post("/api/keys", json={"values": {"DEEPSEEK_API_KEY": " sk-test-123 ", "SMTP_HOST": "smtp.example.com"}}, headers=H)
        assert got.status_code == 200 and "sk-test-123" not in got.text
        lines = env.read_text().splitlines()
        assert "DEEPSEEK_API_KEY=sk-test-123" in lines and "OTHER=keep" in lines and "# DeepSeek" in lines
        assert lines.count("DEEPSEEK_API_KEY=sk-test-123") == 1 and "SMTP_HOST=smtp.example.com" in lines
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
        listed = {k["name"]: k for k in c.get("/api/keys").json()}
        assert listed["DEEPSEEK_API_KEY"]["set"] is True and "value" not in listed["DEEPSEEK_API_KEY"]
        assert listed["SMTP_HOST"]["value"] == "smtp.example.com"  # not a secret
        assert "sk-test-123" not in c.get("/api/keys").text
        assert c.post("/api/keys", json={"values": {"PATH": "/evil"}}, headers=H).status_code == 400
        assert c.post("/api/keys", json={"values": {"OPENAI_API_KEY": "a\nEVIL=1"}}, headers=H).status_code == 400
        c.post("/api/keys", json={"values": {"OLLAMA_API_KEY": "new"}}, headers=H)
        assert "OLLAMA_API_KEY=new" in env.read_text().splitlines() and "old-commented" not in env.read_text()
        c.post("/api/keys", json={"values": {"DEEPSEEK_API_KEY": ""}}, headers=H)
        still_set = "DEEPSEEK_API_KEY" in os.environ  # a bool, so a failure can never print a key
        assert "DEEPSEEK_API_KEY=" in env.read_text().splitlines() and not still_set
    from proxmino.auth import Auth
    Auth(tmp_path / "auth.json").set_password("correct horse battery")
    with TestClient(app, base_url="https://x.trycloudflare.com") as remote:
        h = {"host": "x.trycloudflare.com", "cf-connecting-ip": "1.2.3.4", **H}
        remote.post("/api/login", json={"password": "correct horse battery"}, headers=h)
        assert remote.get("/api/keys", headers=h).status_code == 403  # signed in from outside: still not allowed
