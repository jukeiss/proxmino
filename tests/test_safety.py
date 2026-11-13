"""Safety: the page fetcher only reaches the public web, never this computer or the home network."""
import httpx

from proxmino.search import _is_public_ip, safe_get

from conftest import run


def test_only_global_addresses_count_as_public():
    for bad in ("127.0.0.1", "0.0.0.0", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254", "100.100.1.1",
                "::1", "::ffff:127.0.0.1", "::ffff:192.168.1.1", "fe80::1", "fd00::1", "224.0.0.1", "not-an-ip"):
        assert not _is_public_ip(bad), bad
    for good in ("1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"):
        assert _is_public_ip(good), good


def test_redirects_into_the_local_network_are_refused():
    hops = []

    def handler(req):
        hops.append(str(req.url))
        if req.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:3000/api/keys"})
        return httpx.Response(200, text="secret", headers={"content-type": "text/html"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolve = lambda host: host == "public.example"  # only this name is public
    assert run(safe_get(client, "https://public.example/a", resolve=resolve)) is None
    assert hops == ["https://public.example/a"]  # the local address was never requested
    assert run(safe_get(client, "file:///etc/passwd", resolve=resolve)) is None
    assert run(safe_get(client, "https://user:pw@public.example/", resolve=resolve)) is None


def test_public_redirect_chain_is_followed_and_capped():
    def handler(req):
        n = int(req.url.path.strip("/") or 0)
        if n < 10:
            return httpx.Response(301, headers={"location": f"/{n + 1}"})
        return httpx.Response(200, text="ok")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert run(safe_get(client, "https://public.example/8", resolve=lambda h: True)).text == "ok"
    assert run(safe_get(client, "https://public.example/0", resolve=lambda h: True)) is None  # too many hops


def test_every_response_carries_security_headers(engine, tmp_path):
    from fastapi.testclient import TestClient
    from proxmino.server import create_app
    with TestClient(create_app(engine, run_scheduler=False, auth_path=tmp_path / "a.json"), base_url="http://127.0.0.1") as c:
        for r in (c.get("/"), c.get("/api/chats"), c.post("/api/chats", json={}),  # the last lacks X-Praxis: 403
                  c.get("/api/chats", headers={"host": "evil.example"})):
            csp = r.headers["content-security-policy"]
            assert "frame-ancestors 'none'" in csp and "img-src 'self' data: blob:" in csp and "connect-src 'self'" in csp
            assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
        big = c.post("/api/chats", headers={"X-Proxmino": "1", "content-length": str(30 * 1024 * 1024)}, content=b"x")
        assert big.status_code == 413


def test_data_folder_is_owner_only(tmp_path):
    import stat
    from proxmino.config import lock_down
    d = tmp_path / "data"
    d.mkdir()
    (d / "proxmino.db").write_text("x")
    (d / "proxmino.db").chmod(0o644)
    lock_down(d)
    assert stat.S_IMODE(d.stat().st_mode) == 0o700 and stat.S_IMODE((d / "proxmino.db").stat().st_mode) == 0o600
