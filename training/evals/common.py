"""Shared helpers: ask the running Proxmino app (http://127.0.0.1:3000) a question and run Python safely."""
import json, re, subprocess, sys, tempfile, time
import httpx


def ask(q, url="http://127.0.0.1:3000"):
    t0 = time.time(); text = ""; meta = {}; ev = None
    with httpx.stream("POST", f"{url}/api/chat", timeout=1500, headers={"X-Proxmino": "1"}, json={"message": q}) as r:
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                d = json.loads(line[5:])
                if ev == "token": text += d.get("t", "")
                elif ev == "replace": text = d.get("content", text)
                elif ev == "done": meta = d
    return text, meta.get("model"), meta.get("route"), round(time.time() - t0, 1)


def code_of(text, lang="python|py"):
    blocks = re.findall(rf"```(?:{lang})?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else ""


def run_py(src, timeout=30):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        p = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr)[-400:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def final_lines(text, n=3):
    """The end of the answer (where the final answer is), without notes in brackets."""
    lines = [l for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("(") and l.strip() != "---"]
    return "\n".join(lines[-n:]).replace("$", "").replace(",", "")
