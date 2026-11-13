"""Everything measurable about this install, for the Stats page: models, real speed, training runs, test results,
teacher spending, app usage and the machine. Read-only; no secrets. Missing pieces are simply left out."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRAIN_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+[▲▼=]?\s*([\d.]+)\s+([\d.]+k?)\s*$")
VAL_ROW = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)")


def lab_dir(cfg) -> Path:
    return Path(cfg.get("model_lab") or Path.home() / "praxis-model").expanduser()


async def ollama_models(base: str, client: httpx.AsyncClient) -> dict:
    out = {"installed": [], "loaded": []}
    try:
        tags = (await client.get(f"{base}/api/tags", timeout=5)).json().get("models", [])
        ps = (await client.get(f"{base}/api/ps", timeout=5)).json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return out
    for m in tags:
        d = m.get("details") or {}
        out["installed"].append({"name": m["name"], "gb": round(m.get("size", 0) / 1e9, 2), "family": d.get("family"),
                                 "params": d.get("parameter_size"), "quant": d.get("quantization_level"),
                                 "context": d.get("context_length"), "modified": m.get("modified_at")})
    for m in ps:
        out["loaded"].append({"name": m["name"], "gb": round(m.get("size", 0) / 1e9, 2),
                              "gpu_gb": round(m.get("size_vram", 0) / 1e9, 2), "until": m.get("expires_at"),
                              "context": m.get("context_length")})
    return out


def speed(db, days: int = 30) -> list[dict]:
    """Measured speed per model from saved answers."""
    since = time.time() - days * 86400
    rows = db.q("SELECT json_extract(meta, '$.model') AS model, COUNT(*) AS n, "
                "AVG(json_extract(meta, '$.tokens_per_s')) AS tps, MAX(json_extract(meta, '$.tokens_per_s')) AS best, "
                "AVG(json_extract(meta, '$.first_token_ms')) AS ttft, AVG(json_extract(meta, '$.latency_ms')) AS lat "
                "FROM messages WHERE role = 'assistant' AND created_at >= ? AND json_extract(meta, '$.model') IS NOT NULL "
                "GROUP BY model ORDER BY n DESC", (since,))
    return [{"model": r["model"], "answers": r["n"], "tokens_per_s": round(r["tps"], 1) if r["tps"] else None,
             "best_tokens_per_s": round(r["best"], 1) if r["best"] else None,
             "first_token_ms": int(r["ttft"]) if r["ttft"] else None, "answer_ms": int(r["lat"] or 0)} for r in rows]


def training_runs(lab: Path) -> list[dict]:
    runs = []
    for log in sorted((lab / "logs").glob("*.log")):
        try:
            text = ANSI.sub("", log.read_text(errors="replace"))
        except OSError:
            continue
        train, val, pauses, total = [], [], 0, None
        for line in text.splitlines():
            if m := VAL_ROW.match(line):
                val.append([int(m.group(1)), float(m.group(2))])
            elif m := TRAIN_ROW.match(line):
                train.append([int(m.group(1)), float(m.group(2))])
            elif "[heatwatch]" in line and "pausing" in line:
                pauses += 1
            elif m := re.search(r"iters:\s*(\d+)", line):
                total = int(m.group(1))
        total = total or next((int(x.replace(",", "")) for x in re.findall(r"training \(?([\d,]+) conversations", text)), None)
        if not train and not val:
            continue
        stages = re.findall(r"^\[(\d\d:\d\d)\] (.+)$", text, re.M)
        runs.append({"name": log.stem, "steps_done": max(s for s, _ in train) if train else 0, "steps_total": total,
                     "train_loss": train, "val_loss": val, "cooling_pauses": pauses,
                     "stages": [{"at": t, "what": w} for t, w in stages][-8:], "updated": log.stat().st_mtime})
    return sorted(runs, key=lambda r: -r["updated"])  # the run being trained now comes first


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


NICE = {"s3-base": "SmolLM3-3B (base)", "s3-m1-ckpt225": "praxis-1 trial (step 225)", "s3-m1-final": "praxis-1 trial",
        "s3-m1_1-final": "praxis-1", "s3-phi4-mini": "phi4-mini", "s3-praxis-1-m0b": "praxis-1 (first)",
        "s3-proxmino1": "proxmino:v1", "s3-proxmino11": "proxmino:v1.1", "s3-proxmino8b": "proxmino:8b"}


def tests(lab: Path) -> dict:
    res = lab / "results"
    out = {"benchmarks": [], "reports": [], "eval_seed": []}
    for p in sorted(res.glob("bench-*.json")):
        d = _json(p)
        if d:
            out["benchmarks"].append({"label": d.get("label"), "accuracy": d.get("accuracy"), "ci95": d.get("wilson95"),
                                      "n": d.get("n"), "per_subject": d.get("per_subject"),
                                      "published": d.get("published_for_context")})
    for p in sorted(res.glob("proxmino-*-report.json")):
        d = _json(p)
        if d:
            out["reports"].append({"name": p.stem.replace("-report", ""), **d})
    for p in sorted(res.glob("eval-s3-*.json")):
        d = _json(p)
        if d and d.get("summary"):
            out["eval_seed"].append({"label": NICE.get(d.get("label"), d.get("label")), "summary": {k: v.get("mean") for k, v in d["summary"].items()}})
    return out


def spending(lab: Path) -> list[dict]:
    out = []
    for p in sorted((lab / "distill").glob("*ledger*.json")):
        d = _json(p)
        if d:
            out.append({"run": p.stem, "model": d.get("model"), "tokens_in": d.get("tokens_in"), "tokens_out": d.get("tokens_out"),
                        "dollars": d.get("dollars_estimate", d.get("dollars")), "budget": d.get("budget")})
    raw = lab / "distill" / "v11_raw.jsonl"
    if raw.exists():
        out.append({"run": "v1.1 teacher answers so far", "items": sum(1 for _ in open(raw))})
    return out


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def machine(data_dir: Path) -> dict:
    here = Path(data_dir)
    while not here.exists() and here != here.parent:  # a fresh install has no data folder yet
        here = here.parent
    disk = shutil.disk_usage(str(here))
    info = {"os": f"{platform.system()} {platform.release()}", "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count(), "disk_free_gb": round(disk.free / 1e9, 1), "disk_total_gb": round(disk.total / 1e9, 1)}
    if platform.system() == "Darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or info["cpu"]
        mem = _run(["sysctl", "-n", "hw.memsize"]).strip()
        info["ram_gb"] = round(int(mem) / 2**30) if mem.isdigit() else None
        vm = _run(["vm_stat"])
        page = int(m.group(1)) if (m := re.search(r"page size of (\d+)", vm)) else 16384
        free = sum(int(m.group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative")
                   if (m := re.search(rf"{k}:\s+(\d+)", vm)))
        info["ram_free_gb"] = round(free * page / 2**30, 1) if free else None
        therm = _run(["pmset", "-g", "therm"])
        info["heat"] = "warning" if re.search(r"(level|Limit)\s*=?\s*\d", therm) and "No thermal warning" not in therm else "normal"
        batt = _run(["pmset", "-g", "batt"])
        if m := re.search(r"(\d+)%; (\w+)", batt):
            info["battery"] = {"percent": int(m.group(1)), "state": m.group(2), "plugged": "AC Power" in batt}
    elif Path("/proc/meminfo").exists():
        mi = Path("/proc/meminfo").read_text()
        get = lambda k: int(re.search(rf"{k}:\s+(\d+)", mi).group(1)) // 1024 // 1024 if re.search(rf"{k}:", mi) else None
        info["ram_gb"], info["ram_free_gb"] = get("MemTotal"), get("MemAvailable")
    return info


def app_counts(db, agents: dict) -> dict:
    one = lambda sql: db.q(sql)[0][0]
    return {"chats": one("SELECT count(*) FROM chats"), "messages": one("SELECT count(*) FROM messages"),
            "answers": one("SELECT count(*) FROM messages WHERE role='assistant'"),
            "bots": len(agents), "your_bots": sum(1 for a in agents.values() if not a.builtin),
            "group_chats": one("SELECT count(*) FROM chats WHERE members IS NOT NULL"),
            "artifacts": one("SELECT count(*) FROM artifacts"), "memories": one("SELECT count(*) FROM memory"),
            "sent": one("SELECT count(*) FROM outbox WHERE status='sent'")}
