"""Live data for the Training Lab page: every training run on this machine (main runs and optimizer tests),
what is running right now, its speed and time left, the data mix, the optimizer race and a log tail."""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from . import stats as ST

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+[▲▼=]?\s*([\d.]+)\s+([\d.]+)(k?)\s*$")
VAL = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)")
NOISE = re.compile(r"rope_parameters|Loading weights|Writing model|Fetching|^\s*$|[╭╰│─]")


def _info(text: str) -> dict:
    get = lambda k: (m.group(1).strip() if (m := re.search(rf"│\s+{k}\s+(.+?)\s+│", text)) else None)
    opt = get("optimizer") or ""
    tp = re.search(r"Trainable parameters: ([\d.]+)% \(([\d.]+M)/([\d.]+M)\)", text)
    return {"model": (get("model") or "").replace("./base/", ""), "lora": get("type"), "dataset": (get("dataset") or "").replace("./", ""),
            "optimizer": opt.split("·")[0].strip() or None, "lr": opt.split("lr")[-1].strip() if "lr" in opt else None,
            "max_seq": get("max seq"), "trainable": f"{tp.group(2)} of {tp.group(3)} ({tp.group(1)}%)" if tp else None}


def parse(log: Path, kind: str) -> dict | None:
    try:
        text = ST.ANSI.sub("", log.read_text(errors="replace"))
    except OSError:
        return None
    train, val, tps, tokens, pauses = [], [], [], 0.0, 0
    for line in text.splitlines():
        if m := VAL.match(line):
            val.append([int(m.group(1)), float(m.group(2))])
        elif m := ROW.match(line):
            step = int(m.group(1))
            train.append([step, float(m.group(2))])
            tps.append([step, float(m.group(3))])
            tokens = float(m.group(4)) * (1000 if m.group(5) else 1)
        elif "[heatwatch]" in line and "pausing" in line:
            pauses += 1
    if not train and not val:
        return None
    info = _info(text)
    total = None
    if m := re.search(r"batch · iters\s+\d+ · ([\d,]+)", text):
        total = int(m.group(1).replace(",", ""))
    stages = re.findall(r"^\[(\d\d:\d\d)\] (.+)$", text, re.M)
    return {"name": log.stem, "kind": kind, **info, "steps_done": train[-1][0] if train else 0, "steps_total": total,
            "train": train, "val": val, "tps": tps, "tokens": int(tokens), "pauses": pauses,
            "stages": [{"at": t, "what": w} for t, w in stages][-6:], "updated": log.stat().st_mtime,
            "started": log.stat().st_birthtime if hasattr(log.stat(), "st_birthtime") else None}


def active() -> dict | None:
    """The training process running now, if any, and whether the heat watchdog has it paused."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,stat,etime,command"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "mlx_lm lora" in line and "heatwatch" not in line:
            parts = line.split(None, 3)
            cmd = parts[3]
            name = (m.group(1).rstrip("/").split("/")[-1] if (m := re.search(r"--adapter-path\s+(\S+)", cmd))
                    else (m.group(1) if (m := re.search(r"-c\s+(\S+)", cmd)) else "training"))
            return {"pid": int(parts[0]), "paused": parts[1].startswith("T"), "elapsed": parts[2], "job": name}
    return None


def tail(log: Path, n: int = 40) -> list[str]:
    try:
        lines = ST.ANSI.sub("", log.read_text(errors="replace")).splitlines()
    except OSError:
        return []
    return [l.rstrip() for l in lines if not NOISE.search(l)][-n:]


def build(lab: Path, data_dir: Path) -> dict:
    runs = [r for p in sorted((lab / "logs").glob("*.log")) if (r := parse(p, "main"))]
    runs += [r for p in sorted((lab / "ab").glob("*.log")) if (r := parse(p, "optimizer test"))]
    runs.sort(key=lambda r: -r["updated"])
    now = time.time()
    act = active()
    current = next((r for r in runs if act and r["name"] == act["job"]), None)
    if act and not current:  # just started: no numbers yet, but the log already names the model and settings
        current = {"name": act["job"], "kind": "starting", "steps_done": 0, "steps_total": None, "train": [], "val": [],
                   "tps": [], "tokens": 0, "pauses": 0, "stages": [], "updated": now, "started": now}
        for cand in (lab / "ab" / f"{act['job']}.log", lab / "logs" / f"{act['job']}.log"):
            if cand.exists():
                text = ST.ANSI.sub("", cand.read_text(errors="replace"))
                current.update({k: v for k, v in _info(text).items() if v})
                if m := re.search(r"batch · iters\s+\d+ · ([\d,]+)", text):
                    current["steps_total"] = int(m.group(1).replace(",", ""))
                current["started"] = cand.stat().st_birthtime if hasattr(cand.stat(), "st_birthtime") else now
                break
    if not current and runs and now - runs[0]["updated"] < 1800:
        current = runs[0]
    if current and current["steps_total"] and current["started"] and current["steps_done"]:
        spent = now - current["started"]
        rate = current["steps_done"] / spent  # steps per second, cooling breaks included
        current["eta_s"] = int((current["steps_total"] - current["steps_done"]) / rate) if rate else None
        current["steps_per_min"] = round(rate * 60, 2)
    mix = None
    for d in ("data_big", "data_8b"):
        if (lab / d / "manifest.json").exists():
            mix = {"name": d, **json.loads((lab / d / "manifest.json").read_text())}
            break
    ab = json.loads((lab / "results" / "optimizer-ab.json").read_text()) if (lab / "results" / "optimizer-ab.json").exists() else None
    log_file = None
    if current:
        for cand in (lab / "ab" / f"{current['name']}.log", lab / "logs" / f"{current['name']}.log"):
            if cand.exists():
                log_file = cand
                break
    return {"now": now, "clock": datetime.now().strftime("%H:%M:%S"), "active": act, "current": current, "runs": runs,
            "mix": mix, "ab": ab, "machine": ST.machine(data_dir), "log": tail(log_file) if log_file else []}
