"""Daily and weekly overviews: what Proxmino handled and what needs you. Written to your Obsidian vault
(Proxmino/Daily, Proxmino/Weekly) and, if you switch it on, emailed to yourself."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from . import integrations as IN


DEFAULT_VAULT = "~/Desktop/obsdian vault"


def vault_dir(cfg) -> Path | None:
    v = cfg.get("obsidian.vault", DEFAULT_VAULT)
    if not v:
        return None
    p = Path(v).expanduser()
    return p / cfg.get("obsidian.folder", "Proxmino") if p.is_dir() else None


def build(db, since: float, until: float, title: str) -> str:
    q = lambda sql, *a: db.q(sql, a)
    answers = q("SELECT json_extract(meta,'$.model') AS m, count(*) AS n FROM messages "
                "WHERE role='assistant' AND created_at BETWEEN ? AND ? GROUP BY m ORDER BY n DESC", since, until)
    chats = q("SELECT count(DISTINCT chat_id) AS n FROM messages WHERE created_at BETWEEN ? AND ?", since, until)[0]["n"]
    sent = q("SELECT kind, status, target, substr(text,1,90) AS t FROM outbox WHERE created_at BETWEEN ? AND ? ORDER BY created_at", since, until)
    runs = q("SELECT t.name AS name, r.status AS status, r.error AS error FROM routine_runs r JOIN routines t ON t.id=r.routine_id "
             "WHERE r.started_at BETWEEN ? AND ? ORDER BY r.started_at", since, until)
    inbox = db.inbox_counts_since(since)
    pending = [i for i in db.inbox() if i["status"] in ("new", "drafted")][:15]
    day = lambda t: datetime.fromtimestamp(t).strftime("%a %d %b %H:%M")
    lines = [f"# {title}", "", f"{day(since)} to {day(until)}", "",
             "## What I handled",
             f"- Answered **{(n := sum(a['n'] for a in answers))}** message{'' if n == 1 else 's'} in **{chats}** chat{'' if chats == 1 else 's'}"
             + (" (" + ", ".join(f"{a['m'] or 'none'}: {a['n']}" for a in answers) + ")" if answers else ""),
             f"- Inbox: {sum(inbox.values())} new messages; {inbox.get('drafted', 0)} drafted, {inbox.get('sent', 0)} answered, "
             f"{inbox.get('ignored', 0)} set aside as spam or automated"]
    ok_sent = [s for s in sent if s["status"] == "sent"]
    if ok_sent:
        lines.append(f"- Sent {len(ok_sent)} (all approved by you):")
        lines += [f"  - {s['kind'].replace('_', ' ')} to {s['target'] or 'your feed'}: {s['t']}" for s in ok_sent[:20]]
    if runs:
        lines.append(f"- Routines: {sum(r['status'] == 'ok' for r in runs)} ran, {sum(r['status'] != 'ok' for r in runs)} skipped or failed")
    lines += ["", "## Needs you"]
    review = [f"- **{i['priority']}** · {i['source']} · {i['sender']}: {i['summary']}" + (" (draft ready)" if i["draft"] else "")
              for i in pending]
    review += [f"- Failed to send {s['kind'].replace('_', ' ')} to {s['target']}" for s in sent if s["status"] != "sent"]
    review += [f"- Routine '{r['name']}' {r['status']}: {r['error'] or ''}".rstrip(": ") for r in runs if r["status"] not in ("ok",)]
    lines += review or ["- Nothing. All clear."]
    return "\n".join(lines) + "\n"


async def make(engine, kind: str = "daily", now: float | None = None, email: bool | None = None) -> dict:
    now = now or time.time()
    span = 86400 if kind == "daily" else 7 * 86400
    title = ("Daily overview " + datetime.fromtimestamp(now).strftime("%Y-%m-%d") if kind == "daily"
             else "Weekly overview " + datetime.fromtimestamp(now).strftime("%G-W%V"))
    text = build(engine.db, now - span, now, title)
    saved = None
    folder = vault_dir(engine.cfg)
    if folder:
        sub = folder / ("Daily" if kind == "daily" else "Weekly")
        sub.mkdir(parents=True, exist_ok=True)
        name = (datetime.fromtimestamp(now).strftime("%Y-%m-%d") if kind == "daily"
                else datetime.fromtimestamp(now).strftime("%G-W%V")) + ".md"
        front = f"---\ntype: proxmino-{kind}\ncreated: {datetime.fromtimestamp(now).isoformat(timespec='minutes')}\n---\n"
        (sub / name).write_text(front + text)
        saved = str(sub / name)
    mailed = None
    want_mail = engine.cfg.get("reports.email", False) if email is None else email
    if want_mail and IN.email_ready():
        me = IN.os.environ.get("SMTP_FROM") or IN.os.environ["SMTP_USER"]
        mailed = await engine.outward("send_email", {"to": me, "subject": title, "body": text})
    report = {"kind": kind, "title": title, "text": text, "saved": saved, "mailed": mailed, "at": now}
    engine.db.kv_set(f"report.{kind}", report)
    return report


async def due(engine, now: float | None = None) -> list[dict]:
    """Called by the scheduler: run the daily report once a day after the set hour, the weekly one on Mondays."""
    now = now or time.time()
    dt = datetime.fromtimestamp(now)
    hour = int(engine.cfg.get("reports.hour", 8))
    out = []
    if engine.cfg.get("reports.daily", True) and dt.hour >= hour:
        last = engine.db.kv_get("report.daily.date")
        if last != dt.strftime("%Y-%m-%d"):
            engine.db.kv_set("report.daily.date", dt.strftime("%Y-%m-%d"))
            out.append(await make(engine, "daily", now))
    if engine.cfg.get("reports.weekly", True) and dt.weekday() == 0 and dt.hour >= hour:
        wk = dt.strftime("%G-W%V")
        if engine.db.kv_get("report.weekly.week") != wk:
            engine.db.kv_set("report.weekly.week", wk)
            out.append(await make(engine, "weekly", now))
    return out
