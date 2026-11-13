"""Runs due routines in the background, one at a time.

A routine's answer goes into its own chat, continuing that conversation, so the history reads like a
log. A run is skipped (and recorded as skipped) when no model is turned on, so routines never post
demo text or wake a model the user has not opted into. Runs missed while the app was off are not
replayed one by one: the routine runs once, then continues on schedule.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time

from . import routines as R
from .pipeline import Engine, TurnRequest

log = logging.getLogger("proxmino.scheduler")
OPTION_FLAGS = ("search", "deep", "think", "heavy")


def schedule(cron: str, now: float | None = None, anchor: float | None = None) -> tuple[float | None, float]:
    """(next_run, anchor) as epoch seconds for a new or edited routine. One-time runs at the next minute."""
    now_dt = dt.datetime.fromtimestamp(now or time.time()).astimezone()
    if cron.strip() == R.ONCE:
        return (now_dt + dt.timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp(), now_dt.timestamp()
    R.parse_cron(cron)  # raises on a malformed schedule, so it is never saved
    first = R.next_cron(cron, now_dt)
    anchor_dt = dt.datetime.fromtimestamp(anchor).astimezone() if anchor else first
    return R.next_run(cron, anchor_dt, now_dt).timestamp() if anchor else first.timestamp(), anchor_dt.timestamp()


class Scheduler:
    def __init__(self, engine: Engine, every_s: float = 30):
        self.engine, self.every = engine, every_s
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()

    def start(self) -> None:
        if not self.task:
            self.task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
            self.task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("routine tick failed")
            await asyncio.sleep(self.every)

    async def tick(self, now: float | None = None) -> int:
        """Run every due routine once, check the inbox, and write due overviews. Returns how many routines ran."""
        now = now or time.time()
        await self.chief(now)
        done = 0
        for r in self.engine.db.due_routines(now):
            await self.run(r, now=now)
            done += 1
        return done

    async def chief(self, now: float) -> None:
        """Chief-of-staff chores: pull new messages every few minutes; daily and weekly overviews."""
        from . import inbox as INBOX
        from . import reports as REPORTS
        e = self.engine
        every = float(e.cfg.get("inbox.check_minutes", 15)) * 60
        if every and any(INBOX.status().values()) and now - (e.db.kv_get("inbox.last_check") or 0) >= every:
            if await self.model_ready():
                try:
                    await INBOX.check(e)
                except Exception:
                    log.exception("inbox check failed")
                    e.db.kv_set("inbox.last_check", now)  # try again next interval, not every 30 seconds
        try:
            await REPORTS.due(e, now)
        except Exception:
            log.exception("overview failed")

    async def model_ready(self) -> bool:
        router = self.engine.router
        if router.chain("chat"):
            return True
        if not router.allow_local:
            return False
        for prov in router.providers.values():
            if prov.local and await router.installed(prov):
                return True
        return False

    async def run(self, r: dict, now: float | None = None, manual: bool = False) -> dict:
        """Run one routine now. Updates its next run whether it succeeds, fails or is skipped."""
        async with self.lock:
            db = self.engine.db
            run_id = db.add_routine_run(r["id"])
            status, error, message_id = "ok", None, None
            if not await self.model_ready():
                status, error = "skipped", "No model is turned on (Settings > General: Local models or Cloud providers)."
            else:
                chat_id = r.get("chat_id") if r.get("chat_id") and db.chat(r["chat_id"]) else None
                if not chat_id:
                    chat_id = db.create_chat(f"Routine: {r['name']}"[:80])
                    r["chat_id"] = chat_id
                msgs = db.messages(chat_id)
                parent = msgs[-1]["id"] if msgs else None
                opts = r.get("options") or {}
                req = TurnRequest(message=r["prompt"], chat_id=chat_id, parent_id=parent, agent=opts.get("agent") or None,
                                  tools=bool(opts.get("tools")), unattended=True, **{k: bool(opts.get(k)) for k in OPTION_FLAGS})
                try:
                    async for kind, data in self.engine.turn(req):
                        if kind == "error":
                            status, error = "error", data.get("message")
                        elif kind == "meta":
                            message_id = data.get("assistant_id")
                except Exception as e:  # keep the scheduler alive
                    status, error = "error", f"{type(e).__name__}: {e}"
            db.finish_routine_run(run_id, status, error, message_id)
            if not manual:
                if r["cron"].strip() == R.ONCE:
                    r["enabled"], r["next_run"] = False, None
                else:
                    r["next_run"], r["anchor"] = schedule(r["cron"], now, r.get("anchor"))
            db.save_routine(r)
            return {"run_id": run_id, "status": status, "error": error, "chat_id": r.get("chat_id"), "message_id": message_id}
