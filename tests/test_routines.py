"""Routines: schedule presets and labels (ported from Rakazo), next-run times, and the scheduler."""
import datetime as dt
import time

import pytest
from fastapi.testclient import TestClient

from conftest import run
from proxmino import routines as R
from proxmino.llm import Router
from proxmino.scheduler import Scheduler, schedule
from proxmino.server import create_app


def local(*a):
    return dt.datetime(*a).astimezone()


@pytest.mark.parametrize("preset,cron,label", [
    (R.Preset("Every day", time="9:00 AM"), "0 9 * * *", "Every day at 9:00 AM"),
    (R.Preset("Weekdays", time="8:30 AM"), "30 8 * * 1-5", "Weekdays at 8:30 AM"),
    (R.Preset("Every week", time="6:00 PM"), "0 18 * * 1", "Every Monday at 6:00 PM"),
    (R.Preset("Every month", time="12:00 AM"), "0 0 1 * *", "Monthly on the 1st at 12:00 AM"),
    (R.Preset("Every hour"), "0 * * * *", "Every hour"),
    (R.Preset("Interval", n=15, unit="minutes"), "*/15 * * * *", "Every 15 minutes"),
    (R.Preset("Interval", n=1, unit="hours"), "0 */1 * * *", "Every 1 hour"),
    (R.Preset("Advanced", cron="@once"), "@once", "One-time"),
])
def test_presets_round_trip_and_read_as_sentences(preset, cron, label):
    assert R.cron_from_preset(preset) == cron
    assert R.format_cron(cron) == label
    back = R.preset_from_cron(cron)
    assert R.cron_from_preset(back) == cron


@pytest.mark.parametrize("cron,label", [
    ("30 7 * * 1,5", "Every Monday and Friday at 7:30 AM"),
    ("0 9 * * 0,6", "Weekends at 9:00 AM"),
    ("0 9 * * 5,6,0", "Friday to Sunday at 9:00 AM"),
    ("0 9 * * 1-3", "Monday to Wednesday at 9:00 AM"),
    ("15 * * * *", "Every hour at :15"),
    ("5 */6 * * *", "Every 6 hours at :05"),
    ("0 10 15 * *", "Monthly on the 15th at 10:00 AM"),
    ("0 10 22 * *", "Monthly on the 22nd at 10:00 AM"),
])
def test_field_shapes_read_as_sentences(cron, label):
    assert R.format_cron(cron) == label


def test_next_cron_matches_the_calendar():
    thu = local(2026, 10, 29, 14, 5)  # a Thursday
    assert R.next_cron("0 8 * * 1-5", thu) == local(2026, 10, 30, 8, 0)
    assert R.next_cron("0 8 * * 1", thu) == local(2026, 11, 2, 8, 0)
    assert R.next_cron("*/15 * * * *", thu) == local(2026, 10, 29, 14, 15)
    assert R.next_cron("0 0 1 * *", thu) == local(2026, 11, 1, 0, 0)
    assert R.next_cron("0 12 29 2 *", thu) == local(2028, 2, 29, 12, 0)  # leap day
    # day-of-month and weekday both restricted: either one matches (standard cron)
    assert R.next_cron("0 9 1 * 1", thu) == local(2026, 11, 1, 9, 0)


def test_uneven_intervals_repeat_from_the_anchor():
    anchor = local(2026, 10, 29, 10, 0)
    assert R.next_run("*/7 * * * *", anchor, local(2026, 10, 29, 10, 50)) == local(2026, 10, 29, 10, 56)
    # even steps follow the clock instead
    assert R.next_run("*/15 * * * *", anchor, local(2026, 10, 29, 10, 50)) == local(2026, 10, 29, 11, 0)


def test_bad_schedules_are_rejected():
    for bad in ["* * *", "61 * * * *", "0 25 * * *", "0 9 * * 8", "a b c d e"]:
        with pytest.raises(ValueError):
            R.parse_cron(bad)


# ---------------------------------------------------------------- scheduler

def make_routine(engine, cron="*/5 * * * *", **kw):
    nxt, anchor = schedule(cron)
    rid = engine.db.save_routine({"name": "Digest", "prompt": "Summarize the day", "cron": cron, "enabled": True,
                                  "options": kw.get("options", {}), "anchor": anchor, "next_run": nxt})
    return engine.db.routine(rid)


def test_due_routine_runs_into_its_own_chat_and_reschedules(engine, fake):
    r = make_routine(engine)
    sched = Scheduler(engine)
    later = r["next_run"] + 1
    assert run(sched.tick(later)) == 1
    r2 = engine.db.routine(r["id"])
    runs = engine.db.routine_runs(r["id"])
    assert runs[0]["status"] == "ok" and runs[0]["message_id"]
    chat = engine.db.chat(r2["chat_id"])
    assert chat["title"] == "Routine: Digest"
    msgs = engine.db.messages(r2["chat_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[0]["content"] == "Summarize the day"
    assert r2["next_run"] > later
    # the next run continues the same chat
    run(sched.tick(r2["next_run"] + 1))
    msgs = engine.db.messages(r2["chat_id"])
    assert len(msgs) == 4 and msgs[2]["parent_id"] == msgs[1]["id"]


def test_nothing_runs_before_it_is_due(engine):
    r = make_routine(engine)
    assert run(Scheduler(engine).tick(r["next_run"] - 30)) == 0


def test_skipped_when_no_model_is_on(engine):
    engine.router = Router({}, {"chat": []}, allow_local=False)
    r = make_routine(engine)
    run(Scheduler(engine).tick(r["next_run"] + 1))
    runs = engine.db.routine_runs(r["id"])
    assert runs[0]["status"] == "skipped" and "No model" in runs[0]["error"]
    assert engine.db.routine(r["id"])["chat_id"] is None  # no demo text was posted


def test_one_time_routine_runs_once(engine):
    r = make_routine(engine, cron="@once")
    sched = Scheduler(engine)
    run(sched.tick(r["next_run"] + 1))
    r2 = engine.db.routine(r["id"])
    assert r2["enabled"] is False and r2["next_run"] is None
    assert run(sched.tick(time.time() + 10_000)) == 0


def test_routine_options_reach_the_turn(engine, fake):
    r = make_routine(engine, options={"agent": "forge"})
    run(Scheduler(engine).tick(r["next_run"] + 1))
    msgs = engine.db.messages(engine.db.routine(r["id"])["chat_id"])
    assert msgs[-1]["meta"]["agent"]["id"] == "forge"


def test_routine_api(engine):
    app = create_app(engine, run_scheduler=False)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        h = {"X-Proxmino": "1"}
        p = c.post("/api/routines/preview", json={"freq": "Weekdays", "time": "7:45 AM"}, headers=h).json()
        assert p["cron"] == "45 7 * * 1-5" and p["label"] == "Weekdays at 7:45 AM" and p["next_run"] > time.time()
        bad = c.post("/api/routines", json={"name": "x", "prompt": "y", "freq": "Advanced", "cron": "99 * * * *"}, headers=h)
        assert bad.status_code == 400
        made = c.post("/api/routines", json={"name": "News", "prompt": "Top AI news", "freq": "Every day", "time": "8:00 AM",
                                             "options": {"search": True}}, headers=h).json()
        assert made["label"] == "Every day at 8:00 AM" and made["options"] == {"search": True}
        off = c.put(f"/api/routines/{made['id']}", json={**{k: made[k] for k in ("name", "prompt", "options")},
                                                         "freq": "Every day", "time": "8:00 AM", "enabled": False}, headers=h).json()
        assert off["enabled"] is False and off["next_run"] is None
        ran = c.post(f"/api/routines/{made['id']}/run", headers=h).json()
        assert ran["status"] == "ok" and ran["chat_id"]
        assert c.get(f"/api/routines/{made['id']}/runs").json()[0]["status"] == "ok"
        assert len(c.get("/api/routines").json()) == 1
        c.delete(f"/api/routines/{made['id']}", headers=h)
        assert c.get("/api/routines").json() == []
