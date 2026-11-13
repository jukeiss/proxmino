"""Routines: prompts that run on a schedule and post their answers into their own chat.

The schedule presets, the plain-English labels and the uneven-interval rule are translated from
Rakazo's packages/core/src/cron.ts (https://github.com/elie222/rakazo, Apache License 2.0;
see third_party/rakazo/NOTICE.md). Changes: Python instead of TypeScript, English labels only,
and a small 5-field cron evaluator here in place of the `croner` library.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

ONCE = "@once"
WEEKDAYS = "1-5"
WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
WEEK_ORDER = [1, 2, 3, 4, 5, 6, 0]  # Monday first, so Friday to Sunday reads as one run
FREQS = ["Every hour", "Every day", "Weekdays", "Every week", "Every month", "Interval", "Advanced"]
INTERVAL_MAX = {"minutes": 59, "hours": 23, "days": 31}
SINGULAR = {"minutes": "minute", "hours": "hour", "days": "day"}


@dataclass
class Preset:
    freq: str = "Every day"
    n: int = 3
    unit: str = "minutes"
    time: str = "9:00 AM"
    cron: str = ""


# ------------------------------------------------------------------ presets and labels

def cron_from_preset(p: Preset) -> str:
    if p.freq == "Advanced":
        c = (p.cron or "").strip()
        return ONCE if c == ONCE else c or "*/3 * * * *"
    if p.freq == "Every hour":
        return "0 * * * *"
    if p.freq == "Interval":
        n = p.n if p.n and p.n > 0 else 5
        if p.unit == "days":
            return f"0 0 */{n} * *"
        if p.unit == "hours":
            return f"0 */{n} * * *"
        return f"*/{n} * * * *"
    hour, minute = parse_clock(p.time or "9:00 AM")
    if p.freq == "Weekdays":
        return f"{minute} {hour} * * {WEEKDAYS}"
    if p.freq == "Every week":
        return f"{minute} {hour} * * 1"
    if p.freq == "Every month":
        return f"{minute} {hour} 1 * *"
    return f"{minute} {hour} * * *"


def preset_from_cron(cron: str) -> Preset:
    base, c = Preset(), cron.strip()
    if c == ONCE:
        return Preset(freq="Advanced", cron=ONCE)
    parts = c.split()
    if len(parts) < 5:
        return Preset(freq="Advanced", cron=c)
    minute, hour, day, month, dow = parts[:5]
    if month != "*":
        return Preset(freq="Advanced", cron=c)
    ms, hs, ds = _step(minute), _step(hour), _step(day)
    if minute == "0" and hour == "*" and day == "*" and dow == "*":
        return Preset(freq="Every hour")
    if ms and hour == "*" and day == "*" and dow == "*":
        return Preset(freq="Interval", n=ms, unit="minutes")
    if minute == "0" and hs and day == "*" and dow == "*":
        return Preset(freq="Interval", n=hs, unit="hours")
    if minute == "0" and hour == "0" and ds and dow == "*":
        return Preset(freq="Interval", n=ds, unit="days")
    if not _is_clock(minute, hour):
        return Preset(freq="Advanced", cron=c)
    time = format_clock(int(hour), int(minute))
    if day == "*" and dow == WEEKDAYS:
        return Preset(freq="Weekdays", time=time)
    if day == "*" and dow == "1":
        return Preset(freq="Every week", time=time)
    if day == "1" and dow == "*":
        return Preset(freq="Every month", time=time)
    if day == "*" and dow == "*":
        return Preset(freq="Every day", time=time)
    return Preset(freq="Advanced", cron=c)


def format_schedule(p: Preset) -> str:
    if p.freq == "Interval":
        return f"Every {p.n} {SINGULAR[p.unit] if p.n == 1 else p.unit}"
    if p.freq == "Every hour":
        return "Every hour"
    if p.freq == "Advanced":
        return f"Cron {p.cron or '*/3 * * * *'}"
    if p.freq == "Weekdays":
        return f"Weekdays at {p.time}"
    if p.freq == "Every week":
        return f"Every Monday at {p.time}"
    if p.freq == "Every month":
        return f"Monthly on the 1st at {p.time}"
    return f"Every day at {p.time}"


def format_cron(cron: str) -> str:
    """A schedule as a short sentence, e.g. "Weekdays at 8:00 AM"."""
    if cron.strip() == ONCE:
        return "One-time"
    return _describe_fields(cron) or format_schedule(preset_from_cron(cron))


def _describe_fields(cron: str) -> str | None:
    parts = cron.strip().split()
    if len(parts) != 5 or parts[3] != "*":
        return None
    minute, hour, day, _, dow = parts
    if hour == "*" and day == "*" and dow == "*" and minute.isdigit():
        m = int(minute)
        return f"Every hour at :{m:02d}" if 1 <= m <= 59 else None
    hours = _step(hour)
    if hours and day == "*" and dow == "*" and minute.isdigit():
        m = int(minute)
        return f"Every {hours} hours at :{m:02d}" if 2 <= hours < 24 and 24 % hours == 0 and 1 <= m <= 59 else None
    if not _is_clock(minute, hour):
        return None
    time = format_clock(int(hour), int(minute))
    if day == "*" and dow != "*":
        days = _weekdays(dow)
        return f"{_describe_weekdays(days)} at {time}" if days else None
    if dow == "*" and day.isdigit() and 1 <= int(day) <= 31:
        return f"Monthly on the {_ordinal(int(day))} at {time}"
    return None


def _weekdays(field_: str) -> set[int] | None:
    days: set[int] = set()
    for part in field_.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            return None
        a = int(m.group(1))
        b = a if m.group(2) is None else int(m.group(2))
        if a > b or b > 7:
            return None
        days |= {d % 7 for d in range(a, b + 1)}
    return days


def _describe_weekdays(days: set[int]) -> str:
    if len(days) == 7:
        return "Every day"
    if len(days) == 5 and not days & {0, 6}:
        return "Weekdays"
    if days == {0, 6}:
        return "Weekends"
    ordered = [d for d in WEEK_ORDER if d in days]
    names = [WEEKDAY_NAMES[d] for d in ordered]
    span = WEEK_ORDER.index(ordered[-1]) - WEEK_ORDER.index(ordered[0])
    if len(names) >= 3 and span == len(names) - 1:
        return f"{names[0]} to {names[-1]}"
    return f"Every {names[0]}" if len(names) == 1 else f"Every {', '.join(names[:-1])} and {names[-1]}"


def _ordinal(n: int) -> str:
    if 11 <= n <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][n % 10] if n % 10 < 4 else 'th'}"


def parse_clock(time: str) -> tuple[int, int]:
    raw_h, _, rest = time.partition(":")
    minute = int((rest or "00")[:2])
    hour = int(raw_h)
    if re.search(r"pm", time, re.I) and hour < 12:
        hour += 12
    if re.search(r"am", time, re.I) and hour == 12:
        hour = 0
    return hour, minute


def format_clock(hour: int, minute: int) -> str:
    return f"{12 if hour % 12 == 0 else hour % 12}:{minute:02d} {'PM' if hour >= 12 else 'AM'}"


def _step(expr: str) -> int | None:
    m = re.fullmatch(r"\*/(\d+)", expr)
    return int(m.group(1)) if m and int(m.group(1)) > 0 else None


def _is_clock(minute: str, hour: str) -> bool:
    return minute.isdigit() and int(minute) <= 59 and hour.isdigit() and int(hour) <= 23


# ------------------------------------------------------------------ next run (own evaluator)

RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


@dataclass
class CronSpec:
    minutes: set[int] = field(default_factory=set)
    hours: set[int] = field(default_factory=set)
    days: set[int] = field(default_factory=set)
    months: set[int] = field(default_factory=set)
    dows: set[int] = field(default_factory=set)
    day_any: bool = True
    dow_any: bool = True


def parse_cron(cron: str) -> CronSpec:
    parts = cron.strip().split()
    if len(parts) != 5:
        raise ValueError("A schedule needs five fields: minute hour day month weekday.")
    sets = []
    for text, (lo, hi) in zip(parts, RANGES):
        vals: set[int] = set()
        for item in text.split(","):
            m = re.fullmatch(r"(\*|\d+)(?:-(\d+))?(?:/(\d+))?", item)
            if not m:
                raise ValueError(f"Cannot read '{item}' in '{cron}'.")
            start, end, step = m.group(1), m.group(2), m.group(3)
            a = lo if start == "*" else int(start)
            b = (hi if start == "*" else a) if end is None else int(end)
            if step and start != "*" and end is None:
                b = hi
            s = int(step) if step else 1
            if not (lo <= a <= hi and lo <= b <= hi) or a > b or s < 1:
                raise ValueError(f"'{item}' is out of range in '{cron}'.")
            vals |= set(range(a, b + 1, s))
        sets.append(vals)
    dows = {d % 7 for d in sets[4]}
    return CronSpec(sets[0], sets[1], sets[2], sets[3], dows, parts[2] == "*", parts[4] == "*")


def _day_matches(spec: CronSpec, d: dt.datetime) -> bool:
    dow = (d.weekday() + 1) % 7  # cron: 0 = Sunday
    dom_ok, dow_ok = d.day in spec.days, dow in spec.dows
    if spec.day_any or spec.dow_any:  # standard cron: when both are restricted, either one matches
        return dom_ok and dow_ok
    return dom_ok or dow_ok


def next_cron(cron: str, after: dt.datetime) -> dt.datetime:
    """The first minute strictly after `after` that the schedule matches, in local wall-clock time
    (so 8:00 stays 8:00 across daylight-saving changes). Returns a timezone-aware datetime."""
    spec = parse_cron(cron)
    wall = after.astimezone().replace(tzinfo=None) if after.tzinfo else after
    t = (wall + dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
    limit = wall + dt.timedelta(days=366 * 5)
    while t <= limit:
        if t.month not in spec.months:
            t = (t.replace(day=1, hour=0, minute=0) + dt.timedelta(days=32)).replace(day=1)
            continue
        if not _day_matches(spec, t):
            t = (t + dt.timedelta(days=1)).replace(hour=0, minute=0)
            continue
        if t.hour not in spec.hours:
            t = (t + dt.timedelta(hours=1)).replace(minute=0)
            continue
        if t.minute not in spec.minutes:
            t += dt.timedelta(minutes=1)
            continue
        return t.astimezone()
    raise ValueError(f"'{cron}' never runs.")


def _uneven(cron: str) -> Preset | None:
    """An interval that does not divide its unit (every 7 minutes, every 5 hours, every 3 days)."""
    p = preset_from_cron(cron)
    if (len(cron.split()) != 5 or p.freq != "Interval" or p.n > INTERVAL_MAX[p.unit]
            or (p.unit == "minutes" and 60 % p.n == 0) or (p.unit == "hours" and 24 % p.n == 0)
            or (p.unit == "days" and p.n == 1)):
        return None
    return p


def next_run(cron: str, anchor: dt.datetime, now: dt.datetime) -> dt.datetime | None:
    """Next run after now. Uneven intervals repeat from the anchor instead of resetting at field
    boundaries (as in Rakazo); everything else follows the calendar."""
    if cron.strip() == ONCE:
        return None
    start = max(anchor, now)
    p = _uneven(cron)
    if not p:
        return next_cron(cron, start)
    gap = dt.timedelta(minutes=p.n) if p.unit == "minutes" else dt.timedelta(hours=p.n) if p.unit == "hours" else dt.timedelta(days=p.n)
    steps = int((start - anchor) / gap) + 1
    return anchor + steps * gap


def local_now() -> dt.datetime:
    return dt.datetime.now().astimezone()
