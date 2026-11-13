"""SQLite storage (WAL + FTS5). Stdlib only; migrations via PRAGMA user_version."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

MIGRATIONS = [
    """
    CREATE TABLE chats(
        id TEXT PRIMARY KEY, title TEXT, created_at REAL, updated_at REAL,
        summary TEXT, summary_upto TEXT);
    CREATE TABLE messages(
        id TEXT PRIMARY KEY, chat_id TEXT REFERENCES chats(id) ON DELETE CASCADE,
        parent_id TEXT, role TEXT, content TEXT, thinking TEXT, meta TEXT,
        created_at REAL);
    CREATE INDEX messages_chat ON messages(chat_id, created_at);
    CREATE TABLE memory(id INTEGER PRIMARY KEY, text TEXT NOT NULL, created_at REAL);
    CREATE VIRTUAL TABLE memory_fts USING fts5(text, content='memory', content_rowid='id');
    CREATE TRIGGER memory_ai AFTER INSERT ON memory BEGIN
        INSERT INTO memory_fts(rowid, text) VALUES (new.id, new.text); END;
    CREATE TRIGGER memory_ad AFTER DELETE ON memory BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
    CREATE TABLE search_cache(key TEXT PRIMARY KEY, results TEXT, expires_at REAL);
    CREATE TABLE feedback(message_id TEXT, kind TEXT, note TEXT, created_at REAL);
    """,
    "ALTER TABLE chats ADD COLUMN model TEXT",  # 2: a provider:model pinned to the conversation
    """
    CREATE TABLE routines(
        id TEXT PRIMARY KEY, name TEXT, prompt TEXT, cron TEXT, enabled INTEGER DEFAULT 1,
        chat_id TEXT, options TEXT, anchor REAL, next_run REAL, created_at REAL, updated_at REAL);
    CREATE TABLE routine_runs(
        id TEXT PRIMARY KEY, routine_id TEXT REFERENCES routines(id) ON DELETE CASCADE,
        status TEXT, started_at REAL, finished_at REAL, error TEXT, message_id TEXT);
    CREATE INDEX routine_runs_by_routine ON routine_runs(routine_id, started_at);
    """,  # 3: scheduled routines (adapted from Rakazo, Apache-2.0)
    "ALTER TABLE memory ADD COLUMN agent TEXT",  # 4: per-agent memory (NULL = shared by every agent)
    """
    CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='rowid');
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content); END;
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content); END;
    CREATE TRIGGER messages_au AFTER UPDATE OF content ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
        INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content); END;
    INSERT INTO messages_fts(messages_fts) VALUES ('rebuild');
    CREATE TABLE scratchpad(
        id TEXT PRIMARY KEY, agent TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
        notes TEXT NOT NULL DEFAULT '', created_at REAL, updated_at REAL);
    CREATE INDEX scratchpad_by_agent ON scratchpad(agent, status, updated_at);
    """,  # 5: search across chats; scratchpad (from Rakazo, Apache-2.0)
    """
    CREATE TABLE artifacts(
        id TEXT PRIMARY KEY, chat_id TEXT, title TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1, created_at REAL, updated_at REAL);
    CREATE INDEX artifacts_by_chat ON artifacts(chat_id, updated_at);
    CREATE TABLE knowledge_docs(id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT, chars INTEGER, chunks INTEGER, added_at REAL);
    CREATE VIRTUAL TABLE knowledge_fts USING fts5(doc_id UNINDEXED, seq UNINDEXED, text);
    """,  # 6: artifacts and knowledge (from Rakazo, Apache-2.0)
    """
    ALTER TABLE chats ADD COLUMN agent TEXT;
    ALTER TABLE chats ADD COLUMN members TEXT;
    """,  # 7: a chat with one bot (agent) or a group of bots (members, JSON list of agent ids)
    """
    CREATE TABLE outbox(id TEXT PRIMARY KEY, kind TEXT NOT NULL, target TEXT, text TEXT NOT NULL,
        status TEXT NOT NULL, result TEXT, created_at REAL);
    CREATE INDEX outbox_by_time ON outbox(created_at);
    """,  # 8: everything sent outside this computer (emails, posts), with the outcome
    """
    CREATE TABLE inbox(id TEXT PRIMARY KEY, source TEXT NOT NULL, external_id TEXT NOT NULL, thread TEXT, sender TEXT,
        subject TEXT, text TEXT, priority TEXT, needs_reply INTEGER, summary TEXT, draft TEXT, status TEXT NOT NULL,
        meta TEXT, created_at REAL, updated_at REAL, UNIQUE(source, external_id));
    CREATE INDEX inbox_by_status ON inbox(status, created_at);
    CREATE TABLE kv(key TEXT PRIMARY KEY, value TEXT);
    """,  # 9: incoming messages (email, Discord) with drafts; small settings the scheduler remembers
    """
    CREATE INDEX messages_role_time ON messages(role, created_at);
    CREATE INDEX routine_runs_by_time ON routine_runs(started_at);
    """,  # 10: faster Stats, Usage and overview queries (they filter answers by time)
]

ARTIFACT_KINDS = ("html", "svg", "markdown", "csv", "text", "code")

SCRATCH_STATUSES = ("open", "parked", "done")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class DB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.lock:
                self.conn.executescript(f"BEGIN;{sql};PRAGMA user_version={i};COMMIT;")

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            return self.conn.execute(sql, args).lastrowid

    # chats
    def create_chat(self, title: str, agent: str | None = None, members: list[str] | None = None) -> str:
        cid, now = new_id(), time.time()
        self.x("INSERT INTO chats(id,title,created_at,updated_at,agent,members) VALUES(?,?,?,?,?,?)",
               (cid, title, now, now, agent or None, json.dumps(members) if members else None))
        return cid

    @staticmethod
    def _chat_row(r) -> dict:
        d = dict(r)
        d["members"] = json.loads(d["members"]) if d.get("members") else []
        return d

    def chat(self, cid: str) -> dict | None:
        rows = self.q("SELECT * FROM chats WHERE id=?", (cid,))
        return self._chat_row(rows[0]) if rows else None

    def chats(self, limit: int = 200) -> list[dict]:
        """Newest first, with the last message's text and speaker for the sidebar preview."""
        rows = self.q(
            "SELECT c.id, c.title, c.updated_at, c.agent, c.members, "
            "(SELECT substr(m.content, 1, 140) FROM messages m WHERE m.chat_id = c.id ORDER BY m.created_at DESC LIMIT 1) AS last, "
            "(SELECT json_extract(m.meta, '$.agent.id') FROM messages m WHERE m.chat_id = c.id AND m.role = 'assistant' "
            " ORDER BY m.created_at DESC LIMIT 1) AS last_agent "
            "FROM chats c ORDER BY c.updated_at DESC LIMIT ?", (limit,))
        return [self._chat_row(r) for r in rows]

    # inbox
    def inbox_seen(self, source: str, external_id: str) -> bool:
        return bool(self.q("SELECT 1 FROM inbox WHERE source=? AND external_id=?", (source, external_id)))

    def inbox_add(self, item: dict) -> str:
        iid, now = new_id(), time.time()
        self.x("INSERT OR IGNORE INTO inbox VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (iid, item["source"], item["external_id"], item.get("thread"), item.get("sender"), item.get("subject"),
                item.get("text"), item.get("priority"), int(bool(item.get("needs_reply"))), item.get("summary"),
                item.get("draft"), item.get("status", "new"), json.dumps(item.get("meta") or {}), now, now))
        return iid

    @staticmethod
    def _inbox_row(r) -> dict:
        d = dict(r)
        d["meta"], d["needs_reply"] = json.loads(d["meta"] or "{}"), bool(d["needs_reply"])
        return d

    def inbox(self, status: str | None = None, limit: int = 200) -> list[dict]:
        order = "CASE priority WHEN 'urgent' THEN 0 WHEN 'normal' THEN 1 WHEN 'low' THEN 2 ELSE 3 END, created_at DESC"
        if status:
            rows = self.q(f"SELECT * FROM inbox WHERE status=? ORDER BY {order} LIMIT ?", (status, limit))
        else:
            rows = self.q(f"SELECT * FROM inbox ORDER BY {order} LIMIT ?", (limit,))
        return [self._inbox_row(r) for r in rows]

    def inbox_get(self, iid: str) -> dict | None:
        rows = self.q("SELECT * FROM inbox WHERE id=?", (iid,))
        return self._inbox_row(rows[0]) if rows else None

    def inbox_update(self, iid: str, **fields) -> None:
        allowed = {k: v for k, v in fields.items() if k in ("draft", "status", "priority", "summary")}
        if allowed:
            sets = ", ".join(f"{k}=?" for k in allowed)
            self.x(f"UPDATE inbox SET {sets}, updated_at=? WHERE id=?", (*allowed.values(), time.time(), iid))

    def inbox_counts_since(self, since: float) -> dict:
        rows = self.q("SELECT status, count(*) AS n FROM inbox WHERE created_at>=? GROUP BY status", (since,))
        return {r["status"]: r["n"] for r in rows}

    def kv_get(self, key: str, default=None):
        rows = self.q("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def kv_set(self, key: str, value) -> None:
        self.x("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def outbox_add(self, kind: str, target: str, text: str, status: str, result: str) -> None:
        self.x("INSERT INTO outbox VALUES(?,?,?,?,?,?,?)", (new_id(), kind, target, text, status, result[:500], time.time()))

    def outbox(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM outbox ORDER BY created_at DESC LIMIT ?", (limit,))]

    def outbox_count_since(self, kind: str, since: float) -> int:
        return self.q("SELECT count(*) FROM outbox WHERE kind=? AND status='sent' AND created_at>=?", (kind, since))[0][0]

    def set_chat_members(self, cid: str, members: list[str]) -> None:
        self.x("UPDATE chats SET members=? WHERE id=?", (json.dumps(members) if members else None, cid))

    def set_chat_model(self, cid: str, model: str | None) -> None:
        self.x("UPDATE chats SET model=? WHERE id=?", (model or None, cid))

    def rename_chat(self, cid: str, title: str) -> None:
        self.x("UPDATE chats SET title=? WHERE id=?", (title, cid))

    def delete_chat(self, cid: str) -> None:
        self.x("DELETE FROM chats WHERE id=?", (cid,))

    def set_summary(self, cid: str, summary: str, upto: str) -> None:
        self.x("UPDATE chats SET summary=?, summary_upto=? WHERE id=?", (summary, upto, cid))

    # messages
    def add_message(self, chat_id: str, parent_id: str | None, role: str, content: str,
                    thinking: str = "", meta: dict | None = None, mid: str | None = None) -> str:
        mid, now = mid or new_id(), time.time()
        self.x("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)",
               (mid, chat_id, parent_id, role, content, thinking, json.dumps(meta or {}), now))
        self.x("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
        return mid

    def message(self, mid: str) -> dict | None:
        rows = self.q("SELECT * FROM messages WHERE id=?", (mid,))
        return _msg(rows[0]) if rows else None

    def messages(self, chat_id: str) -> list[dict]:
        return [_msg(r) for r in self.q(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,))]

    def path_to(self, mid: str | None) -> list[dict]:
        """Messages from the root to `mid` (inclusive) along parent links."""
        if not mid:
            return []
        rows = self.q("""
            WITH RECURSIVE p(id, parent_id, depth) AS (
                SELECT id, parent_id, 0 FROM messages WHERE id=?
                UNION ALL SELECT m.id, m.parent_id, p.depth+1 FROM messages m JOIN p ON m.id=p.parent_id)
            SELECT m.* FROM p JOIN messages m ON m.id=p.id ORDER BY p.depth DESC""", (mid,))
        return [_msg(r) for r in rows]

    # memory
    def add_memory(self, text: str, agent: str | None = None) -> int:
        return self.x("INSERT INTO memory(text,created_at,agent) VALUES(?,?,?)", (text.strip(), time.time(), agent or None))

    def memories(self, agent: str | None = None, visible_to: bool = False) -> list[dict]:
        """All memories; with visible_to, only the shared ones plus those of `agent`."""
        if visible_to:
            rows = self.q("SELECT id,text,created_at,agent FROM memory WHERE agent IS NULL OR agent=? ORDER BY id", (agent,))
        else:
            rows = self.q("SELECT id,text,created_at,agent FROM memory ORDER BY id")
        return [dict(r) for r in rows]

    def search_memory(self, query: str, limit: int = 8, agent: str | None = None) -> list[str]:
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if len(t) > 2]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms[:20])
        rows = self.q("SELECT f.text FROM memory_fts f JOIN memory m ON m.id = f.rowid WHERE memory_fts MATCH ? "
                      "AND (m.agent IS NULL OR m.agent = ?) ORDER BY rank LIMIT ?", (match, agent, limit))
        return [r["text"] for r in rows]

    def delete_memory(self, mem_id: int | None = None) -> None:
        if mem_id is None:
            self.x("DELETE FROM memory")
        else:
            self.x("DELETE FROM memory WHERE id=?", (mem_id,))

    # search cache
    def cache_get(self, key: str) -> Any:
        rows = self.q("SELECT results FROM search_cache WHERE key=? AND expires_at>?", (key, time.time()))
        return json.loads(rows[0]["results"]) if rows else None

    def cache_put(self, key: str, value: Any, ttl_s: float) -> None:
        self.x("INSERT OR REPLACE INTO search_cache VALUES(?,?,?)",
               (key, json.dumps(value), time.time() + ttl_s))

    def usage(self, days: int = 30) -> list[dict]:
        """Answers per model over the last `days`, from saved assistant messages (activity, not billing)."""
        since = time.time() - days * 86400
        rows = self.q("SELECT json_extract(meta, '$.model') AS model, COUNT(*) AS answers, "
                      "AVG(json_extract(meta, '$.latency_ms')) AS avg_ms, "
                      "SUM(length(content)) AS chars, MAX(created_at) AS last, "
                      "SUM(json_extract(meta, '$.tokens_in')) AS tin, SUM(json_extract(meta, '$.tokens_out')) AS tout "
                      "FROM messages WHERE role = 'assistant' AND created_at >= ? GROUP BY model ORDER BY answers DESC",
                      (since,))
        return [{"model": r["model"] or "none", "answers": r["answers"], "avg_ms": int(r["avg_ms"] or 0),
                 "chars": r["chars"] or 0, "last": r["last"], "tokens_in": r["tin"] or 0, "tokens_out": r["tout"] or 0}
                for r in rows]

    # ---------------------------------------------------------- search across chats

    def search_messages(self, query: str, limit: int = 30) -> list[dict]:
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if len(t) > 1]
        if not terms:
            return []
        match = " ".join(f'"{t}"*' for t in terms[:12])  # every word, prefix match on each
        rows = self.q(
            "SELECT m.id, m.chat_id, m.role, m.created_at, c.title, "
            "snippet(messages_fts, 0, '[[', ']]', ' ... ', 14) AS snip "
            "FROM messages_fts JOIN messages m ON m.rowid = messages_fts.rowid JOIN chats c ON c.id = m.chat_id "
            "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?", (match, limit))
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- scratchpad (Rakazo: a bot's own open work)

    def scratch_list(self, agent: str | None = None, include_done: bool = False) -> list[dict]:
        where, args = [], []
        if agent:
            where.append("agent=?")
            args.append(agent)
        if not include_done:
            where.append("status IN ('open','parked')")
        sql = "SELECT * FROM scratchpad" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY updated_at DESC"
        return [dict(r) for r in self.q(sql, tuple(args))]

    def scratch_get(self, iid: str) -> dict | None:
        rows = self.q("SELECT * FROM scratchpad WHERE id=?", (iid,))
        return dict(rows[0]) if rows else None

    def scratch_add(self, agent: str, title: str, status: str = "open", notes: str = "") -> dict:
        title = title.strip()[:200]
        if not title:
            raise ValueError("A scratchpad item needs a title.")
        iid, now = new_id()[:8], time.time()
        self.x("INSERT INTO scratchpad(id, agent, title, status, notes, created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
               (iid, agent, title, status if status in SCRATCH_STATUSES else "open", notes[:4000], now, now))
        return self.scratch_get(iid)

    def scratch_update(self, iid: str, title: str | None = None, status: str | None = None, notes: str | None = None) -> dict:
        item = self.scratch_get(iid)
        if not item:
            raise ValueError(f"No scratchpad item {iid}.")
        if status is not None and status not in SCRATCH_STATUSES:
            raise ValueError("status must be open, parked or done.")
        self.x("UPDATE scratchpad SET title=?, status=?, notes=?, updated_at=? WHERE id=?",
               ((title or item["title"]).strip()[:200], status or item["status"],
                (notes if notes is not None else item["notes"])[:4000], time.time(), iid))
        return self.scratch_get(iid)

    def scratch_remove(self, iid: str) -> bool:
        return bool(self.q("DELETE FROM scratchpad WHERE id=? RETURNING id", (iid,)))

    # ---------------------------------------------------------- artifacts (Rakazo: things a bot makes)

    def artifact_save(self, title: str, kind: str, content: str, chat_id: str | None = None, aid: str | None = None) -> dict:
        kind = kind if kind in ARTIFACT_KINDS else "text"
        if not content.strip():
            raise ValueError("An artifact needs content.")
        if len(content) > 500_000:
            raise ValueError("An artifact can be at most 500,000 characters.")
        now = time.time()
        old = self.artifact(aid) if aid else None
        if old:
            self.x("UPDATE artifacts SET title=?, kind=?, content=?, version=version+1, updated_at=? WHERE id=?",
                   ((title or old["title"]).strip()[:120], kind, content, now, aid))
            return self.artifact(aid)
        aid = new_id()[:10]
        self.x("INSERT INTO artifacts(id, chat_id, title, kind, content, created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
               (aid, chat_id, (title or "Untitled").strip()[:120], kind, content, now, now))
        return self.artifact(aid)

    def artifact(self, aid: str | None) -> dict | None:
        rows = self.q("SELECT * FROM artifacts WHERE id=?", (aid,)) if aid else []
        return dict(rows[0]) if rows else None

    def artifacts(self, chat_id: str | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT id, chat_id, title, kind, version, length(content) AS size, created_at, updated_at FROM artifacts"
        rows = self.q(sql + (" WHERE chat_id=?" if chat_id else "") + " ORDER BY updated_at DESC LIMIT ?",
                      ((chat_id, limit) if chat_id else (limit,)))
        return [dict(r) for r in rows]

    def artifact_delete(self, aid: str) -> None:
        self.x("DELETE FROM artifacts WHERE id=?", (aid,))

    # ---------------------------------------------------------- knowledge (local documents the model can search)

    def knowledge_add(self, name: str, text: str, kind: str = "text") -> dict:
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("The document has no text.")
        did = new_id()[:10]
        self.x("INSERT INTO knowledge_docs(id, name, kind, chars, chunks, added_at) VALUES(?,?,?,?,?,?)",
               (did, name.strip()[:200] or "Untitled", kind, len(text), len(chunks), time.time()))
        with self.lock:
            self.conn.executemany("INSERT INTO knowledge_fts(doc_id, seq, text) VALUES(?,?,?)",
                                  [(did, i, c) for i, c in enumerate(chunks)])
        return dict(self.q("SELECT * FROM knowledge_docs WHERE id=?", (did,))[0])

    def knowledge_docs(self) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM knowledge_docs ORDER BY added_at DESC")]

    def knowledge_delete(self, did: str) -> None:
        self.x("DELETE FROM knowledge_fts WHERE doc_id=?", (did,))
        self.x("DELETE FROM knowledge_docs WHERE id=?", (did,))

    def knowledge_search(self, query: str, limit: int = 5) -> list[dict]:
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if len(t) > 2]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms[:20])
        rows = self.q("SELECT k.doc_id, k.seq, k.text, d.name FROM knowledge_fts k JOIN knowledge_docs d ON d.id = k.doc_id "
                      "WHERE knowledge_fts MATCH ? ORDER BY rank LIMIT ?", (match, limit))
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- routines

    def routines(self) -> list[dict]:
        return [_routine(r) for r in self.q("SELECT * FROM routines ORDER BY created_at")]

    def routine(self, rid: str) -> dict | None:
        rows = self.q("SELECT * FROM routines WHERE id=?", (rid,))
        return _routine(rows[0]) if rows else None

    def save_routine(self, r: dict) -> str:
        rid = r.get("id") or new_id()
        now = time.time()
        self.x("INSERT INTO routines(id, name, prompt, cron, enabled, chat_id, options, anchor, next_run, created_at, updated_at) "
               "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, prompt=excluded.prompt, "
               "cron=excluded.cron, enabled=excluded.enabled, chat_id=excluded.chat_id, options=excluded.options, "
               "anchor=excluded.anchor, next_run=excluded.next_run, updated_at=excluded.updated_at",
               (rid, r["name"], r["prompt"], r["cron"], int(bool(r.get("enabled", True))), r.get("chat_id"),
                json.dumps(r.get("options") or {}), r.get("anchor"), r.get("next_run"), r.get("created_at") or now, now))
        return rid

    def delete_routine(self, rid: str) -> None:
        self.x("DELETE FROM routines WHERE id=?", (rid,))

    def due_routines(self, now: float) -> list[dict]:
        return [_routine(r) for r in self.q(
            "SELECT * FROM routines WHERE enabled=1 AND next_run IS NOT NULL AND next_run<=? ORDER BY next_run", (now,))]

    def add_routine_run(self, routine_id: str) -> str:
        rid = new_id()
        self.x("INSERT INTO routine_runs(id, routine_id, status, started_at) VALUES(?,?,?,?)",
               (rid, routine_id, "running", time.time()))
        return rid

    def finish_routine_run(self, run_id: str, status: str, error: str | None = None, message_id: str | None = None) -> None:
        self.x("UPDATE routine_runs SET status=?, finished_at=?, error=?, message_id=? WHERE id=?",
               (status, time.time(), error, message_id, run_id))

    def routine_runs(self, routine_id: str, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM routine_runs WHERE routine_id=? ORDER BY started_at DESC LIMIT ?",
                                        (routine_id, limit))]

    def add_feedback(self, message_id: str, kind: str, note: str = "") -> None:
        self.x("INSERT INTO feedback VALUES(?,?,?,?)", (message_id, kind, note, time.time()))


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Split a document into passages of about `size` characters, at paragraph or sentence ends when possible."""
    text = text.replace("\r\n", "\n").strip()
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
            end = cut + 1 if cut > 0 else end
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def _routine(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["enabled"] = bool(d["enabled"])
    d["options"] = json.loads(d["options"] or "{}")
    return d


def _msg(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["meta"] = json.loads(d["meta"] or "{}")
    return d
