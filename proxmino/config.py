"""Configuration: config.yaml merged with data/settings.json overrides and .env secrets."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PROXMINO_DATA") or os.environ.get("PRAXIS_DATA") or ROOT / "data")  # PRAXIS_DATA: old name


def migrate_old_names() -> None:
    """The app used to be called Praxis: move its database and bots folder to the new names once, keeping everything."""
    if (DATA_DIR / "praxis.db").exists() and not (DATA_DIR / "proxmino.db").exists():
        for suffix in ("", "-wal", "-shm"):  # the database and its write-ahead files move together
            if (DATA_DIR / f"praxis.db{suffix}").exists():
                (DATA_DIR / f"praxis.db{suffix}").rename(DATA_DIR / f"proxmino.db{suffix}")
    old_dir, new_dir = Path.home() / ".praxis", Path.home() / ".proxmino"
    if old_dir.is_dir() and not new_dir.exists():
        old_dir.rename(new_dir)
ENV_FILE = Path(os.environ.get("PROXMINO_ENV_FILE") or ROOT / ".env")  # a server keeps it in its data volume
SETTINGS_FILE = DATA_DIR / "settings.json"


def lock_down(path: Path = DATA_DIR) -> None:
    """Chats, keys and logins are private: only this user may read the data folder (owner-only permissions)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
        for f in path.iterdir():
            if f.is_file():
                f.chmod(0o600)
    except OSError:
        pass

# Keys the Settings panel may change. Everything else is file-only.
EDITABLE = {"defaults", "tiers", "search"}


_FROM_DOTENV: set[str] = set()  # names this app set from .env, so a reload can update or clear them


def _load_dotenv(path: Path) -> None:
    """Load .env. Variables already set by the shell win; ones we loaded earlier are refreshed,
    so keys added or removed while the app runs take effect on reload."""
    found: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            found[k.strip()] = v.strip().strip('"').strip("'")
    for k in list(_FROM_DOTENV):
        if not found.get(k):
            os.environ.pop(k, None)
            _FROM_DOTENV.discard(k)
    for k, v in found.items():
        if v and (k not in os.environ or k in _FROM_DOTENV):
            os.environ[k] = v
            _FROM_DOTENV.add(k)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, path: Path | None = None, overrides: dict | None = None):
        _load_dotenv(ENV_FILE)
        self.path = path or ROOT / "config.yaml"
        self.base = yaml.safe_load(self.path.read_text()) if self.path.exists() else {}
        self.extra = overrides or {}
        self.reload()

    def reload(self) -> None:
        _load_dotenv(ENV_FILE)
        saved = {}
        if SETTINGS_FILE.exists() and not self.extra:
            try:
                saved = json.loads(SETTINGS_FILE.read_text())
            except json.JSONDecodeError:
                saved = {}
        self.data = _merge(_merge(self.base, saved), self.extra)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def save_settings(self, patch: dict) -> None:
        patch = {k: v for k, v in patch.items() if k in EDITABLE}
        saved = {}
        if SETTINGS_FILE.exists():
            try:
                saved = json.loads(SETTINGS_FILE.read_text())
            except json.JSONDecodeError:
                saved = {}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(_merge(saved, patch), indent=2))
        self.reload()

    @property
    def product_name(self) -> str:
        return self.get("product_name", "Proxmino")
