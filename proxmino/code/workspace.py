"""A project folder the coding agent works in: bounded paths, capped outputs, sandboxed commands."""
from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import edits as E

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", "dist", "build", ".next", ".tox", "target", ".idea", ".vscode"}
MAX_READ_LINES = 400
MAX_CHARS = 20_000
MAX_GREP = 80
MAX_FILE_BYTES = 2_000_000
TEST_DIRS = {"tests", "test", "__tests__", "spec", "testdata"}
TEST_FILES = ("test_*.py", "*_test.py", "conftest.py", "*_test.go", "*.test.js", "*.test.ts", "*.test.jsx",
              "*.test.tsx", "*.spec.js", "*.spec.ts", "*_spec.rb", "*Test.java", "*Tests.swift")
_range = range  # read_file has an argument named range


class ToolError(Exception):
    """An error the model can act on; the message says how to fix the call."""


@dataclass
class CmdResult:
    command: str
    exit_code: int | None
    output: str
    seconds: float
    timed_out: bool
    sandboxed: bool

    def text(self) -> str:
        head = f"$ {self.command}\nexit {self.exit_code}" + (" (timed out)" if self.timed_out else "") + f" in {self.seconds}s"
        return head + ("\n" + self.output if self.output else "")


def clip(s: str, limit: int = MAX_CHARS) -> str:
    return s if len(s) <= limit else s[:limit] + f"\n... ({len(s) - limit} more characters; narrow the request)"


class Workspace:
    def __init__(self, root: str | Path, extra_read: list[str] | None = None, test_cmd: str | None = None,
                 allow_test_edits: bool = False):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ToolError(f"Workspace {self.root} is not a folder.")
        home = Path(os.path.realpath(os.path.expanduser("~")))
        if self.root == home or self.root in home.parents or len(self.root.parts) < 3:
            raise ToolError(f"{self.root} is too broad for a workspace (it would expose the home folder). "
                            "Run from a project folder instead.")
        self.extra_read = [os.path.realpath(os.path.expanduser(p)) for p in (extra_read or [])]
        for p in self.extra_read:
            if Path(p) == home or Path(p) in home.parents:
                raise ToolError(f"code.read_paths entry {p} would expose the whole home folder; list specific folders.")
        self.test_cmd = test_cmd
        self.allow_test_edits = allow_test_edits

    # ------------------------------------------------------------- paths

    def path(self, rel: str) -> Path:
        if not rel or not isinstance(rel, str):
            raise ToolError("path is required, relative to the workspace root, e.g. src/app.py")
        p = (self.root / rel.lstrip("/") if not os.path.isabs(rel) else Path(rel)).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"{rel} is outside the workspace. Use a path inside {self.root.name}/.")
        return p

    def is_test(self, p: Path) -> bool:
        """Test files and test folders: read-only for the agent unless the user allows edits."""
        rel = p.relative_to(self.root)
        if any(part in TEST_DIRS for part in rel.parts[:-1]):
            return True
        return any(fnmatch.fnmatch(p.name, pat) for pat in TEST_FILES)

    def writable(self, rel: str) -> Path:
        p = self.path(rel)
        if self.is_test(p) and not self.allow_test_edits:
            raise ToolError(f"{rel} is a test file. Tests are read-only: fix the code under test instead. "
                            "(The user can allow test edits with --allow-test-edits or code.allow_test_edits.)")
        return p

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    def files(self) -> list[Path]:
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(filenames):
                if not f.startswith(".") or f in {".gitignore", ".env.example"}:
                    out.append(Path(dirpath) / f)
        return out

    # ------------------------------------------------------------- read tools

    def read_file(self, path: str, range: list[int] | None = None) -> str:
        p = self.path(path)
        if not p.is_file():
            hint = self._suggest(path)
            raise ToolError(f"{path} does not exist." + (f" Did you mean {hint}?" if hint else " Use list_dir or glob to find it."))
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"{path} is larger than 2 MB; use grep to find the part you need.")
        lines = p.read_text(errors="replace").splitlines()
        total = len(lines)
        start, end = 1, min(total, MAX_READ_LINES)
        if range:
            if len(range) != 2 or not all(isinstance(x, int) for x in range):
                raise ToolError("range must be two line numbers, e.g. [10, 40].")
            start, end = max(1, range[0]), range[1]
            if start > total:
                raise ToolError(f"{path} has {total} lines; start {start} is past the end.")
            end = min(end, total, start + MAX_READ_LINES - 1)
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in _range(start, end + 1))  # `range` is the argument here
        more = f"\n... lines {end + 1}-{total} not shown; pass range to read more." if end < total else ""
        return clip(f"{path} (lines {start}-{end} of {total})\n{body}{more}")

    def list_dir(self, path: str = ".") -> str:
        p = self.path(path or ".")
        if not p.is_dir():
            raise ToolError(f"{path} is not a folder.")
        items = []
        for c in sorted(p.iterdir()):
            if c.name in SKIP_DIRS or (c.name.startswith(".") and c.name != ".gitignore"):
                continue
            items.append(c.name + ("/" if c.is_dir() else ""))
        shown = items[:300]
        return "\n".join(shown) + (f"\n... {len(items) - 300} more" if len(items) > 300 else "") or "(empty)"

    def glob(self, pattern: str) -> str:
        if not pattern:
            raise ToolError("pattern is required, e.g. **/*.py")
        hits = [self.rel(f) for f in self.files() if fnmatch.fnmatch(self.rel(f), pattern)
                or fnmatch.fnmatch(f.name, pattern)]
        return "\n".join(hits[:200]) + (f"\n... {len(hits) - 200} more" if len(hits) > 200 else "") if hits else "No files match."

    def grep(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        if not pattern:
            raise ToolError("pattern is required: a regular expression such as 'def retry' or a code identifier.")
        try:
            rx = re.compile(pattern)
        except re.error as err:
            raise ToolError(f"Invalid regular expression ({err}). Escape special characters or search a plain word.") from None
        base = self.path(path or ".")
        targets = [base] if base.is_file() else [f for f in self.files() if base == self.root or base in f.parents]
        hits, total = [], 0
        for f in targets:
            if glob and not fnmatch.fnmatch(f.name, glob):
                continue
            try:
                if f.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = f.read_text(errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(hits) < MAX_GREP:
                        hits.append(f"{self.rel(f)}:{i}: {line.strip()[:200]}")
        if not hits:
            return f"No matches for {pattern!r}. Try a shorter pattern or a different spelling."
        more = f"\n... {total - len(hits)} more matches; narrow the pattern or the path." if total > len(hits) else ""
        return "\n".join(hits) + more

    def _suggest(self, path: str) -> str | None:
        name = Path(path).name
        for f in self.files():
            if f.name == name:
                return self.rel(f)
        return None

    # ------------------------------------------------------------- write tools

    def preview_edit(self, path: str, edits: str) -> tuple[str, str, str]:
        """Return (before, after, diff) without writing. Raises ToolError on a bad edit."""
        p = self.writable(path)
        before = p.read_text() if p.is_file() else ""
        if not p.exists() and "SEARCH" not in edits:
            raise ToolError(f"{path} does not exist. Use write_file to create it.")
        try:
            after = E.apply_edits(path, before, E.parse_blocks(edits))
        except E.EditError as err:
            raise ToolError(str(err)) from None
        return before, after, E.unified_diff(path, before, after)

    def preview_lines(self, path: str, start: int, end: int, new: str) -> tuple[str, str, str]:
        p = self.writable(path)
        if not p.is_file():
            raise ToolError(f"{path} does not exist. Use write_file to create it.")
        before = p.read_text()
        try:
            after = E.replace_lines(path, before, start, end, new)
        except E.EditError as err:
            raise ToolError(str(err)) from None
        return before, after, E.unified_diff(path, before, after)

    def apply_text(self, path: str, after: str, diff: str) -> str:
        p = self.writable(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(after)
        return clip(diff or "No change.")

    def edit_file(self, path: str, edits: str) -> str:
        _, after, diff = self.preview_edit(path, edits)
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(after)
        return clip(diff or "No change.")

    def preview_write(self, path: str, content: str) -> str:
        p = self.writable(path)
        before = p.read_text() if p.is_file() else ""
        return E.unified_diff(path, before, content)

    def write_file(self, path: str, content: str) -> str:
        if content is None:
            raise ToolError("content is required.")
        p = self.writable(path)
        problem = E.check_syntax(path, content)
        if problem:
            raise ToolError(f"Not written: {problem}")
        diff = self.preview_write(path, content)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return clip(diff or "No change.")

    # ------------------------------------------------------------- commands

    def _profile(self, tmp: str) -> str:
        home = os.path.realpath(os.path.expanduser("~"))
        reads = [str(self.root), tmp, os.path.realpath(sys.base_prefix), *self.extra_read]
        q = lambda p: str(p).replace("\\", "\\\\").replace('"', '\\"')  # quote for the profile language
        reads = [q(r) for r in reads]
        home, tmp_q, root_q = q(home), q(tmp), q(self.root)
        allow_read = " ".join(f'(subpath "{r}")' for r in reads)
        # Path lookups stat every parent folder, so the parents of allowed paths need metadata access.
        parents = {str(p) for r in reads for p in Path(r).parents if str(p).startswith(home)}
        meta = " ".join(f'(literal "{p}")' for p in sorted(parents))
        return ('(version 1)(allow default)(deny network*)'
                f'(deny file-write*)(allow file-write* (subpath "{root_q}") (subpath "{tmp_q}") (literal "/dev/null"))'
                f'(deny file-read* (subpath "{home}"))(allow file-read* {allow_read})'
                + (f'(allow file-read-metadata {meta})' if meta else ""))

    async def run(self, command: str, timeout: int = 60) -> CmdResult:
        """Run a shell command in the workspace: no network, writes only inside the workspace."""
        if not command or not command.strip():
            raise ToolError("command is required.")
        timeout = max(1, min(int(timeout or 60), 600))
        tmp = os.path.realpath(tempfile.mkdtemp(prefix="proxmino-code-"))
        cmd = ["/bin/sh", "-c", command]
        sandboxed = sys.platform == "darwin" and shutil.which("sandbox-exec") is not None
        if sandboxed:
            cmd = ["sandbox-exec", "-p", self._profile(tmp)] + cmd
        path = [str(self.root / ".venv" / "bin"), str(self.root / "node_modules" / ".bin"),
                "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        env = {"PATH": ":".join(path), "HOME": tmp, "TMPDIR": tmp, "LANG": "en_US.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "CI": "1", "NO_COLOR": "1"}
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(self.root), env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            out, _ = await proc.communicate()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        text = out.decode(errors="replace")
        if len(text) > MAX_CHARS:  # keep the end: test summaries and tracebacks live there
            text = f"... ({len(text) - MAX_CHARS} earlier characters cut)\n" + text[-MAX_CHARS:]
        return CmdResult(command, proc.returncode, text.rstrip(), round(time.monotonic() - start, 2), timed_out, sandboxed)

    def test_command(self) -> str | None:
        """The configured test command, or a guess from the project files."""
        if self.test_cmd:
            return self.test_cmd
        r = self.root
        py = "python" if (r / ".venv" / "bin" / "python").exists() else "python3"
        if (r / "pytest.ini").exists() or (r / "conftest.py").exists() or (r / "tests").is_dir() and any((r / "tests").glob("test_*.py")):
            return f"{py} -m pytest -q -x"
        if (r / "pyproject.toml").exists() and "pytest" in (r / "pyproject.toml").read_text(errors="ignore"):
            return f"{py} -m pytest -q -x"
        if (r / "package.json").exists() and '"test"' in (r / "package.json").read_text(errors="ignore"):
            return "npm test --silent"
        if (r / "go.mod").exists():
            return "go test ./..."
        if (r / "Cargo.toml").exists():
            return "cargo test --quiet"
        return None

    async def run_tests(self, timeout: int = 300) -> CmdResult:
        cmd = self.test_command()
        if not cmd:
            raise ToolError("No test framework found (looked for pytest, npm test, go test, cargo test). Use run with a command.")
        return await self.run(cmd, timeout)

