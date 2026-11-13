"""Run Python snippets with a timeout, resource limits and (on macOS) no network or outside writes."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass

MAX_OUTPUT = 20_000


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int | None
    seconds: float
    timed_out: bool
    sandboxed: bool


def _limits(seconds: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50_000_000, 50_000_000))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass
        os.setsid()
    return apply


def _profile(workdir: str) -> str:
    wd = os.path.realpath(workdir)
    return (f'(version 1)(allow default)(deny network*)'
            f'(deny file-write*)(allow file-write* (subpath "{wd}") (literal "/dev/null"))'
            f'(deny file-read* (subpath "{os.path.expanduser("~")}"))'
            f'(allow file-read* (subpath "{wd}") (subpath "{os.path.realpath(sys.base_prefix)}"))')


async def run_python(code: str, seconds: int = 10) -> RunResult:
    workdir = tempfile.mkdtemp(prefix="proxmino-run-")
    path = os.path.join(workdir, "main.py")
    with open(path, "w") as f:
        f.write(code)
    # The base interpreter, not the venv one: snippets need no app packages, and the venv lives
    # under the home folder, which the sandbox hides.
    cmd = [os.path.realpath(getattr(sys, "_base_executable", sys.executable)), "-I", "-B", path]
    sandboxed = sys.platform == "darwin" and shutil.which("sandbox-exec") is not None
    if sandboxed:
        cmd = ["sandbox-exec", "-p", _profile(workdir)] + cmd
    env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=workdir, env=env, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, preexec_fn=_limits(seconds))
    timed_out = False
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=seconds + 2)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        out, err = await proc.communicate()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    def clip(b: bytes) -> str:
        s = b.decode(errors="replace")
        return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + f"\n... ({len(s) - MAX_OUTPUT} more characters)"

    return RunResult(clip(out), clip(err), proc.returncode, round(time.monotonic() - start, 2), timed_out, sandboxed)
