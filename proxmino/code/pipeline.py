"""Fixed localize, patch, test pipeline for models that are weak at free-form tool use.

The harness decides every step; the model only fills one narrow slot at a time:
  1. run the tests to see what fails
  2. localize: files from the traceback and the failing tests, plus the model's pick from the repo map
  3. patch: several candidates, each one SEARCH/REPLACE block for one shown file
  4. validate: apply each candidate alone, run the tests, keep the best, revert the rest
Yields the same events as the tool loop (model, call, result, final), so the CLI and UI show both.
"""
from __future__ import annotations

import ast
import re
from collections.abc import AsyncIterator
from pathlib import Path

from .. import prompts as P
from ..llm import Params, Router, parse_json, strip_think
from . import repomap
from .agent import Approver, CodeRequest, normalize_edits, parse_call
from .edits import Edit, EditError, apply_edits, parse_blocks, unified_diff
from .permissions import ALLOW, ASK, Policy
from .workspace import ToolError, Workspace

LOCALIZE = """Which source files most likely need a change to reach the goal? Test files are read-only, so pick the code under test.
Return JSON only: {{"files": ["path", ...]}} with 1 to 3 paths from the repo map.

Goal: {goal}

Failing test output:
{tests}

Repo map:
{map}"""

PATCH = """Fix the code so the goal is met and the failing tests pass. Change only what is needed. Do not change tests.

Goal: {goal}

Failing test output:
{tests}

{files}

Reply with one or more SEARCH/REPLACE blocks and nothing else. SEARCH must copy whole lines exactly from the file above, without the line numbers:
<<<<<<< SEARCH
exact current lines
=======
new lines
>>>>>>> REPLACE"""

REWRITE = """Fix the code so the goal is met and the failing tests pass. Do not change tests.

Goal: {goal}

Failing test output:
{tests}

{files}

Find the one function that has the bug and rewrite it. Reply with only the corrected function, complete, in a ```python block. Keep its name and arguments."""

FAIL_COUNT = re.compile(r"(?:failures|errors)=(\d+)|(\d+) (?:failed|errors?)\b", re.I)
TRACE_FILE = re.compile(r'File "([^"]+)", line (\d+)|^([\w./-]+\.\w+):(\d+)', re.M)
PER_FILE_LINES = 220
# Small models bend the markers: fewer angle brackets, code fences around the halves, "REPLACE" on its own.
LOOSE_BLOCK = re.compile(r"(?:^|\n)<*\s*SEARCH\s*\n(?:```\w*\n)?(.*?)\n?(?:```\s*\n)?={3,}\s*\n(?:```\w*\n)?(.*?)\n?(?:```\s*\n)?>+\s*REPLACE", re.S)


def failures(output: str, ok: bool) -> int:
    """Number of failing tests in a test run (0 if it passed; large if it could not run)."""
    if ok:
        return 0
    n = sum(int(a or b) for a, b in FAIL_COUNT.findall(output))
    return n if n else 1000


class Pipeline:
    def __init__(self, router: Router, ws: Workspace, policy: Policy, product: str = "Proxmino",
                 tier: str = "code", candidates: int = 3, approve: Approver | None = None):
        self.router, self.ws, self.policy, self.product = router, ws, policy, product
        self.tier, self.n, self.approve = tier, candidates, approve

    async def _ask(self, prompt: str, temperature: float, max_tokens: int = 1200) -> tuple[str, str]:
        msgs = [{"role": "system", "content": P.PREFIX.format(name=self.product)}, {"role": "user", "content": prompt}]
        model, it = await self.router.open(self.tier, msgs, Params(temperature=temperature, max_tokens=max_tokens))
        return model, strip_think("".join([c async for c in it]))

    async def _tests(self) -> tuple[bool, str, int]:
        r = await self.ws.run_tests()
        ok = r.exit_code == 0 and not r.timed_out
        return ok, r.text(), failures(r.output, ok)

    # ------------------------------------------------------------- localize

    def _source_files(self) -> list[str]:
        return [self.ws.rel(f) for f in self.ws.files() if f.suffix in repomap.CODE_EXT and not self.ws.is_test(f)]

    def _from_trace(self, output: str) -> list[str]:
        """Non-test project files named in tracebacks, most recent frame last."""
        out = []
        for m in TRACE_FILE.finditer(output):
            name = m.group(1) or m.group(3)
            p = Path(name) if Path(name).is_absolute() else self.ws.root / name
            try:
                p = p.resolve()
                if self.ws.root in p.parents and p.is_file() and not self.ws.is_test(p):
                    rel = self.ws.rel(p)
                    if rel not in out:
                        out.append(rel)
            except (OSError, ValueError):
                continue
        return out

    def _from_tests(self, output: str) -> tuple[list[str], set[str]]:
        """Files defining names that the failing tests call."""
        names: set[str] = set()
        for f in self.ws.files():
            if not self.ws.is_test(f) or f.suffix != ".py":
                continue
            text = f.read_text(errors="replace")
            for tm in re.finditer(r"def (test_\w+)", text):
                if tm.group(1) in output:
                    body = text[tm.end():].split("\n    def ", 1)[0].split("\ndef ", 1)[0]
                    names |= set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", body)) - {"assertEqual", "assertTrue", "assertFalse",
                                                                              "assertRaises", "assertIn", "len", "print", "self"}
        files = []
        for rel in self._source_files():
            text = (self.ws.root / rel).read_text(errors="replace")
            if any(re.search(rf"^\s*(def|class|function|func|fn)\s+{re.escape(n)}\b", text, re.M) for n in names):
                files.append(rel)
        return files, names

    def _imports(self, rel: str, sources: set[str]) -> list[str]:
        """Project source files that a Python file imports (one hop)."""
        if not rel.endswith(".py"):
            return []
        text = (self.ws.root / rel).read_text(errors="replace")
        pkg = Path(rel).parent
        out = []
        for m in re.finditer(r"^\s*from\s+(\.*)([\w.]*)\s+import\s+([\w, ]+)|^\s*import\s+([\w.]+)", text, re.M):
            dots, mod, names, plain = m.groups()
            if plain:
                bases = [plain.replace(".", "/")]
            else:
                base = pkg
                for _ in range(max(0, len(dots) - 1)):
                    base = base.parent
                root = (base / mod.replace(".", "/")) if dots else Path(mod.replace(".", "/"))
                bases = [str(root)] + [str(root / n.strip()) for n in names.split(",") if n.strip()]
            for b in bases:
                for cand in (f"{b}.py", f"{b}/__init__.py"):
                    cand = cand.lstrip("./")
                    if cand in sources and cand != rel and cand not in out and not cand.endswith("__init__.py"):
                        out.append(cand)
        return out

    async def localize(self, goal: str, test_out: str) -> tuple[list[str], set[str], str]:
        trace = self._from_trace(test_out)
        by_name, names = self._from_tests(test_out)
        sources = set(self._source_files())
        picked: list[str] = []
        model = ""
        if not (trace or by_name) or len(sources) > 3:
            model, reply = await self._ask(LOCALIZE.format(goal=goal, tests=test_out[-3000:], map=repomap.build(self.ws, 4000)), 0.1, 200)
            got = parse_json(reply, {})
            picked = [f for f in (got.get("files", []) if isinstance(got, dict) else []) if isinstance(f, str) and f in sources]
        files = list(dict.fromkeys(trace[-2:] + by_name + picked))
        for f in list(files):  # the bug may sit in a helper the tested code imports
            files += [i for i in self._imports(f, sources) if i not in files]
        files = files[:4] or sorted(sources)[:3]
        return files, names, model

    def _show(self, rel: str, names: set[str]) -> str:
        """The file with line numbers; for long files only the functions the tests use, with context."""
        lines = (self.ws.root / rel).read_text(errors="replace").splitlines()
        spans = [(1, len(lines))]
        if len(lines) > PER_FILE_LINES and rel.endswith(".py") and names:
            try:
                tree = ast.parse("\n".join(lines))
                spans = [(n.lineno, n.end_lineno) for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names] or spans
            except SyntaxError:
                pass
        parts = []
        for a, b in spans:
            a, b = max(1, a - 3), min(len(lines), b + 3, a + PER_FILE_LINES)
            parts.append("\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(a, b + 1)))
        return f"File {rel} (line numbers are not part of the file):\n" + "\n...\n".join(parts)

    # ------------------------------------------------------------- patch and validate

    def _rewrite(self, reply: str, files: list[str]) -> dict[str, str] | None:
        """Replace whole top-level functions named in a ```python reply, keeping the file's indentation."""
        m = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
        code = (m.group(1) if m else reply).strip("\n")
        try:
            new_tree = ast.parse(code)
        except SyntaxError:
            return None
        new_funcs = {n.name: n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        if not new_funcs:
            return None
        code_lines = code.split("\n")
        out = {}
        for f in files:
            if not f.endswith(".py"):
                continue
            text = out.get(f, (self.ws.root / f).read_text())
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            lines = text.split("\n")
            for node in sorted((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                and n.name in new_funcs), key=lambda n: -n.lineno):
                new = new_funcs.pop(node.name)
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
                body = code_lines[min([new.lineno] + [d.lineno for d in new.decorator_list]) - 1:new.end_lineno]
                body = [indent + l if l.strip() else l for l in body]
                lines[start:node.end_lineno] = body
                text = "\n".join(lines)
            if text != (self.ws.root / f).read_text():
                out[f] = text
        return out or None

    def _parse(self, reply: str, files: list[str]) -> dict[str, str] | None:
        """Either reply format is accepted, whichever slot was asked for."""
        if "SEARCH" in reply or '"tool"' in reply:
            return self._candidate(reply, files) or self._rewrite(reply, files)
        return self._rewrite(reply, files) or self._candidate(reply, files)

    def _candidate(self, reply: str, files: list[str]) -> dict[str, str] | None:
        """Turn a reply into {path: new_text}. Blocks apply to the first shown file they match."""
        try:
            blocks = parse_blocks(reply) if "SEARCH" in reply else []
        except EditError:
            blocks = [Edit(m.group(1), m.group(2)) for m in LOOSE_BLOCK.finditer(reply)]
        if not blocks:
            call = parse_call(reply)  # some models answer with an edit_file call anyway
            if call and call["tool"] == "edit_file":
                try:
                    blocks = parse_blocks(normalize_edits(call["args"]))
                except (ToolError, EditError):
                    return None
        if not blocks:
            return None
        texts = {f: (self.ws.root / f).read_text() for f in files}
        out = dict(texts)
        for b in blocks:
            for f in files:
                try:
                    out[f] = apply_edits(f, out[f], [b])
                    break
                except EditError:
                    continue
            else:
                return None
        changed = {f: t for f, t in out.items() if t != texts[f]}
        return changed or None

    async def run(self, req: CodeRequest) -> AsyncIterator[tuple[str, dict]]:
        steps = 0

        def final(text, changed=(), passed=None):
            return "final", {"text": text, "changed": sorted(changed), "tests_passed": passed, "steps": steps}

        if not self.ws.test_command():
            yield final("No test command found, so the fixed pipeline cannot check a patch. Use --strategy loop.")
            return
        if self.policy.mode == "plan" or self.policy.check("run_tests", {}).action not in (ALLOW, ASK):
            yield final("The fixed pipeline needs to run tests and edit files; use --plan with the loop strategy instead.")
            return
        if self.policy.check("run_tests", {}).action == ASK and not (
                self.approve and await self.approve("run_tests", {}, f"$ {self.ws.test_command()}", "run the tests to find the failure")):
            yield final("The tests were not approved, so nothing was changed.")
            return

        ok, out, base = await self._tests()
        yield "result", {"tool": "run_tests", "ok": ok, "output": out}
        if ok:
            yield final("All tests already pass. The fixed pipeline needs a failing test that shows the goal; "
                        "add one, or use --strategy loop.", passed=True)
            return

        files, names, model = await self.localize(req.goal, out)
        steps += 1
        if model:
            yield "model", {"model": model, "step": steps}
        yield "call", {"tool": "localize", "args": {"files": files}}

        shown = "\n\n".join(self._show(f, names) for f in files)
        py = all(f.endswith(".py") for f in files)
        # Python: whole-function rewrites are the easiest slot for small models; edit blocks add variety.
        plan = ([("rewrite", 0.2), ("block", 0.4), ("rewrite", 0.8), ("block", 0.8), ("rewrite", 1.0)] if py else
                [("block", 0.2), ("block", 0.6), ("block", 0.9), ("block", 0.4), ("block", 0.8)])[:self.n]
        cands: list[dict[str, str]] = []
        for kind, t in plan:
            steps += 1
            tmpl = REWRITE if kind == "rewrite" else PATCH
            model, reply = await self._ask(tmpl.format(goal=req.goal, tests=out[-3000:], files=shown), t)
            yield "model", {"model": model, "step": steps}
            c = self._parse(reply, files)
            if c and c not in cands:
                cands.append(c)
            yield "call", {"tool": "patch", "args": {"kind": kind, "temperature": t, "usable": bool(c)}}
        if not cands:
            yield final("None of the candidate patches matched the code, so nothing was changed.", passed=False)
            return

        originals = {f: (self.ws.root / f).read_text() for c in cands for f in c}
        best, best_fail, best_out = None, base, out
        try:
            for i, c in enumerate(cands, 1):
                for f, text in c.items():
                    self.ws.writable(f).write_text(text)
                ok_i, out_i, fail_i = await self._tests()
                yield "result", {"tool": "run_tests", "ok": ok_i, "output": f"Candidate {i} of {len(cands)}: {out_i}"}
                for f in c:
                    (self.ws.root / f).write_text(originals[f])
                if fail_i < best_fail:
                    best, best_fail, best_out = c, fail_i, out_i
                if ok_i:
                    break
        finally:
            for f, text in originals.items():  # never leave a candidate behind
                (self.ws.root / f).write_text(text)

        if best is None:
            yield final(f"Tried {len(cands)} candidate patch(es); none reduced the failing tests ({base}). Nothing was changed.", passed=False)
            return
        diff = "".join(unified_diff(f, originals[f], t) for f, t in best.items())
        decision = self.policy.check("edit_file", {})
        if decision.action == ASK and not (self.approve and await self.approve("edit_file", {"path": ", ".join(best)}, diff, "apply the best patch")):
            yield final("The best patch was not approved, so nothing was changed.\n\n" + diff, passed=False)
            return
        for f, text in best.items():
            self.ws.writable(f).write_text(text)
        yield "result", {"tool": "edit_file", "ok": True, "output": diff}
        passed = best_fail == 0
        summary = (f"{'Fixed' if passed else 'Partly fixed'}: changed {', '.join(best)}. "
                   f"Failing tests went from {base} to {best_fail} after trying {len(cands)} candidate patch(es).")
        yield final(f"{summary}\n\n{diff}", best, passed)


def choose_strategy(cfg, router: Router, override: str | None = None) -> str:
    """'loop' for models listed as strong in code.loop_models, else 'pipeline'."""
    if override in ("loop", "pipeline"):
        return override
    s = cfg.get("code.strategy", "auto")
    if s in ("loop", "pipeline"):
        return s
    strong = set(cfg.get("code.loop_models", []) or [])
    first = next(iter(router.chain("code")), None)
    return "loop" if first and f"{first[0].name}:{first[1]}" in strong else "pipeline"
