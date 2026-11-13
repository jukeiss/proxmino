"""Repo map: a compact outline of the codebase, ranked by importance and trimmed to a budget.

Python is parsed with `ast`; JavaScript/TypeScript, Go, Rust, Swift, Java and Kotlin use light
regexes for top-level definitions. Files are ranked by how often other files import them, then by
git recency (or modification time).
"""
from __future__ import annotations

import ast
import re
import subprocess
from collections import Counter
from pathlib import Path

from .workspace import Workspace

CODE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".go", ".rs", ".swift", ".java", ".kt", ".rb", ".php", ".c", ".h", ".cpp", ".cs"}
DOC_FILES = {"README.md", "pyproject.toml", "package.json", "go.mod", "Cargo.toml", "requirements.txt", "Makefile"}

_DEF = {
    "js": re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\s*\*?\s*(\w+)\s*\([^)]*\)|class\s+(\w+)|"
                     r"(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>|interface\s+(\w+)|type\s+(\w+)\s*=)", re.M),
    "go": re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+\s*\([^)]*\))|^type\s+(\w+\s+(?:struct|interface))", re.M),
    "rs": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn\s+(\w+\s*(?:<[^>]*>)?\s*\([^)]*\))|struct\s+(\w+)|enum\s+(\w+)|trait\s+(\w+)|impl(?:<[^>]*>)?\s+([\w:<>, ]+))", re.M),
    "generic": re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|final\s+|open\s+)*"
                          r"(?:class|struct|enum|protocol|interface|func|fun|def|function)\s+(\w+)", re.M),
}
_LANG = {".js": "js", ".jsx": "js", ".ts": "js", ".tsx": "js", ".mjs": "js", ".go": "go", ".rs": "rs"}
_IMPORT = re.compile(r"^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w, ]+)|.*?require\(['\"]([^'\"]+)['\"]\)|"
                     r".*?from\s+['\"]([^'\"]+)['\"])", re.M)


def _py_outline(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    def sig(f) -> str:
        a = f.args
        names = [x.arg for x in a.posonlyargs + a.args]
        if a.vararg:
            names.append("*" + a.vararg.arg)
        names += [x.arg for x in a.kwonlyargs]
        if a.kwarg:
            names.append("**" + a.kwarg.arg)
        pre = "async def " if isinstance(f, ast.AsyncFunctionDef) else "def "
        return f"{pre}{f.name}({', '.join(names)})"

    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("__"):
            out.append(sig(node))
        elif isinstance(node, ast.ClassDef):
            out.append(f"class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not sub.name.startswith("_") or sub.name == "__init__"):
                    out.append("    " + sig(sub))
    return out


def outline(path: Path, text: str) -> list[str]:
    if path.suffix == ".py":
        return _py_outline(text)
    rx = _DEF.get(_LANG.get(path.suffix, ""), _DEF["generic"])
    out = []
    for m in rx.finditer(text):
        name = next((g for g in m.groups() if g), None)
        if name:
            out.append(re.sub(r"\s+", " ", name.strip()))
    return list(dict.fromkeys(out))


def _git_recency(root: Path) -> dict[str, int]:
    """Rank of each file by its most recent commit (lower is more recent)."""
    try:
        out = subprocess.run(["git", "-C", str(root), "log", "--name-only", "--pretty=format:", "-n", "200"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    seen: dict[str, int] = {}
    for line in out.splitlines():
        if line and line not in seen:
            seen[line] = len(seen)
    return seen


def build(ws: Workspace, budget_chars: int = 6000) -> str:
    files = [f for f in ws.files() if f.suffix in CODE_EXT or f.name in DOC_FILES]
    texts: dict[Path, str] = {}
    for f in files:
        try:
            if f.stat().st_size <= 400_000:
                texts[f] = f.read_text(errors="replace")
        except OSError:
            pass

    # Import in-degree: how many files mention each module's stem in an import line.
    stems = Counter()
    for f, t in texts.items():
        for m in _IMPORT.finditer(t):
            for g in m.groups():
                if g:
                    for part in re.split(r"[,\s]+", g):
                        tail = part.strip().strip(".").split(".")[-1].split("/")[-1]
                        if tail:
                            stems[tail] += 1
    recency = _git_recency(ws.root)

    def score(f: Path):
        rel = ws.rel(f)
        return (-(stems.get(f.stem, 0) + (3 if f.name in DOC_FILES or f.stem in {"main", "__main__", "app", "server", "index"} else 0)),
                recency.get(rel, 10_000), -f.stat().st_mtime if f.exists() else 0, rel)

    ranked = sorted(texts, key=score)
    lines, used, omitted = [], 0, 0
    for f in ranked:
        rel = ws.rel(f)
        defs = outline(f, texts[f]) if f.suffix in CODE_EXT else []
        n = texts[f].count("\n") + 1
        block = [f"{rel} ({n} lines)"] + [f"  {d}" for d in defs[:25]]
        if len(defs) > 25:
            block.append(f"  ... {len(defs) - 25} more definitions")
        size = sum(len(x) + 1 for x in block)
        if used + size > budget_chars:
            if used + len(block[0]) + 1 <= budget_chars:
                lines.append(block[0])
                used += len(block[0]) + 1
            else:
                omitted += 1
            continue
        lines.extend(block)
        used += size
    if omitted:
        lines.append(f"... {omitted} more files not shown; use glob or grep.")
    return "\n".join(lines) if lines else "No source files found."
