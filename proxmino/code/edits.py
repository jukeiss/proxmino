"""Search/replace edit engine.

Blocks look like:

    <<<<<<< SEARCH
    old exact lines
    =======
    new lines
    >>>>>>> REPLACE

Matching is exact first, then whitespace-tolerant (trailing spaces and indentation). A block that
matches nowhere, or in more than one place, fails with the closest region so the model can retry.
All blocks for a file apply together or not at all, and Python and JSON results are syntax-checked.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass

BLOCK = re.compile(r"<{5,9} ?SEARCH\s*?\n(.*?)\n?={5,9}\s*?\n(.*?)\n?>{5,9} ?REPLACE", re.S)


class EditError(Exception):
    pass


@dataclass
class Edit:
    search: str
    replace: str


def parse_blocks(text: str) -> list[Edit]:
    edits = [Edit(m.group(1), m.group(2)) for m in BLOCK.finditer(text)]
    if not edits and "SEARCH" in text:
        raise EditError("Could not parse the edit. Use <<<<<<< SEARCH, the old lines, =======, the new lines, "
                        "then >>>>>>> REPLACE, each marker on its own line.")
    return edits


_range = range


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _norm(line: str) -> str:
    return line.strip()


def _find(lines: list[str], needle: list[str]) -> list[int]:
    n = len(needle)
    return [i for i in range(len(lines) - n + 1) if lines[i:i + n] == needle]


def _reindent(new: list[str], old_file: list[str], old_search: list[str]) -> list[str]:
    """Shift the replacement by the indent difference between the file and the SEARCH text,
    taken from the first line pair whose indentation differs."""
    def indent(s: str) -> str:
        return s[:len(s) - len(s.lstrip())]
    fi = si = ""
    for f, q in zip(old_file, old_search):
        if f.strip() and indent(f) != indent(q):
            fi, si = indent(f), indent(q)
            break
    if fi == si:
        return new
    if fi and si and set(fi + si) == {" "} and len(fi) % len(si) == 0:
        # Same nesting at a different indent width (2 vs 4 spaces): scale every line.
        k = len(fi) // len(si)
        return [" " * ((len(l) - len(l.lstrip(" "))) * k) + l.lstrip(" ") if l.strip() else l for l in new]
    out = []
    for l in new:
        if not l.strip():
            out.append(l)
        elif si and l.startswith(si):
            out.append(fi + l[len(si):])
        elif not si:
            out.append(fi + l)
        else:
            out.append(l)
    return out


def _closest(lines: list[str], needle: list[str]) -> str:
    if not lines or not needle:
        return ""
    n = len(needle)
    target = "\n".join(_norm(l) for l in needle)
    best, best_i = 0.0, 0
    for i in range(max(1, len(lines) - n + 1)):
        r = difflib.SequenceMatcher(None, target, "\n".join(_norm(l) for l in lines[i:i + n])).quick_ratio()
        if r > best:
            best, best_i = r, i
    lo, hi = max(0, best_i - 2), min(len(lines), best_i + n + 2)
    shown = "\n".join(f"{i + 1:>5}  {lines[i]}" for i in _range(lo, hi))
    return (f"The file around the closest match (line numbers are not part of the file):\n{shown}\n"
            'Copy whole lines exactly, or edit by line number instead: {"path": ..., "lines": [start, end], "replace": "new lines"}.')


def apply_one(text: str, e: Edit) -> str:
    if e.search == "":
        if text.strip():
            raise EditError("An empty SEARCH section only works for a new or empty file.")
        return e.replace + ("\n" if e.replace and not e.replace.endswith("\n") else "")
    trailing_nl = text.endswith("\n")
    lines = text.split("\n")
    if trailing_nl:
        lines = lines[:-1]
    s_lines, r_lines = e.search.split("\n"), e.replace.split("\n") if e.replace else []

    hits = _find(lines, s_lines)
    new_lines = r_lines
    if not hits:
        norm = [_norm(l) for l in lines]
        hits = _find(norm, [_norm(l) for l in s_lines])
        if hits and len(hits) == 1:
            i = hits[0]
            new_lines = _reindent(r_lines, lines[i:i + len(s_lines)], s_lines)
    if not hits:
        # Last resort: the model joined or split lines. Compare with all whitespace collapsed.
        target = _collapse(e.search)
        spans = [(i, k) for k in _range(1, min(12, len(lines)) + 1) for i in _range(len(lines) - k + 1)
                 if lines[i].strip() and lines[i + k - 1].strip() and _collapse(" ".join(lines[i:i + k])) == target]
        if len(spans) == 1:
            i, k = spans[0]
            return "\n".join(lines[:i] + r_lines + lines[i + k:]) + ("\n" if trailing_nl else "")
    if not hits:
        raise EditError("SEARCH text not found. Copy the lines exactly from the file.\n" + _closest(lines, s_lines))
    if len(hits) > 1:
        where = ", ".join(str(h + 1) for h in hits[:5])
        raise EditError(f"SEARCH text matches {len(hits)} places (lines {where}). Add surrounding lines so it is unique.")
    i = hits[0]
    out = lines[:i] + new_lines + lines[i + len(s_lines):]
    return "\n".join(out) + ("\n" if trailing_nl else "")


def replace_lines(path: str, text: str, start: int, end: int, new: str) -> str:
    """Replace lines start..end (1-based, inclusive) with new text, keeping the file's indentation."""
    trailing_nl = text.endswith("\n")
    lines = text.split("\n")
    if trailing_nl:
        lines = lines[:-1]
    total = len(lines)
    if not (1 <= start <= end <= total):
        raise EditError(f"lines must be [start, end] with 1 <= start <= end <= {total} (the file has {total} lines).")
    old = lines[start - 1:end]
    new_lines = new.rstrip("\n").split("\n") if new else []
    new_lines = _reindent(new_lines, old, new_lines[:len(old)])
    out = "\n".join(lines[:start - 1] + new_lines + lines[end:]) + ("\n" if trailing_nl else "")
    problem = check_syntax(path, out)
    if problem and not check_syntax(path, text):
        raise EditError(f"The edit would break the file, so it was not applied. {problem}. "
                        f"Lines {start}-{end} were:\n" + "\n".join(old))
    return out


def check_syntax(path: str, text: str) -> str | None:
    """Return an error message, or None if the file parses (only checked for known types)."""
    if path.endswith(".py"):
        try:
            compile(text, path, "exec", dont_inherit=True)
        except SyntaxError as err:
            return f"Python syntax error at line {err.lineno}: {err.msg}"
    elif path.endswith(".json"):
        try:
            json.loads(text)
        except ValueError as err:
            return f"JSON error: {err}"
    return None


def apply_edits(path: str, text: str, edits: list[Edit]) -> str:
    """Apply all edits to text, atomically. Raises EditError naming the failing block."""
    if not edits:
        raise EditError("No SEARCH/REPLACE blocks found.")
    out = text
    for n, e in enumerate(edits, 1):
        try:
            out = apply_one(out, e)
        except EditError as err:
            raise EditError(f"Block {n} of {len(edits)} failed; no changes were made. {err}") from None
    problem = check_syntax(path, out)
    if problem and not check_syntax(path, text):
        raise EditError(f"The edit would break the file, so it was not applied. {problem}")
    return out


def unified_diff(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile=f"a/{path}", tofile=f"b/{path}"))
