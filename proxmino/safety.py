"""Output style filter and citation verification."""
from __future__ import annotations

import re

# Pictographic emoji, symbols, flags, keycaps, plus joiners and variation selectors.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF"
    "\U0000FE0F\U0000200D\U000020E3\U0001F3FB-\U0001F3FF]+")


def strip_emoji(text: str) -> str:
    return _EMOJI.sub("", text)


# A citation is [n] not glued to an identifier, so code like arr[1] is left alone.
_CITE = re.compile(r"(?<![\w)])\[(\d{1,2})\]")
_CODE = re.compile(r"```.*?(?:```|$)|`[^`\n]*`", re.S)


def verify_citations(answer: str, valid_ids: set[int]) -> tuple[str, list[int], list[int]]:
    """Remove citation markers that point at sources that do not exist.

    Returns (cleaned_answer, used_ids, invalid_ids).
    """
    used, invalid = [], []

    def fix(m: re.Match) -> str:
        n = int(m.group(1))
        if n in valid_ids:
            if n not in used:
                used.append(n)
            return m.group(0)
        invalid.append(n)
        return ""

    parts, last = [], 0
    for m in _CODE.finditer(answer):  # never touch code
        parts.append(_CITE.sub(fix, answer[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_CITE.sub(fix, answer[last:]))
    return "".join(parts), used, invalid


TRUSTED = ("wikipedia.org", ".gov", ".edu", "who.int", "nature.com", "nih.gov", "europa.eu")


def confidence(used: list[int], domains: dict[int, str]) -> str | None:
    """high: two independent sources, or a trusted reference such as Wikipedia; medium: one ordinary site.
    No badge when nothing was cited (a missing badge reads better than a red "low" on every chat answer)."""
    distinct = {domains[n] for n in used if n in domains}
    if len(distinct) >= 2 or any(d.endswith(t) for d in distinct for t in TRUSTED):
        return "high"
    return "medium" if distinct else None
