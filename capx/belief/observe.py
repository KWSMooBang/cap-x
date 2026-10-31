"""VDM Observe for the estimated Belief Graph v2 (``bg_est2``).

The VDM (``--visual-differencing-model``) looks at marked camera images (``som.py``) and answers a
list of statements with ``1`` (true), ``0`` (false) or ``?`` (cannot tell) in one ``Observe:`` line,
plus an optional ``Notes:`` line (prompts M2 / M5 in ``prompts_v2.py``).

Statements are edge keys written as ``(pred args)``: ``(on bowl_A plate_B)``, ``(in cream_cheese_A
basket_A)``, ``(on bowl_A table)``, ``(at basket_A center)``, ``(inhand bowl_B)``, ``(handempty)``,
``(above gripper plate_B)``, ``(lifted bowl_A)``.

The parser is lenient: order, spacing and case do not matter; ``true/false/yes/no`` are read as
1/0; statements that were not asked (or name unknown objects) are ignored and listed; a reply in
which no asked statement can be read is a failure (the estimator then falls back to geometry, the
v1 way).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

EdgeKey = tuple[str, str, "str | None"]
GRIPPER = "gripper"
VALUES = ("1", "0", "?")
_VALUE_WORDS = {"1": "1", "0": "0", "?": "?", "true": "1", "false": "0", "yes": "1", "no": "0",
                "unknown": "?", "unclear": "?", "n/a": "?"}

# Notes keywords -> flag kind. A flagged object's relations become uncertain (S4 quotes the note).
NOTE_FLAGS = (
    ("wrong mark", re.compile(r"wrong (object|mark|place)|mislabel|mark(ed)? .{0,30}(wrong|misplaced|off)|"
                              r"not on (the|its) (object|bowl|plate)", re.I)),
    ("fallen", re.compile(r"\bfall(en)?\b|\bfell\b|knocked|upside[ -]down|on its side|toppled|tipped over", re.I)),
    ("tilted", re.compile(r"tilt|tipped|leaning|propped|on the rim|half on|partly on|partially on", re.I)),
)


def statement_text(key: EdgeKey) -> str:
    pred, subj, target = key
    if pred == "handempty":
        return "(handempty)"
    if target is None:
        return f"({pred} {subj})"
    return f"({pred} {subj} {target})"


def statement_key(pred: str, args: Sequence[str]) -> EdgeKey | None:
    """``("on", ["bowl_A", "plate_B"])`` -> edge key, None for a malformed statement."""
    pred = pred.lower()
    if pred == "handempty":
        return ("handempty", GRIPPER, None) if not args or list(args) == [GRIPPER] else None
    if pred in ("inhand", "lifted"):
        return (pred, args[0], None) if len(args) == 1 else None
    if pred == "above":
        if len(args) == 2 and args[0] == GRIPPER:
            return ("above", GRIPPER, args[1])
        return ("above", GRIPPER, args[0]) if len(args) == 1 else None
    if pred in ("on", "in", "at"):
        return (pred, args[0], args[1]) if len(args) == 2 else None
    return None


_STATEMENT = re.compile(r"\(\s*([A-Za-z_]+)((?:[\s,]+[\w\-]+)*)\s*\)\s*[:=]?\s*(true|false|yes|no|unknown|unclear|n/a|[01?])",
                        re.I)


@dataclass
class ObserveParse:
    values: dict[EdgeKey, str] = field(default_factory=dict)  # asked key -> "1" | "0" | "?"
    notes: str | None = None
    unknown: list[str] = field(default_factory=list)  # statements read but not asked / not understood
    ok: bool = False
    error: str | None = None

    def line(self, order: Sequence[EdgeKey] | None = None) -> str:
        keys = [k for k in (order or self.values) if k in self.values]
        return " | ".join(f"{statement_text(k)} {self.values[k]}" for k in keys)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "values": {statement_text(k): v for k, v in self.values.items()},
                "notes": self.notes, "unknown": self.unknown}


def parse_observe(text: str, asked: Sequence[EdgeKey], names: Sequence[str] = ()) -> ObserveParse:
    """Read the ``Observe:`` line (or, failing that, any ``(statement) value`` in the reply) and ``Notes:``.

    Names are matched without case (``BOWL_a`` -> ``bowl_A``); only ``asked`` keys are kept."""
    out = ObserveParse()
    text = text or ""
    canon = {n.lower(): n for n in list(names) + [GRIPPER, "table"]}
    for k in asked:
        for n in k[1:]:
            if n is not None:
                canon.setdefault(n.lower(), n)
    asked_set = set(asked)

    lines = [ln.strip().strip("*`> ").strip() for ln in text.splitlines()]
    obs_lines = [ln for ln in lines if re.match(r"(?i)\**observe\**\s*:", ln)]
    body = " | ".join(re.sub(r"(?i)^\**observe\**\s*:", "", ln) for ln in obs_lines) if obs_lines else text
    for ln in lines:
        m = re.match(r"(?i)\**notes?\**\s*:\s*(.*)", ln)
        if m and m.group(1).strip():
            out.notes = m.group(1).strip()
            break
    if out.notes:
        # A Notes line inside the scanned body must not be read as statements.
        body = body.replace(out.notes, " ")
    for m in _STATEMENT.finditer(body):
        pred = m.group(1)
        args = [canon.get(a.lower(), a) for a in re.split(r"[\s,]+", m.group(2).strip()) if a]
        val = _VALUE_WORDS.get(m.group(3).lower())
        key = statement_key(pred, args)
        raw = f"({pred} {' '.join(args)})".replace(" )", ")")
        if key is None or key not in asked_set or val is None:
            out.unknown.append(raw)
            continue
        out.values.setdefault(key, val)  # the first answer for a statement wins
    out.ok = bool(out.values)
    if not out.ok:
        out.error = "no asked statement could be read"
    return out


def note_flags(notes: str | None, names: Sequence[str], marks: dict[str, str]) -> dict[str, str]:
    """alias -> flag kind (tilted / fallen / wrong mark) for the objects a Notes line talks about.

    The note is split into sentences; a sentence with a flag keyword flags every alias (or mark,
    ``mark B`` / ``B`` as a lone capital or number) it names."""
    out: dict[str, str] = {}
    if not notes:
        return out
    by_lower = {n.lower(): n for n in names}
    mark_to_alias = {m.lower(): a for a, m in marks.items()}
    for sentence in re.split(r"(?<=[.;!])\s+|\s+(?:but|and)\s+", notes):
        kind = next((k for k, rx in NOTE_FLAGS if rx.search(sentence)), None)
        if kind is None:
            continue
        hit: list[str] = []
        for tok in re.findall(r"[A-Za-z0-9_]+", sentence):
            alias = by_lower.get(tok.lower())
            if alias is not None and alias not in hit:
                hit.append(alias)
        # Marks only as "mark B" / "marks 2 and 3" (a lone "A" is usually an article).
        for m in re.finditer(r"(?i)\bmarks?\s+((?:[A-Z]|\d+)(?:\s*(?:,|and|&)\s*(?:[A-Z]|\d+))*)\b", sentence):
            for tok in re.findall(r"[A-Za-z]+|\d+", m.group(1)):
                alias = mark_to_alias.get(tok.lower()) if tok.lower() != "and" else None
                if alias is not None and alias not in hit:
                    hit.append(alias)
        for a in hit:
            out.setdefault(a, kind)
    return out


@dataclass
class ObserveResult:
    kind: str  # initial (M2) | turn (M5) | event (event frame)
    statements: list[EdgeKey]
    reply: str = ""
    parse: ObserveParse = field(default_factory=ObserveParse)
    error: str | None = None
    usage: Any = None
    prompt_text: str = ""  # the prompt's text parts (images left out), for the trace

    @property
    def ok(self) -> bool:
        return self.error is None and self.parse.ok

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "statements": [statement_text(k) for k in self.statements], "ok": self.ok,
                "error": self.error or self.parse.error, "reply": self.reply, "parsed": self.parse.to_dict()}


def prompt_text_only(prompt: list[dict[str, Any]]) -> str:
    """Text parts of a chat prompt (images replaced by ``<image>``), for logs."""
    out = []
    for msg in prompt:
        c = msg.get("content")
        if isinstance(c, str):
            out.append(f"[{msg.get('role')}] {c}")
            continue
        parts = []
        for p in c or []:
            parts.append(p.get("text", "") if p.get("type") == "text" else "<image>")
        out.append(f"[{msg.get('role')}] " + "\n".join(parts))
    return "\n".join(out)


class VdmObserver:
    """``query_fn(prompt) -> str | {"content", "usage"}`` is the VDM."""

    def __init__(self, query_fn: Callable[[list[dict[str, Any]]], Any]) -> None:
        self.query_fn = query_fn
        self.calls = 0

    def observe(self, kind: str, prompt: list[dict[str, Any]], statements: Sequence[EdgeKey],
                names: Sequence[str]) -> ObserveResult:
        res = ObserveResult(kind, list(statements), prompt_text=prompt_text_only(prompt))
        if not statements:
            res.error = "no statements"
            return res
        self.calls += 1
        try:
            out = self.query_fn(prompt)
            if isinstance(out, dict):
                res.reply, res.usage = out.get("content") or "", out.get("usage")
            else:
                res.reply = str(out or "")
        except Exception as exc:  # server error: geometry only
            res.error = f"query failed: {exc}"
            return res
        try:
            if not isinstance(res.reply, str):  # e.g. a list of content parts or a number
                res.reply = json.dumps(res.reply) if isinstance(res.reply, (list, dict)) else str(res.reply)
            res.parse = parse_observe(res.reply, statements, names)
        except Exception as exc:  # an odd reply never ends the trial: parse failure -> geometry only
            res.parse = ObserveParse(error=f"parse failed: {type(exc).__name__}: {exc}")
            res.error = res.parse.error
        return res
