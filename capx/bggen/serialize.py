"""Ver1 BG-CoT line format (design: 그래프 스키마와 CoT 형식)::

    Subtask: place(bowl_A, plate_A)
    Belief: (on bowl_A plate_A) 0.90 obs | (on bowl_B plate_C) 0.90 mem | (inhand bowl_A) 0.03 robot | (handempty) 0.97 robot
    Delta: 1/3 · next (on bowl_B plate_B)
    Effect: (on bowl_A plate_A) 0>1 | (on bowl_A plate_B) 1>0
    Observe: (on bowl_A plate_A) 1 | (on bowl_A plate_B) 0 | (on bowl_B plate_C) ?

One row per event / keyframe (``kind: event``) and a 1 Hz expansion (``kind: hz``): each second takes the
Belief / Delta of the last row at or before it, the Subtask of the current segment (mode A: the next
row's label, i.e. the segment's intent; mode B: the last row's causal estimate), and the Effect /
Observe of the event rows inside that second. B1K renames on/in to Ver1's ontop/inside.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .vocab import EdgeKey, statement_text

NONE = "none"


def belief_items(row: Any, rename: dict[str, str] | None = None, max_items: int = 12,
                 goal_keys: Sequence[EdgeKey] = ()) -> list[str]:
    changed = set(row.changed)
    picked = []
    for k, (p, src) in row.belief.items():
        if k[0] == "lifted" and p < 0.5:
            continue
        if p >= 0.5 or src == "conflict" or (k in changed and k[0] in ("on", "in", "inhand", "open", "toggled_on")):
            rank = (0 if k in goal_keys else 1, 0 if k[0] in ("on", "in", "at") else 1 if k[0] in ("open", "toggled_on", "cooked") else 2)
            picked.append((rank, statement_text(k, rename), p, src))
    picked.sort(key=lambda x: (x[0], x[1]))
    return [f"{s} {p:.2f} {src}" for _, s, p, src in picked[:max_items]]


def effect_text(row: Any, rename: dict[str, str] | None = None) -> str:
    return " | ".join(f"{statement_text(k, rename)} {a}>{b}" for k, a, b in row.effect) or NONE


def observe_text(row: Any, rename: dict[str, str] | None = None) -> str:
    return " | ".join(f"{statement_text(k, rename)} {v}" for k, v in row.observed) or NONE


def delta_text(row: Any, rename: dict[str, str] | None = None) -> str:
    d = row.delta
    if rename:
        for a, b in rename.items():
            d = d.replace(f"({a} ", f"({b} ")
    return d


def row_fields(row: Any, rename: dict[str, str] | None = None, max_items: int = 12,
               goal_keys: Sequence[EdgeKey] = ()) -> dict[str, str]:
    return {"Subtask": row.subtask, "Belief": " | ".join(belief_items(row, rename, max_items, goal_keys)) or NONE,
            "Delta": delta_text(row, rename), "Effect": effect_text(row, rename), "Observe": observe_text(row, rename)}


def text(fields: dict[str, str]) -> str:
    return "\n".join(f"{k}: {fields[k]}" for k in ("Subtask", "Belief", "Delta", "Effect", "Observe"))


def cot_rows(rows: Sequence[Any], fps: float, rename: dict[str, str] | None = None, offline: bool = True,
             max_items: int = 12, goal_keys: Sequence[EdgeKey] = (), n_frames: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        f = row_fields(r, rename, max_items, goal_keys)
        out.append({"kind": "event", "t": r.t, "sec": round(r.t / fps, 2), "event": r.kind, "obj": r.obj, "arm": r.arm,
                    **f, "s4": r.s4, "keyframe": r.keyframe, "text": text(f)})
    if not rows:
        return out
    last_t = n_frames - 1 if n_frames else rows[-1].t
    j = 0
    prev_frame = -1
    for sec in range(int(last_t // fps) + 1):
        frame = int(round(sec * fps))
        while j + 1 < len(rows) and rows[j + 1].t <= frame:
            j += 1
        base = rows[j]
        f = row_fields(base, rename, max_items, goal_keys)
        f["Subtask"] = base.subtask  # a row's Subtask holds from its frame on (forward and backward rows alike)
        inside = [r for r in rows if prev_frame < r.t <= frame]
        f["Effect"] = " | ".join(e for r in inside if (e := effect_text(r, rename)) != NONE) or NONE
        f["Observe"] = " | ".join(o for r in inside if (o := observe_text(r, rename)) != NONE) or NONE
        out.append({"kind": "hz", "t": frame, "sec": sec, "from_row": base.t, **f, "text": text(f)})
        prev_frame = frame
    return out


def write_jsonl(path: str | Path, items: Sequence[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def row_to_json(row: Any) -> dict[str, Any]:
    """Machine-readable row (all keys as statement text) for evaluation."""
    extra = {}
    if getattr(row, "rechecked", None):
        extra["rechecked"] = {statement_text(k): v for k, v in row.rechecked.items()}
    if getattr(row, "rejected", None):
        extra["rejected"] = [statement_text(k) for k in row.rejected]
    if getattr(row, "b1b_origin", None) or getattr(row, "b1b_shadowed", None):
        extra["b1b_origin"] = {statement_text(k): list(o) for k, o in row.b1b_origin.items()}
        extra["b1b_shadowed"] = [{"stmt": statement_text(k), "value": list(v), "origin": list(o)} for k, v, o in row.b1b_shadowed]
    return {"t": row.t, "kind": row.kind, "arm": row.arm, "obj": row.obj,
            "observed": [[statement_text(k), v] for k, v in row.observed], **extra,
            "belief": {statement_text(k): [p, s] for k, (p, s) in row.belief.items()},
            "changed": [statement_text(k) for k in row.changed], "delta": row.delta, "goal_done": row.goal_done,
            "goal_total": row.goal_total, "subtask": row.subtask, "touched": row.touched, "held": row.held,
            "hindsight": {statement_text(k): v for k, v in row.hindsight.items()}, "s4": row.s4,
            "effect": [[statement_text(k), a, b] for k, a, b in row.effect]}
