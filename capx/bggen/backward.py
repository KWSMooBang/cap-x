"""Backward pass of mode A (design: 뒤로 가는 패스에서 해도 되는 것과 안 되는 것).

Rule: the label at t says only what was knowable from observations up to t. Allowed here:

* B1 track ID unification: an alias the tracker later merged into another (``Tracker.merges``) is
  renamed in every row (the object was the same one at t too).
* B2 single-shot noise: an Observe 0 / 1 that disagrees with the previous and the next 0 / 1 answers
  (which agree), with no robot contact on its objects in between, is corrected to the neighbours'
  value, and the belief rows it changed are restored. Hindsight answers (mode A tool) count as
  neighbours. A ``?`` is never touched.
* B3 Effect = state difference between a segment's start and the row where its end state is
  confirmed (decided by an observing source); attached to that row only.
* B4 Subtask = the segment's intent, from the approach before the segment starts (Ver1 style).

Forbidden (enforced, and tested): filling ``?`` from later frames, changing Delta / goal progress,
writing a hidden process state (cooked behind a closed door) as an observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .graph import decided
from .vocab import OPENABLE_KEYS, EdgeKey, is_container, statement_text


@dataclass
class Segment:
    start_t: int  # grasp / close
    end_t: int  # release / open
    arm: str | None
    obj: str
    kind: str  # object | articulated
    start_row: int
    end_row: int
    effect_row: int | None = None
    op: str = ""
    label_pick: str = ""
    label_place: str = ""
    effect: list[tuple[EdgeKey, int, int]] = field(default_factory=list)


@dataclass
class BackwardResult:
    rows: list[Any]
    log: list[dict[str, Any]]
    segments: list[Segment]


def _rename(key: EdgeKey, a: str, b: str) -> EdgeKey:
    return tuple(b if x == a else x for x in key)  # type: ignore[return-value]


def unify_ids(rows: list[Any], merges: Sequence[dict[str, Any]], log: list[dict[str, Any]]) -> None:
    for m in merges:
        a, b = m.get("from"), m.get("into")
        if not a or not b:
            continue
        n = 0
        for r in rows:
            r.observed = [(_rename(k, a, b), v) for k, v in r.observed]
            nb = {}
            for k, v in r.belief.items():
                k2 = _rename(k, a, b)
                if k2 in nb and k2 != k:
                    continue  # the surviving alias's own edge wins
                nb[k2] = v
                n += k2 != k
            r.belief = nb
            r.changed = [_rename(k, a, b) for k in r.changed]
            r.hindsight = {_rename(k, a, b): v for k, v in r.hindsight.items()}
            r.touched = [b if x == a else x for x in r.touched]
            r.held = {k: (b if v == a else v) for k, v in r.held.items()}
            if r.obj == a:
                r.obj = b
        log.append({"rule": "B1", "from": a, "into": b, "t": m.get("t"), "entries_renamed": n})


def _touched_between(rows: Sequence[Any], i0: int, i1: int, key: EdgeKey) -> bool:
    objs = {x for x in (key[1], key[2]) if x is not None}
    return any(objs & set(rows[j].touched) for j in range(i0 + 1, i1 + 1))


def remove_noise(rows: list[Any], log: list[dict[str, Any]]) -> None:
    obs: dict[EdgeKey, list[tuple[float, int, str]]] = {}
    for i, r in enumerate(rows):
        for k, v in r.observed:
            if v in ("0", "1"):
                obs.setdefault(k, []).append((float(i), i, v))
        for k, v in r.hindsight.items():
            if v in ("0", "1"):
                obs.setdefault(k, []).append((i + 0.5, i, v))
    for k, seq in obs.items():
        seq.sort()
        for n in range(1, len(seq) - 1):
            pos, i, v = seq[n]
            if pos != float(i):
                continue  # only real Observe answers are corrected
            _, ip, vp = seq[n - 1]
            _, inx, vn = seq[n + 1]
            if vp != vn or vp == v:
                continue
            if _touched_between(rows, ip, max(inx, i), k) or ip == i:
                continue
            r = rows[i]
            r.observed = [(kk, vp if kk == k else vv) for kk, vv in r.observed]
            prev = rows[ip].belief.get(k)
            restored = []
            if prev is not None:
                for j in range(i, len(rows)):
                    cur = rows[j].belief.get(k)
                    if cur is None or j > inx or (j != i and dict(rows[j].observed).get(k) == "?"):
                        break  # a row where the key was unknown keeps its value (never filled from later)
                    if decided(cur[0]) == decided(prev[0]) and j != i:
                        break
                    rows[j].belief[k] = (prev[0], cur[1] if cur[1] != "obs" or j != i else "obs")
                    restored.append(rows[j].t)
            log.append({"rule": "B2", "key": statement_text(k), "t": r.t, "was": v, "now": vp,
                        "neighbours": [rows[ip].t, rows[inx].t], "belief_restored_at": restored})


def segments(rows: Sequence[Any], categories: dict[str, str]) -> list[Segment]:
    """Gripper segments: grasp -> release (object) and close_empty -> open on an object (articulated / press)."""
    out: list[Segment] = []
    open_seg: dict[Any, Segment] = {}
    for i, r in enumerate(rows):
        if r.kind in ("grasp", "close_empty") and r.obj:
            kind = "articulated" if categories.get(r.obj) in OPENABLE_KEYS or r.kind == "close_empty" else "object"
            open_seg[r.arm] = Segment(r.t, r.t, r.arm, r.obj, kind, i, i)
        elif r.kind in ("release", "open") and r.arm in open_seg:
            s = open_seg.pop(r.arm)
            s.end_t, s.end_row = r.t, i
            out.append(s)
    for s in open_seg.values():  # never released: ends with the episode
        s.end_t, s.end_row = rows[-1].t, len(rows) - 1
        out.append(s)
    return sorted(out, key=lambda s: s.start_t)


def _state(row: Any, obj: str) -> dict[EdgeKey, bool]:
    out = {}
    for k, (p, src) in row.belief.items():
        if k[1] != obj or k[0] in ("inhand", "lifted", "handempty"):
            continue
        d = decided(p)
        if d is not None and src != "conflict":
            out[k] = d
    return out


def effects(rows: list[Any], segs: list[Segment], log: list[dict[str, Any]]) -> None:
    for n, s in enumerate(segs):
        start = _state(rows[max(s.start_row - 1, 0)], s.obj)
        limit = segs[n + 1].start_row if n + 1 < len(segs) else len(rows)
        for j in range(s.end_row, max(limit, s.end_row + 1)):
            r = rows[j]
            loc = [k for k, (p, src) in r.belief.items() if k[1] == s.obj and k[0] in ("on", "in", "open", "toggled_on")]
            uncertain = any(0.3 < r.belief[k][0] < 0.7 for k in loc)
            confirmed = any(r.belief[k][0] >= 0.7 and r.belief[k][1] in ("obs", "robot", "mem") for k in loc)
            if confirmed and not uncertain:
                s.effect_row = j
                break
        if s.effect_row is None:
            log.append({"rule": "B3", "segment": [s.start_t, s.end_t, s.obj], "effect": None,
                        "why": "end state never confirmed before the next segment"})
            continue
        end = _state(rows[s.effect_row], s.obj)
        eff = [(k, int(start[k]), int(end[k])) for k in sorted(set(start) & set(end), key=str) if start[k] != end[k]]
        eff += [(k, 0, 1) for k in sorted(set(end) - set(start), key=str) if end[k] and k[0] in ("on", "in")]
        s.effect = eff
        rows[s.effect_row].effect = list(eff)
        log.append({"rule": "B3", "segment": [s.start_t, s.end_t, s.obj], "t": rows[s.effect_row].t,
                    "effect": [f"{statement_text(k)} {a}>{b}" for k, a, b in eff]})


def _where(row: Any, obj: str) -> str | None:
    best = max(((k, v) for k, v in row.belief.items() if k[1] == obj and k[0] in ("on", "in")),
               key=lambda x: x[1][0], default=None)
    return best[0][2] if best is not None and best[1][0] >= 0.5 else None


def subtasks(rows: list[Any], segs: list[Segment], categories: dict[str, str], base_moving: Sequence[tuple[int, int]],
             navigation: bool, done: bool, log: list[dict[str, Any]]) -> None:
    prev_end = -1
    for s in segs:
        src = _where(rows[max(s.start_row - 1, 0)], s.obj) or "?"
        er = s.effect_row if s.effect_row is not None else s.end_row
        dst = _where(rows[er], s.obj) or "?"
        eff = {statement_text(k): (a, b) for k, a, b in s.effect}
        if s.kind == "articulated":
            op = "open" if eff.get(f"(open {s.obj})") == (0, 1) else "close" if eff.get(f"(open {s.obj})") == (1, 0) else \
                "toggle_on" if eff.get(f"(toggled_on {s.obj})") == (0, 1) else "manipulate"
            s.label_pick = s.label_place = f"{op}({s.obj})"
        else:
            op = "put_in" if dst != "?" and is_container(categories.get(dst, "")) else "place"
            s.label_pick, s.label_place = f"pick({s.obj}, {src})", f"{op}({s.obj}, {dst})"
        for i in range(prev_end + 1, len(rows)):
            t = rows[i].t
            if i <= s.start_row:
                lab = s.label_pick
                if navigation and any(a <= t < b for a, b in base_moving):
                    lab = f"move_to({s.obj})"
                rows[i].subtask = lab
            elif i <= er:
                rows[i].subtask = s.label_place
            else:
                break
        prev_end = er
        log.append({"rule": "B4", "segment": [s.start_t, s.end_t, s.obj], "pick": s.label_pick, "place": s.label_place})
    for i in range(prev_end + 1, len(rows)):  # "done" only from the row where the goal was confirmed
        r = rows[i]
        if done and r.goal_total and r.goal_done >= r.goal_total:
            r.subtask = "done"


def check_forbidden(fwd: Sequence[Any], bwd: Sequence[Any], hidden_keys: Sequence[tuple[int, EdgeKey]] = ()) -> list[str]:
    """Violations of the forbidden list (empty = clean). Compares the backward rows with the forward rows."""
    bad = []
    for a, b in zip(fwd, bwd, strict=False):
        if a.delta != b.delta or a.goal_done != b.goal_done:
            bad.append(f"t={a.t}: Delta changed")
        av = dict(a.observed)
        for k, v in b.observed:
            if av.get(k) == "?" and v != "?":
                bad.append(f"t={a.t}: {statement_text(k)} ? filled with {v}")
        for k, (p, _src) in b.belief.items():
            pa = a.belief.get(k)
            if av.get(k) == "?" and pa is not None and decided(pa[0]) != decided(p):
                bad.append(f"t={a.t}: belief of unobserved {statement_text(k)} changed")
    for t, k in hidden_keys:
        for r in bwd:
            if r.t == t and k in r.belief and r.belief[k][1] == "obs":
                bad.append(f"t={t}: hidden {statement_text(k)} written as obs")
    return bad


def backward(fwd_rows: Sequence[Any], merges: Sequence[dict[str, Any]], categories: dict[str, str],
             base_moving: Sequence[tuple[int, int]] = (), navigation: bool = False, goal_done: bool = False) -> BackwardResult:
    rows = [r.copy() for r in fwd_rows]
    log: list[dict[str, Any]] = []
    unify_ids(rows, merges, log)
    remove_noise(rows, log)
    segs = segments(rows, categories)
    effects(rows, segs, log)
    subtasks(rows, segs, categories, base_moving, navigation, goal_done, log)
    return BackwardResult(rows, log, segs)
