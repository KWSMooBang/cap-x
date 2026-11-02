"""Backward pass of mode A (design: 뒤로 가는 패스에서 해도 되는 것과 안 되는 것).

Rule: the label at t says only what was knowable from observations up to t. Allowed here:

* B1 track ID unification: an alias the tracker later merged into another (``Tracker.merges``) is
  renamed in every row (the object was the same one at t too).
* B1b same-category merge (bggen-fix3 B4; ``merge_tracks_backward``): two aliases of one category never detected
  together, the later one appearing after the earlier one was lost, the gap explained by a carry or a base motion,
  are one object (B1K hotdog_C..F). Renamed like B1 in every row (belief, Observe, S4, Subtask); the later alias's
  edges win at the rows where both exist. Never below the init's count of the category. Not in mode B.
* B2 single-shot noise: an Observe 0 / 1 that disagrees with the previous and the next 0 / 1 answers
  (which agree), with no robot contact on its objects in between, is corrected to the neighbours'
  value, and the belief rows it changed are restored. Hindsight answers (mode A tool) count as
  neighbours. A ``?`` is never touched.
* B3 Effect = state difference between a segment's start and the row where its end state is
  confirmed (decided by an observing source); attached to that row only.
* B4 Subtask = the segment's intent, from the approach before the segment starts (Ver1 style). A row's
  Subtask holds from its frame on: pick -> place switches at the close event row, the next pick starts at the
  release row (its approach); the row that confirms the end state (Effect) never moves a boundary. Only a close
  on nothing (a press) or on an openable is articulated (open / close / toggle_on / manipulate); a retracted grasp
  of a bowl keeps pick / place labels, and a segment without an Effect keeps them too (B3 logs ``effect []``).
* B4 keeps the forward boundaries (bggen-fix6 C; ``keep_forward``): every frame where the forward Subtask changed is a
  boundary of the backward Subtask too. Within each B4 span (a segment's approach, its hold, the rows after the last
  segment) the B4 label goes to the LAST forward sub-span only; the earlier forward sub-spans keep their forward labels
  (batch4 B1K 020: one place label over 2030-6432 erased the forward boundary 2933; the tail erased 7537 / 7783). The
  only exception is two adjacent sub-spans that end up with the same label (merged; logged ``B4-merged``).

Forbidden (enforced, and tested): filling ``?`` from later frames, changing Delta / goal progress,
writing a hidden process state (cooked behind a closed door) as an observation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    uncertain: bool = False  # a provisional grasp never verified (the release did not settle it either)
    retracted: bool = False  # the co-motion check found the object where it was (nothing carried)


@dataclass
class BackwardResult:
    rows: list[Any]
    log: list[dict[str, Any]]
    segments: list[Segment]
    merges_backward: list[dict[str, Any]] = field(default_factory=list)  # B1b


def _rename(key: EdgeKey, a: str, b: str) -> EdgeKey:
    return tuple(b if x == a else x for x in key)  # type: ignore[return-value]


def unify_ids(rows: list[Any], merges: Sequence[dict[str, Any]], log: list[dict[str, Any]]) -> None:
    for m in merges:
        a, b = m.get("from"), m.get("into")
        if not a or not b:
            continue
        # a forward late merge: the surviving alias's own edge wins (the fresh alias lived for a frame or two). A
        # backward merge (``prefer_from``): the later alias was the one being observed, the surviving alias's edges
        # at those rows are stale memory -> the renamed edge wins
        prefer = bool(m.get("prefer_from"))
        word = re.compile(rf"\b{re.escape(a)}\b")
        n = 0
        for r in rows:
            r.observed = [(_rename(k, a, b), v) for k, v in r.observed]
            # provenance for the evaluation (B1b only): ``b1b_origin`` final key -> EVERY merged alias it passed through
            # (a chain X -> Y -> Z gives [X, Y]); ``b1b_shadowed`` the entries a collision displaced (key, value, the
            # same alias list; [] = the survivor's own). Each merge renames shadowed keys too and extends their lists.
            old_origin = {k: list(v) for k, v in getattr(r, "b1b_origin", {}).items()}
            origin: dict[EdgeKey, list[str]] = {}
            shadow = []
            for sk, sv, so in getattr(r, "b1b_shadowed", []):
                sk2 = _rename(sk, a, b)
                shadow.append((sk2, sv, list(so) + ([a] if sk2 != sk else [])))
            nb: dict[EdgeKey, Any] = {}
            for k, v in r.belief.items():
                k2 = _rename(k, a, b)
                ren = k2 != k
                o = old_origin.get(k, []) + ([a] if ren else [])
                n += ren
                if k2 in nb:
                    new_wins = ren if prefer else not ren  # the preferred origin wins a collision
                    if not new_wins:
                        if prefer:
                            shadow.append((k2, v, o))
                        continue
                    if prefer:
                        shadow.append((k2, nb[k2], origin.get(k2, [])))
                nb[k2] = v
                origin[k2] = o
            r.belief = nb
            if prefer or old_origin:
                r.b1b_origin = {k: o for k, o in origin.items() if o}
                r.b1b_shadowed = shadow
            r.s4 = [word.sub(b, x) for x in r.s4]
            r.changed = [_rename(k, a, b) for k in r.changed]
            r.hindsight = {_rename(k, a, b): v for k, v in r.hindsight.items()}
            r.rechecked = {_rename(k, a, b): v for k, v in getattr(r, "rechecked", {}).items()}
            r.rejected = [_rename(k, a, b) for k in getattr(r, "rejected", [])]
            r.touched = [b if x == a else x for x in r.touched]
            r.held = {k: (b if v == a else v) for k, v in r.held.items()}
            if r.obj == a:
                r.obj = b
        log.append({"rule": "B1", "from": a, "into": b, "t": m.get("t"), "entries_renamed": n})


def sightings(trace: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per alias from the forward trace: category, frames it was detected at, its position then (B1b input)."""
    out: dict[str, dict[str, Any]] = {}
    for line in trace:
        for tr in line.get("tracks", []):
            s = out.setdefault(tr["alias"], {"category": tr["category"], "seen": [], "pos": []})
            if tr.get("seen_now") and (not s["seen"] or s["seen"][-1] != line["t"]):
                s["seen"].append(line["t"])
                s["pos"].append(tr.get("pos"))
    return out


def merge_tracks_backward(sight: dict[str, dict[str, Any]], rows: Sequence[Any], base_moving: Sequence[tuple[int, int]],
                          caps: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """B1b (bggen-fix3 B4, mode A only): aliases of one category that are one object seen twice. Aliases in order of
    their first detection; a later alias joins an earlier one (its cluster) when (1) they were never detected at the
    same time (the earlier one's last detection is before the later one's first), and (2) the gap between is explained
    -- the base moved (the camera lost it and found it again) or the earlier object was carried (held / released) in
    it. Among candidates the nearest (last position vs first position) wins, then the most recently lost. A known
    count (``caps``: the init's names) is never merged below: an alias joins a cluster only while the clusters (plus
    names never detected) would exceed the cap. Returns merges {"from", "into", ...} in order (``prefer_from``)."""
    caps = caps or {}
    held = {a: [r.t for r in rows if a in set(r.held.values())] for a in sight}
    released = {a: [r.t for r in rows if r.kind == "release" and r.obj == a] for a in sight}
    out: list[dict[str, Any]] = []
    for cat in sorted({s["category"] for s in sight.values()}):
        names = [a for a, s in sight.items() if s["category"] == cat]
        seen = sorted((a for a in names if sight[a]["seen"]), key=lambda a: (sight[a]["seen"][0], a))
        never = len(names) - len(seen)
        cap = caps.get(cat)
        clusters: list[dict[str, Any]] = []
        for b in seen:
            sb = sight[b]
            first = sb["seen"][0]
            cands = []
            for c in clusters:
                if c["last_t"] >= first:
                    continue  # detected together (or overlapping): two objects
                why = []
                if any(s0 < first and s1 > c["last_t"] for s0, s1 in base_moving):
                    why.append("base moved")
                if any(c["last_t"] <= t <= first for m in c["members"] for t in held.get(m, []) + released.get(m, [])):
                    why.append("carried")
                if not why:
                    continue
                p0, p1 = c["last_pos"], sb["pos"][0]
                d = float(np.linalg.norm(np.asarray(p0, float) - np.asarray(p1, float))) if p0 is not None and p1 is not None else 0.0
                cands.append((d, -c["last_t"], len(cands), c, why))
            over = cap is None or len(clusters) + 1 + never > cap
            if cands and over:
                d, _, _, c, why = min(cands, key=lambda x: x[:3])
                out.append({"t": first, "from": b, "into": c["root"], "prefer_from": True, "backward": True,
                            "why": f"never seen together, gap {c['last_t']}-{first} explained ({', '.join(why)}), "
                                   f"distance {d:.2f} m", "gap": [c["last_t"], first]})
                c["members"].append(b)
                c["last_t"], c["last_pos"] = sb["seen"][-1], sb["pos"][-1]
                continue
            clusters.append({"root": b, "members": [b], "last_t": sb["seen"][-1], "last_pos": sb["pos"][-1]})
    return sorted(out, key=lambda m: m["t"])


def _touched_between(rows: Sequence[Any], i0: int, i1: int, key: EdgeKey) -> bool:
    objs = {x for x in (key[1], key[2]) if x is not None}
    return any(objs & set(rows[j].touched) for j in range(i0 + 1, i1 + 1))


def remove_noise(rows: list[Any], log: list[dict[str, Any]]) -> None:
    obs: dict[EdgeKey, list[tuple[float, int, str]]] = {}
    for i, r in enumerate(rows):
        # R10: the first answer of a conflict is never evidence here (not adopted, or superseded by the adopted
        # re-check, which would count the same moment twice); the adopted look re-check IS the row's answer (one
        # entry per row and key, correctable like any Observe answer)
        rejected = set(getattr(r, "rejected", []))
        rechecked = getattr(r, "rechecked", {})
        for k, v in r.observed:
            if v in ("0", "1") and k not in rejected and k not in rechecked:
                obs.setdefault(k, []).append((float(i), i, v))
        for k, v in rechecked.items():
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
            if k in getattr(r, "rechecked", {}):
                r.rechecked[k] = vp
            else:
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
        if r.kind in ("grasp", "close", "close_empty") and r.obj:
            # bggen-fix3 C2: a retracted grasp of an object that does not open / close (a bowl) stays an object
            # segment (pick / place labels: the approach and the close were a pick attempt); only a close on nothing
            # (a press) or on an openable is articulated
            kind = "articulated" if categories.get(r.obj) in OPENABLE_KEYS or r.kind == "close_empty" else "object"
            open_seg[r.arm] = Segment(r.t, r.t, r.arm, r.obj, kind, i, i, uncertain="grasp_unverified" in r.notes,
                                      retracted="grasp_retracted" in r.notes)
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
        if s.uncertain:
            log.append({"rule": "B3", "segment": [s.start_t, s.end_t, s.obj], "effect": None,
                        "why": "grasp never verified: no Effect"})
            continue
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
             navigation: bool, done: bool, log: list[dict[str, Any]], keep_forward: bool = True) -> None:
    fwd = [r.subtask for r in rows]  # the forward (causal) labels, before B4 replaces them
    spans: set[int] = {0}  # first rows of the B4 spans (approach / hold / tail)
    prev_end = -1
    for s in segs:
        spans |= {prev_end + 1, s.start_row}
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
            if s.uncertain:  # the grasp was never verified: an attempt, not a known pick-and-place
                s.label_pick, s.label_place = "maybe_" + s.label_pick, "maybe_" + s.label_place
        # A row's Subtask holds from its frame on (as the forward rows and the 1 Hz rows read it):
        # approach rows after the previous release .. the row before the close -> pick; the close row (the pick ->
        # place transition is the close event itself) .. the row before the release -> place; the release row
        # starts the next segment's approach. The row where the end state is confirmed (``er``, Effect) does not
        # move a boundary: confirmation time is evidence only.
        for i in range(prev_end + 1, len(rows)):
            t = rows[i].t
            if i < s.start_row:
                lab = s.label_pick
                if navigation and any(a <= t < b for a, b in base_moving):
                    lab = f"move_to({s.obj})"
                rows[i].subtask = lab
            elif i < s.end_row or (i == s.end_row and s.end_row == s.start_row):
                rows[i].subtask = s.label_place
            else:
                break
        prev_end = max(s.end_row - 1, s.start_row)
        log.append({"rule": "B4", "segment": [s.start_t, s.end_t, s.obj], "pick": s.label_pick, "place": s.label_place,
                    "pick_to_place_t": s.start_t, "next_from_t": s.end_t, "confirmed_t": rows[er].t})
    last = segs[-1] if segs else None
    spans.add(prev_end + 1)
    for i in range(prev_end + 1, len(rows)):  # after the last release: "done" only from the row where the goal was confirmed
        r = rows[i]
        if done and r.goal_total and r.goal_done >= r.goal_total:
            r.subtask = "done"
        elif last is not None:
            r.subtask = last.label_place  # the last place, not yet confirmed as the goal
    if keep_forward:
        _keep_forward(rows, fwd, spans, prev_end + 1, log)


def _keep_forward(rows: list[Any], fwd: Sequence[str], spans: set[int], tail: int, log: list[dict[str, Any]]) -> None:
    """bggen-fix6 C: the backward labels keep every forward boundary. Units = runs of one backward label, cut at the B4
    span starts; a unit with forward boundaries inside keeps the forward labels up to its last forward boundary and the
    backward label from there (the end of the unit: the pick / place that B4 knows). After the last segment (rows from
    ``tail``) B4 knows no further segment: a tail unit with forward boundaries inside keeps the forward labels throughout
    (a ``done`` unit, from the goal-confirmed row, is a normal unit). A forward boundary that would still disappear
    (the backward label of one side equals the forward label of the other: e.g. the hold's end got the place label the
    forward rows name right after the release) -> the unit on that side whose end carries the backward label takes the
    forward labels throughout (``B4-forward-unit``); one that remains (none expected) is logged ``B4-merged``."""
    n = len(rows)
    if n < 2 or len(fwd) != n:
        return
    bwd = [r.subtask for r in rows]
    starts = [0] + [i for i in range(1, n) if bwd[i] != bwd[i - 1] or i in spans]
    units = [[a, starts[u + 1] if u + 1 < len(starts) else n, None] for u, a in enumerate(starts)]  # [a, b, c]
    unit_of = [0] * n
    for u, (a, b, _) in enumerate(units):
        for i in range(a, b):
            unit_of[i] = u
    for un in units:
        a, b, _ = un
        inner = [i for i in range(a + 1, b) if fwd[i] != fwd[i - 1]]
        if not inner:
            un[2] = a  # the backward label throughout
            continue
        c = inner[-1]
        if fwd[c - 1] == bwd[a] or (a >= tail and bwd[a] != "done"):
            c = b  # forward labels throughout (the tail; or the sub-span before c already reads as the backward label)
        un[2] = c

    def label(i: int) -> str:
        return fwd[i] if i < units[unit_of[i]][2] else bwd[i]

    flipped = []
    for _ in range(len(units) + 1):
        lost = [i for i in range(1, n) if fwd[i] != fwd[i - 1] and label(i) == label(i - 1)]
        if not lost:
            break
        changed = False
        for i in lost:
            for u in (unit_of[i], unit_of[i - 1]):  # the later unit first (the tail after a release)
                if units[u][2] < units[u][1]:  # this unit carries a backward label: forward throughout
                    units[u][2] = units[u][1]
                    flipped.append(rows[units[u][0]].t)
                    changed = True
                    break
        if not changed:
            break
    kept = []
    for a, b, c in units:
        if c > a:
            kept.append({"unit": [rows[a].t, rows[b - 1].t], "label": bwd[a], "forward_kept_until": rows[c - 1].t,
                         "forward_boundaries": [rows[i].t for i in range(a + 1, b) if fwd[i] != fwd[i - 1]]})
    for i in range(n):
        rows[i].subtask = label(i)
    if kept:
        log.append({"rule": "B4-keep", "units": kept})
    if flipped:
        log.append({"rule": "B4-forward-unit", "t": flipped, "why": "a forward boundary at the unit's edge"})
    merged = [rows[i].t for i in range(1, n) if fwd[i] != fwd[i - 1] and rows[i].subtask == rows[i - 1].subtask]
    if merged:
        log.append({"rule": "B4-merged", "t": merged, "why": "adjacent sub-spans with the same label"})


def check_forbidden(fwd: Sequence[Any], bwd: Sequence[Any], hidden_keys: Sequence[tuple[int, EdgeKey]] = (),
                    renamed: dict[str, str] | None = None) -> list[str]:
    """Violations of the forbidden list (empty = clean). Compares the backward rows with the forward rows.
    ``renamed``: B1b alias map (from -> into); the forward rows are read through it (the renamed alias's entry)."""
    bad = []
    ren = renamed or {}

    def rk(k: EdgeKey) -> EdgeKey:
        return tuple(ren.get(x, x) if x is not None else x for x in k)  # type: ignore[return-value]

    for a, b in zip(fwd, bwd, strict=False):
        if a.delta != b.delta or a.goal_done != b.goal_done:
            bad.append(f"t={a.t}: Delta changed")
        av = dict(a.observed)
        abel = dict(a.belief)
        if ren:
            av = {**{rk(k): v for k, v in a.observed if rk(k) == k}, **{rk(k): v for k, v in a.observed if rk(k) != k}}
            abel = {**{k: v for k, v in a.belief.items() if rk(k) == k}, **{rk(k): v for k, v in a.belief.items() if rk(k) != k}}
        for k, v in b.observed:
            if av.get(k) == "?" and v != "?":
                bad.append(f"t={a.t}: {statement_text(k)} ? filled with {v}")
        for k, (p, _src) in b.belief.items():
            pa = abel.get(k)
            if av.get(k) == "?" and pa is not None and decided(pa[0]) != decided(p):
                bad.append(f"t={a.t}: belief of unobserved {statement_text(k)} changed")
    for t, k in hidden_keys:
        for r in bwd:
            if r.t == t and k in r.belief and r.belief[k][1] == "obs":
                bad.append(f"t={t}: hidden {statement_text(k)} written as obs")
    return bad


def backward(fwd_rows: Sequence[Any], merges: Sequence[dict[str, Any]], categories: dict[str, str],
             base_moving: Sequence[tuple[int, int]] = (), navigation: bool = False, goal_done: bool = False,
             sight: dict[str, dict[str, Any]] | None = None, caps: dict[str, int] | None = None) -> BackwardResult:
    """``sight`` (``sightings(trace)``): enables B1b, the same-category merge of aliases never seen together."""
    rows = [r.copy() for r in fwd_rows]
    log: list[dict[str, Any]] = []
    unify_ids(rows, merges, log)
    if sight:
        sight = {a: {**s, "seen": list(s["seen"]), "pos": list(s["pos"])} for a, s in sight.items()}
        for m in merges:  # forward merges first: one alias, the union of its sightings
            a, b = m.get("from"), m.get("into")
            if a in sight and b in sight:
                sa = sight.pop(a)
                both = sorted(zip(sight[b]["seen"] + sa["seen"], sight[b]["pos"] + sa["pos"], strict=True), key=lambda x: x[0])
                sight[b]["seen"], sight[b]["pos"] = [x[0] for x in both], [x[1] for x in both]
    late = merge_tracks_backward(sight, rows, base_moving, caps) if sight else []
    unify_ids(rows, late, log)
    for m in late:
        log.append({"rule": "B1b", **m})
    remove_noise(rows, log)
    segs = segments(rows, categories)
    effects(rows, segs, log)
    subtasks(rows, segs, categories, base_moving, navigation, goal_done, log)
    return BackwardResult(rows, log, segs, late)
