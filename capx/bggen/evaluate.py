"""Evaluation of generated labels against the oracle (design: 검증 사다리 · 이 에피소드에서 재는 것).
Separate from the generator: reads ``<out>/rows_*.jsonl``, ``trace.jsonl``, ``summary.json`` and
``<root>/oracle/``.

Oracle object <-> generated track (``track_map``): over the WHOLE episode, not only frame 0. At every trace
line, each track detected now (or held) is matched to the nearest oracle object of its category by 3D
position (greedy, within ``map_radius``); the map is carried forward and updated when a track is matched
to another object. Categories with one oracle object (fridge, microwave, countertop) map every track of
the category to it (no position needed: the B1K countertop has none). The map, its changes and the
per-alias counts are written to eval.json / eval.md.

Metrics, for the forward pass (= mode B) and forward + backward (mode A), one line each:

* relation accuracy: decided relations (P >= 0.7 or <= 0.3) in each row's Belief vs the oracle, over the
  relations the oracle covers, with the map at the row's frame; undecided rate (0.3 < P < 0.7) over the same
  keys. Event tolerance ``tol`` frames (default 5, B1K 15): a decided value counts as right if the oracle has
  it at any frame within +-tol (an annotation boundary or a release one frame off is not an error); how
  many rows that saved is reported. Per relation kind in a table.
* articulated grasps (B1K: fridge / microwave door handles): the oracle inhand covers only objects that move as a
  body, so inhand / handempty rows inside an "open door" / "close door" / "turn on switch" segment (+-tol) are
  counted apart (``articulated``) and not in the accuracy.
* Subtask boundary error (frames), two references: ``intent`` = the oracle's segment starts (approach included:
  LIBERO pick from the previous release, B1K skill annotation starts), ``contact`` = frames where the oracle's
  inhand changes (grasp / release). Each reference boundary vs the nearest frame where the Subtask changes.
* name consistency: per alias, the share of its matched observations that went to its most frequent oracle
  object (a stable name = 1.0); oracle objects that got several aliases are listed (splits).
* goal consistency: the last graph satisfies the goal's last step, vs the oracle's final success
* re-observation agreement: a decided relation answered ? (hidden) and later answered 0 / 1 -- does the
  answer match the remembered value
"""

from __future__ import annotations

import json
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .graph import decided
from .oracle import Oracle, load_oracle
from .vocab import parse_statement

DEFAULT_TOL = {"b1k": 15}  # frames; others 5
DEFAULT_MAP_RADIUS = {"b1k": 0.6}  # m; others 0.15
ARTICULATED_LABELS = ("open door", "close door", "turn on switch", "turn off switch", "open", "close", "toggle_on")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


# --- oracle object <-> track --------------------------------------------------------------------------

class TrackMap:
    """Alias -> oracle object over time (see the module docstring)."""

    def __init__(self, trace: Sequence[dict[str, Any]], oracle: Oracle, radius: float = 0.15) -> None:
        self.radius = radius
        by_cat: dict[str, list[str]] = {}
        for sim, cat in oracle.objects.items():
            by_cat.setdefault(cat, []).append(sim)
        single = {c: s[0] for c, s in by_cat.items() if len(s) == 1}
        cur: dict[str, str] = {}
        self.ts: list[int] = []
        self.maps: list[dict[str, str]] = []
        self.changes: list[dict[str, Any]] = []
        self.counts: dict[str, dict[str, int]] = {}
        self.unmatched: dict[str, int] = {}
        for line in sorted(trace, key=lambda x: x["t"]):
            t = line["t"]
            st = oracle.at(t) or {}
            pos = {k: np.asarray(v, float) for k, v in st.get("pos", {}).items()}
            pairs = []
            for tr in line.get("tracks", []):
                a, cat = tr["alias"], tr["category"]
                if cat in single:
                    if cur.get(a) != single[cat]:
                        self._set(cur, t, a, single[cat], None)
                    if tr.get("seen_now"):
                        self._count(a, single[cat])
                    continue
                if tr.get("pos") is None or not (tr.get("seen_now") or tr.get("held_by")):
                    continue
                for sim in by_cat.get(cat, []):
                    if sim in pos:
                        pairs.append((float(np.linalg.norm(np.asarray(tr["pos"], float) - pos[sim])), a, sim))
            used_a: set[str] = set()
            used_s: set[str] = set()
            for d, a, sim in sorted(pairs):
                if a in used_a or sim in used_s or d > radius:
                    continue
                used_a.add(a)
                used_s.add(sim)
                if cur.get(a) != sim:
                    self._set(cur, t, a, sim, d)
                self._count(a, sim)
            for _d, a, _s in pairs:
                if a not in used_a:
                    self.unmatched[a] = self.unmatched.get(a, 0) + 1
            self.ts.append(t)
            self.maps.append(dict(cur))

    def _set(self, cur: dict[str, str], t: int, a: str, sim: str, d: float | None) -> None:
        self.changes.append({"t": t, "alias": a, "from": cur.get(a), "to": sim, "dist": None if d is None else round(d, 3)})
        cur[a] = sim

    def _count(self, a: str, sim: str) -> None:
        c = self.counts.setdefault(a, {})
        c[sim] = c.get(sim, 0) + 1

    def at(self, t: int) -> dict[str, str]:
        i = bisect_right(self.ts, t) - 1
        return self.maps[i] if i >= 0 else {}

    def first(self) -> dict[str, str]:
        return self.maps[0] if self.maps else {}

    def final(self) -> dict[str, str]:
        return self.maps[-1] if self.maps else {}

    def to_dict(self, max_changes: int = 60) -> dict[str, Any]:
        per_obj: dict[str, list[str]] = {}
        for a, c in self.counts.items():
            per_obj.setdefault(max(c, key=c.get), []).append(a)
        return {"radius": self.radius, "first": self.first(), "final": self.final(),
                "n_changes": sum(1 for c in self.changes if c["from"] is not None),
                "changes": self.changes[:max_changes], "counts": self.counts, "aliases_per_object": per_obj,
                "unmatched_observations": self.unmatched}


def alias_map(trace0: dict[str, Any], oracle: Oracle, t: int = 0, radius: float = float("inf")) -> dict[str, str]:
    """Frame-0 map only (kept for comparison with the earlier evaluation)."""
    return TrackMap([{**trace0, "t": t}], oracle, radius).first()


# --- relations ---------------------------------------------------------------------------------------

def _sim_key(key: tuple, amap: dict[str, str]) -> list[tuple] | None:
    pred, a, b = key
    if pred in ("inhand", "handempty"):
        return None
    sa = amap.get(a)
    if sa is None:
        return None
    if b is None:
        return [(pred, sa, None)]
    if b == "table":
        return [(pred, sa, "table")]
    sb = amap.get(b)
    if sb is None:
        return None
    return [(pred, sa, sb), (pred, sa, f"{sb}_contain_region")]


def _truth(key: tuple, state: dict[str, Any], amap: dict[str, str]) -> bool | None:
    """The oracle value of a generated key in one oracle state (None: not covered / not mapped)."""
    inhand = state.get("inhand", {})
    if key[0] == "inhand":
        arm = key[2] or "main"
        if arm not in inhand or key[1] not in amap:
            return None
        return inhand[arm] == amap[key[1]]
    if key[0] == "handempty":
        arm = key[1] or "main"
        if arm not in inhand:
            return None
        return inhand[arm] is None
    sk = _sim_key(key, amap)
    if sk is None:
        return None
    truth = {(p[0], p[1], p[2] if len(p) > 2 else None) for p in state.get("true", [])}
    return any(k in truth for k in sk)


def _window(oracle: Oracle, t: int, tol: int) -> list[dict[str, Any]]:
    """Oracle states that hold at some frame of [t - tol, t + tol] (the one at t first)."""
    out = [s for s in (oracle.at(t), oracle.at(t - tol)) if s is not None]
    lo, hi = bisect_left(oracle._ts, t - tol), bisect_right(oracle._ts, t + tol)
    return out + oracle.states[lo:hi]


def articulated_intervals(oracle: Oracle) -> list[tuple[int, int, str]]:
    return [(int(s["start"]), int(s["end"]), str(s.get("label"))) for s in oracle.segments
            if str(s.get("label", "")).lower() in ARTICULATED_LABELS]


def relation_accuracy(rows: Sequence[dict[str, Any]], oracle: Oracle, tmap: TrackMap | dict[str, str], names: Sequence[str],
                      arms: Sequence[str | None], tol: int = 0) -> dict[str, Any]:
    covered = set(oracle.relations)
    art = articulated_intervals(oracle)
    right = wrong = undecided = saved = unmapped = 0
    per_pred: dict[str, list[int]] = {}
    art_counts: dict[str, int] = {}
    errors = []
    for r in rows:
        t = r["t"]
        amap = tmap.at(t) if isinstance(tmap, TrackMap) else tmap
        states = _window(oracle, t, tol)
        if not states:
            continue
        in_art = any(a - tol <= t <= b + tol for a, b, _ in art)
        for stmt, (p, src) in r["belief"].items():
            key = parse_statement(stmt, names, arms)
            if key is None or key[0] not in covered:
                continue
            if in_art and key[0] in ("inhand", "handempty"):
                d = decided(p)
                tag = "undecided" if d is None else f"{key[0]}={int(d)}"
                art_counts[tag] = art_counts.get(tag, 0) + 1
                continue
            vals = [_truth(key, s, amap) for s in states]
            if vals[0] is None:
                unmapped += 1
                continue
            d = decided(p)
            if d is None:
                undecided += 1
                per_pred.setdefault(key[0], [0, 0, 0])[2] += 1
                continue
            options = {v for v in vals if v is not None}
            ok = d in options
            saved += ok and d != vals[0]
            right += ok
            wrong += not ok
            pp = per_pred.setdefault(key[0], [0, 0, 0])
            pp[0 if ok else 1] += 1
            if not ok and len(errors) < 40:
                errors.append({"t": t, "stmt": stmt, "p": p, "source": src, "truth": vals[0],
                               "objects": {x: amap.get(x) for x in key[1:] if x and x != "table" and x in amap}})
    n = right + wrong
    return {"accuracy": round(right / n, 4) if n else None, "decided": n, "undecided": undecided,
            "undecided_rate": round(undecided / (n + undecided), 4) if n + undecided else None,
            "tolerance_frames": tol, "right_only_within_tolerance": saved, "not_mapped": unmapped,
            "per_pred": {k: {"right": v[0], "wrong": v[1], "undecided": v[2],
                             "accuracy": round(v[0] / (v[0] + v[1]), 4) if v[0] + v[1] else None} for k, v in per_pred.items()},
            "articulated": {"intervals": [list(x) for x in art], "excluded_robot_keys": art_counts},
            "errors": errors}


# --- boundaries --------------------------------------------------------------------------------------

def _boundary(gen: Sequence[int], ref: Sequence[int]) -> dict[str, Any]:
    if not ref or not gen:
        return {"mean": None, "median": None, "n_oracle": len(ref), "n_generated": len(gen)}
    errs = []
    for b in ref:
        g = min(gen, key=lambda g: abs(b - g))
        errs.append(g - b)  # signed: + late, - early
    ab = [abs(e) for e in errs]
    return {"mean": round(statistics.mean(ab), 1), "median": statistics.median(ab), "n_oracle": len(ref),
            "n_generated": len(gen), "signed": errs, "reference": list(ref)}


def contact_boundaries(oracle: Oracle) -> list[int]:
    """Frames where the oracle's inhand changes on any arm (grasp / release)."""
    out, prev = [], None
    for s in oracle.states:
        cur = s.get("inhand") or {}
        if prev is not None and cur != prev and s["t"] > 0:
            out.append(int(s["t"]))
        prev = cur
    return sorted(set(out))


def boundary_error(rows: Sequence[dict[str, Any]], oracle_or_segments: Oracle | Sequence[dict[str, Any]]) -> dict[str, Any]:
    """{"intent": vs segment starts, "contact": vs oracle inhand changes} (an old call with a segment list returns the
    intent result only)."""
    gen = [rows[i]["t"] for i in range(1, len(rows)) if rows[i]["subtask"] != rows[i - 1]["subtask"]]
    if not isinstance(oracle_or_segments, Oracle):
        return _boundary(gen, sorted({int(s["start"]) for s in oracle_or_segments if int(s["start"]) > 0}))
    o = oracle_or_segments
    intent = sorted({int(s["start"]) for s in o.segments if int(s["start"]) > 0})
    return {"intent": _boundary(gen, intent), "contact": _boundary(gen, contact_boundaries(o)), "generated": gen}


# --- names / re-observation --------------------------------------------------------------------------

def name_consistency(tmap: TrackMap) -> dict[str, Any]:
    ok = n = 0
    unstable = []
    for a, c in tmap.counts.items():
        tot, top = sum(c.values()), max(c.values())
        ok += top
        n += tot
        if top < tot:
            unstable.append({"alias": a, "objects": c})
    per_obj = tmap.to_dict()["aliases_per_object"]
    split = {s: v for s, v in per_obj.items() if len(v) > 1}
    return {"rate": round(ok / n, 4) if n else None, "n": n, "unstable_aliases": unstable[:20],
            "objects_with_several_aliases": split,
            "map_changes": sum(1 for c in tmap.changes if c["from"] is not None)}


def reobservation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    agree = total = 0
    hidden: dict[str, bool] = {}  # stmt -> remembered value while answered ?
    prev_belief: dict[str, Any] = {}
    for r in rows:
        for stmt, v in r["observed"]:
            mem = prev_belief.get(stmt)
            if v == "?" and mem is not None and decided(mem[0]) is not None:
                hidden.setdefault(stmt, bool(decided(mem[0])))
            elif v in ("0", "1") and stmt in hidden:
                total += 1
                agree += hidden.pop(stmt) == (v == "1")
        prev_belief = r["belief"]
    return {"rate": round(agree / total, 4) if total else None, "n": total}


# --- run -----------------------------------------------------------------------------------------------

def evaluate(out_dir: str | Path, root: str | Path, tol: int | None = None, map_radius: float | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    oracle = load_oracle(root)
    trace = _jsonl(out / "trace.jsonl")
    summary = json.loads((out / "summary.json").read_text())
    bench = summary.get("benchmark", "")
    tol = DEFAULT_TOL.get(bench, 5) if tol is None else tol
    radius = DEFAULT_MAP_RADIUS.get(bench, 0.15) if map_radius is None else map_radius
    names = sorted({t["alias"] for line in trace for t in line.get("tracks", [])} | {t["alias"] for t in summary.get("tracks", [])})
    arms = summary.get("arms") or [None]
    tmap = TrackMap(trace, oracle, radius)
    res: dict[str, Any] = {"alias_map": tmap.first(), "track_map": tmap.to_dict(), "oracle_relations": oracle.relations,
                           "tolerance_frames": tol, "name_consistency": name_consistency(tmap),
                           "init_failed": summary.get("init_failed"), "track_identity": summary.get("track_identity")}
    fin = summary.get("goal", {}).get("final_satisfied")
    res["goal_consistency"] = {"final_graph_satisfies_goal": fin, "oracle_success": oracle.final.get("success"),
                               "goal_known": bool(summary.get("goal", {}).get("branches"))}
    for tag, fname in (("forward", "rows_forward.jsonl"), ("forward+backward", "rows_backward.jsonl")):
        rows = _jsonl(out / fname)
        if not rows:
            continue
        res[tag] = {"relations": relation_accuracy(rows, oracle, tmap, names, arms, tol),
                    "subtask_boundary": boundary_error(rows, oracle),
                    "reobservation": reobservation(rows)}
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    (out / "eval.md").write_text(eval_table(res))
    return res


def eval_table(res: dict[str, Any]) -> str:
    lines = []
    if res.get("init_failed"):
        lines += [f"> **INIT FAILED — NO GOAL**: {res['init_failed']}", ""]
    lines += [f"Event tolerance: +-{res.get('tolerance_frames')} frames (a value the oracle has within it counts as right).", "",
              "| pass | rel. accuracy | decided | undecided rate | right only within tol. | boundary err intent (mean / median) "
              "| boundary err contact (mean / median) | re-observation |",
              "|---|---|---|---|---|---|---|---|"]
    for tag in ("forward", "forward+backward"):
        m = res.get(tag)
        if not m:
            continue
        r, b, o = m["relations"], m["subtask_boundary"], m["reobservation"]
        bi, bc = b["intent"], b["contact"]
        lines.append(f"| {tag} | {r['accuracy']} | {r['decided']} | {r['undecided_rate']} | {r['right_only_within_tolerance']} | "
                     f"{bi['mean']} / {bi['median']} (n={bi['n_oracle']}) | {bc['mean']} / {bc['median']} (n={bc['n_oracle']}) | "
                     f"{o['rate']} (n={o['n']}) |")
    lines += ["", "Accuracy per relation kind (right / wrong / undecided):", "",
              "| relation | " + " | ".join(t for t in ("forward", "forward+backward") if res.get(t)) + " |",
              "|---|" + "---|" * sum(1 for t in ("forward", "forward+backward") if res.get(t))]
    preds = sorted({p for t in ("forward", "forward+backward") if res.get(t) for p in res[t]["relations"]["per_pred"]})
    for p in preds:
        cells = []
        for t in ("forward", "forward+backward"):
            if not res.get(t):
                continue
            v = res[t]["relations"]["per_pred"].get(p)
            cells.append("-" if v is None else f"{v['accuracy']} ({v['right']} / {v['wrong']} / {v['undecided']})")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    fw = res.get("forward+backward") or res.get("forward")
    if fw:
        art = fw["relations"]["articulated"]
        lines += ["", f"Articulated grasps (door handle / switch segments +-tol, inhand / handempty counted apart): "
                      f"{art['intervals']} -> {art['excluded_robot_keys']}",
                  f"Keys not evaluated (an object not mapped to the oracle): {fw['relations']['not_mapped']}",
                  f"Generated boundaries (frames): {fw['subtask_boundary'].get('generated')}",
                  f"Signed boundary errors (+ late): intent {fw['subtask_boundary']['intent'].get('signed')}, "
                  f"contact {fw['subtask_boundary']['contact'].get('signed')}"]
    nc, gc, tm = res["name_consistency"], res["goal_consistency"], res["track_map"]
    lines += ["", f"Name consistency: {nc['rate']} (n={nc['n']}); objects with several aliases: {nc['objects_with_several_aliases']}",
              f"Goal consistency: final graph satisfies goal = {gc['final_graph_satisfies_goal']} (goal known: {gc['goal_known']}), "
              f"oracle success = {gc['oracle_success']}", "",
              f"## Oracle object <-> track map (3D, radius {tm['radius']} m, updated over the episode; {tm['n_changes']} re-maps)",
              f"first: {tm['first']}", f"final: {tm['final']}"]
    for c in [c for c in tm["changes"] if c["from"] is not None or c["t"] > 0][:30]:
        lines.append(f"- t={c['t']}: {c['alias']} {c['from']} -> {c['to']} (d={c['dist']})")
    if res.get("track_identity"):
        lines += ["", f"Generator track identity: {res['track_identity']}"]
    return "\n".join(lines) + "\n"
