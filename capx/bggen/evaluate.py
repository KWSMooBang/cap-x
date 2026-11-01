"""Evaluation of generated labels against the oracle (design: 검증 사다리 · 이 에피소드에서 재는 것).
Separate from the generator: reads ``<out>/rows_*.jsonl``, ``trace.jsonl``, ``summary.json`` and
``<root>/oracle/``.

Metrics, for the forward pass (= mode B) and forward + backward (mode A), one line each:

* relation accuracy: decided relations (P >= 0.7 or <= 0.3) in each row's Belief vs the oracle at that
  frame, over the relations the oracle covers; undecided rate (0.3 < P < 0.7) over the same keys
* Subtask boundary error (frames): each oracle boundary vs the nearest frame where the Subtask changes
* name consistency: share of detected-track observations whose nearest same-category simulator object is
  the one the alias was mapped to at frame 0 (aliases are mapped by nearest position per category)
* goal consistency: the last graph satisfies the goal's last step, vs the oracle's final success
* re-observation agreement: a decided relation answered ? (hidden) and later answered 0 / 1 -- does the
  answer match the remembered value
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .graph import decided
from .oracle import Oracle, load_oracle
from .vocab import parse_statement


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def alias_map(trace0: dict[str, Any], oracle: Oracle, t: int = 0) -> dict[str, str]:
    st = oracle.at(t)
    if st is None:
        return {}
    pos = {k: np.asarray(v, float) for k, v in st.get("pos", {}).items()}
    out: dict[str, str] = {}
    pairs = []
    for tr in trace0.get("tracks", []):
        if tr.get("pos") is None:
            continue
        for sim, cat in oracle.objects.items():
            if cat == tr["category"] and sim in pos:
                pairs.append((float(np.linalg.norm(np.asarray(tr["pos"]) - pos[sim])), tr["alias"], sim))
    used = set()
    for _d, a, s in sorted(pairs):
        if a in out or s in used:
            continue
        out[a] = s
        used.add(s)
    return out


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


def relation_accuracy(rows: Sequence[dict[str, Any]], oracle: Oracle, amap: dict[str, str], names: Sequence[str],
                      arms: Sequence[str | None]) -> dict[str, Any]:
    covered = set(oracle.relations)
    right = wrong = undecided = 0
    per_pred: dict[str, list[int]] = {}
    errors = []
    for r in rows:
        truth = oracle.truth(r["t"])
        st = oracle.at(r["t"]) or {}
        inhand = st.get("inhand", {})
        for stmt, (p, src) in r["belief"].items():
            key = parse_statement(stmt, names, arms)
            if key is None or key[0] not in covered:
                continue
            if key[0] == "inhand":
                arm = key[2] or "main"
                if arm not in inhand or key[1] not in amap:
                    continue
                true = inhand[arm] == amap[key[1]]
            elif key[0] == "handempty":
                arm = key[1] or "main"
                if arm not in inhand:
                    continue
                true = inhand[arm] is None
            else:
                sk = _sim_key(key, amap)
                if sk is None:
                    continue
                true = any(k in truth for k in sk)
            d = decided(p)
            if d is None:
                undecided += 1
                continue
            ok = d == true
            right += ok
            wrong += not ok
            per_pred.setdefault(key[0], [0, 0])[0 if ok else 1] += 1
            if not ok and len(errors) < 40:
                errors.append({"t": r["t"], "stmt": stmt, "p": p, "source": src, "truth": true})
    n = right + wrong
    return {"accuracy": round(right / n, 4) if n else None, "decided": n, "undecided": undecided,
            "undecided_rate": round(undecided / (n + undecided), 4) if n + undecided else None,
            "per_pred": {k: {"right": v[0], "wrong": v[1]} for k, v in per_pred.items()}, "errors": errors}


def boundary_error(rows: Sequence[dict[str, Any]], segments: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gen = [rows[i]["t"] for i in range(1, len(rows)) if rows[i]["subtask"] != rows[i - 1]["subtask"]]
    ref = sorted({int(s["start"]) for s in segments if int(s["start"]) > 0})
    if not ref or not gen:
        return {"mean": None, "median": None, "n_oracle": len(ref), "n_generated": len(gen)}
    errs = [min(abs(b - g) for g in gen) for b in ref]
    return {"mean": round(statistics.mean(errs), 1), "median": statistics.median(errs), "n_oracle": len(ref),
            "n_generated": len(gen), "errors": errs}


def name_consistency(trace: Sequence[dict[str, Any]], oracle: Oracle, amap: dict[str, str]) -> dict[str, Any]:
    ok = n = 0
    swaps = []
    for line in trace:
        st = oracle.at(line["t"])
        if st is None:
            continue
        pos = {k: np.asarray(v, float) for k, v in st.get("pos", {}).items()}
        for tr in line.get("tracks", []):
            a = tr["alias"]
            if a not in amap or not tr.get("seen_now") or tr.get("pos") is None:
                continue
            cands = [(float(np.linalg.norm(np.asarray(tr["pos"]) - pos[s])), s) for s, c in oracle.objects.items()
                     if c == tr["category"] and s in pos]
            if not cands:
                continue
            n += 1
            nearest = min(cands)[1]
            if nearest == amap[a]:
                ok += 1
            elif len(swaps) < 20:
                swaps.append({"t": line["t"], "alias": a, "mapped": amap[a], "nearest": nearest})
    return {"rate": round(ok / n, 4) if n else None, "n": n, "mismatches": swaps}


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


def evaluate(out_dir: str | Path, root: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    oracle = load_oracle(root)
    trace = _jsonl(out / "trace.jsonl")
    summary = json.loads((out / "summary.json").read_text())
    names = [t["alias"] for t in summary.get("tracks", [])]
    arms = summary.get("arms") or [None]
    amap = alias_map(trace[0], oracle) if trace else {}
    res: dict[str, Any] = {"alias_map": amap, "oracle_relations": oracle.relations,
                           "name_consistency": name_consistency(trace, oracle, amap)}
    fin = summary.get("goal", {}).get("final_satisfied")
    res["goal_consistency"] = {"final_graph_satisfies_goal": fin, "oracle_success": oracle.final.get("success")}
    for tag, fname in (("forward", "rows_forward.jsonl"), ("forward+backward", "rows_backward.jsonl")):
        rows = _jsonl(out / fname)
        if not rows:
            continue
        res[tag] = {"relations": relation_accuracy(rows, oracle, amap, names, arms),
                    "subtask_boundary": boundary_error(rows, oracle.segments),
                    "reobservation": reobservation(rows)}
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    (out / "eval.md").write_text(eval_table(res))
    return res


def eval_table(res: dict[str, Any]) -> str:
    lines = ["| pass | rel. accuracy | decided | undecided rate | subtask boundary err (mean / median frames) | re-observation |",
             "|---|---|---|---|---|---|"]
    for tag in ("forward", "forward+backward"):
        m = res.get(tag)
        if not m:
            continue
        r, b, o = m["relations"], m["subtask_boundary"], m["reobservation"]
        lines.append(f"| {tag} | {r['accuracy']} | {r['decided']} | {r['undecided_rate']} | {b['mean']} / {b['median']} "
                     f"(n={b['n_oracle']}) | {o['rate']} (n={o['n']}) |")
    nc, gc = res["name_consistency"], res["goal_consistency"]
    lines += ["", f"name consistency: {nc['rate']} (n={nc['n']}); goal consistency: final graph satisfies goal = "
                  f"{gc['final_graph_satisfies_goal']}, oracle success = {gc['oracle_success']}",
              f"alias map: {res['alias_map']}"]
    return "\n".join(lines) + "\n"
