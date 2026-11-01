"""Aggregate many bggen runs: per-episode ``eval.json`` + ``summary.json`` -> per-task and overall tables.

    python scripts/bggen/aggregate.py RUNS_DIR [--out RUNS_DIR/aggregate] [--price-in 0.45 --price-out 3]

``RUNS_DIR`` = ``runs/<RUN_TAG>`` of the batch (one folder per episode, ``batch_status.json`` beside them). An episode
folder's group comes from its ``batch_episode.json`` (written by ``capx.bggen.batch``), else from the folder name
(``libero_mem_t06_demo_1`` -> ``libero_mem t06``, ``b1k_t45_e00450010`` -> ``b1k t45``).

Pass-dependent metrics get two lines per group, ``forward`` (= mode B) and ``forward+backward`` (mode A): relation
accuracy, undecided rate, accuracy per relation kind, subtask boundary error (intent / contact, mean |frames|),
re-observation agreement. Per-episode metrics (one value per episode): name consistency, goal satisfied / steps confirmed,
init failed / re-queried, cut / empty replies, agent fallbacks, model calls, tokens, cost. Numbers are the mean over
episodes with [min, max]; counts are sums. Cost = prompt tokens x ``price_in`` + completion tokens (reasoning included)
x ``price_out`` per 1M (default Qwen/Qwen3.8-27B 0.45 / 3 $).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PASSES = ("forward", "forward+backward")
PRICE = (0.45, 3.0)
USABLE = ("ok", "skipped")  # batch_status.json states whose run folder is aggregated  # $ per 1M prompt / completion tokens (Qwen/Qwen3.8-27B)


def group_of(name: str) -> str:
    m = re.match(r"(libero_mem|b1k)_t(\d+)_", name)
    return f"{m.group(1)} t{int(m.group(2)):02d}" if m else "other"


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def episode_record(d: Path, price: tuple[float, float] = PRICE) -> dict[str, Any]:
    """One episode's numbers (missing files -> the fields are None and ``missing`` lists them)."""
    ev, sm, be = _load(d / "eval.json"), _load(d / "summary.json"), _load(d / "batch_episode.json")
    rec: dict[str, Any] = {"id": d.name, "group": (be or {}).get("group") or group_of(d.name),
                           "missing": [f for f, x in (("eval.json", ev), ("summary.json", sm)) if x is None]}
    ev, sm = ev or {}, sm or {}
    for p in PASSES:
        m = ev.get(p) or {}
        r, b, o = m.get("relations") or {}, m.get("subtask_boundary") or {}, m.get("reobservation") or {}
        rec[p] = {"acc": r.get("accuracy"), "acc_remap": (m.get("relations_remap") or {}).get("accuracy"),
                  "decided": r.get("decided"), "undecided_rate": r.get("undecided_rate"),
                  "unmappable_wrong": r.get("unmappable_wrong"),
                  "per_pred": {k: v.get("accuracy") for k, v in (r.get("per_pred") or {}).items()},
                  "per_pred_counts": {k: [v.get("right", 0), v.get("wrong", 0), v.get("undecided", 0)]
                                      for k, v in (r.get("per_pred") or {}).items()},
                  "boundary_intent": (b.get("intent") or {}).get("mean"), "boundary_contact": (b.get("contact") or {}).get("mean"),
                  "reobs": o.get("rate"), "reobs_n": o.get("n")}
    nc, gc = ev.get("name_consistency") or {}, ev.get("goal_consistency") or {}
    goal = sm.get("goal") or {}
    best = goal.get("best")
    br, done = goal.get("branches") or [], goal.get("done") or []
    init = sm.get("init") or {}
    models = sm.get("models") or {}
    usage = {k: sum(int(u.get(k) or 0) for u in models.values()) for k in
             ("calls", "failures", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cut", "empty", "requeries",
              "incomplete", "images")}
    usage["seconds"] = round(sum(float(u.get("seconds") or 0) for u in models.values()), 1)
    rec.update({
        "name_consistency": nc.get("rate"), "map_changes": nc.get("map_changes"),
        "goal_known": bool(br), "goal_satisfied": gc.get("final_graph_satisfies_goal", goal.get("final_satisfied")),
        "oracle_success": gc.get("oracle_success"),
        "steps_done": done[best] if best is not None and best < len(done) else None,
        "steps_total": len(br[best]) if best is not None and best < len(br) else None,
        "region_missing": bool(goal.get("region_missing") or gc.get("region_goal_missing")),
        # goal steps that could not be asked until the end (object still in the hand / never detected); the goal
        # decision is unchanged, this separates "not observable" from "observed wrong"
        "goal_unobservable": len(goal.get("unobservable") or []),
        "merges_backward": (sm.get("track_identity") or {}).get("merges_backward"),
        "init_failed": sm.get("init_failed"), "init_attempts": len(init.get("attempts") or []) or None,
        "agent_fallbacks": sm.get("agent_fallbacks"), "frames": sm.get("n_frames"), "rows": sm.get("rows"),
        "usage": usage, "calls_per_role": {k: u.get("calls") for k, u in models.items()},
        "models": sorted({str(u.get("model")) for u in models.values() if u.get("model")}),
        "cost_usd": round((usage["prompt_tokens"] * price[0] + usage["completion_tokens"] * price[1]) / 1e6, 4),
    })
    return rec


def _stat(vals: Sequence[Any]) -> dict[str, Any] | None:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None
    return {"mean": round(statistics.mean(v), 4), "min": round(min(v), 4), "max": round(max(v), 4), "n": len(v)}


def summarize(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in recs if not r["missing"]]
    out: dict[str, Any] = {"episodes": len(recs), "evaluated": len(ok), "ids": [r["id"] for r in recs]}
    for p in PASSES:
        preds = sorted({k for r in ok for k in r[p]["per_pred"]})
        pooled = {}
        for k in preds:
            c = [sum(r[p]["per_pred_counts"].get(k, [0, 0, 0])[i] for r in ok) for i in range(3)]
            pooled[k] = {"right": c[0], "wrong": c[1], "undecided": c[2],
                         "accuracy": round(c[0] / (c[0] + c[1]), 4) if c[0] + c[1] else None}
        out[p] = {"acc": _stat([r[p]["acc"] for r in ok]), "acc_remap": _stat([r[p]["acc_remap"] for r in ok]),
                  "undecided_rate": _stat([r[p]["undecided_rate"] for r in ok]),
                  "decided": sum(r[p]["decided"] or 0 for r in ok),
                  "per_pred": {k: _stat([r[p]["per_pred"].get(k) for r in ok]) for k in preds},
                  "per_pred_pooled": pooled,
                  "boundary_intent": _stat([r[p]["boundary_intent"] for r in ok]),
                  "boundary_contact": _stat([r[p]["boundary_contact"] for r in ok]),
                  "reobs": _stat([r[p]["reobs"] for r in ok]), "reobs_n": sum(r[p]["reobs_n"] or 0 for r in ok)}
    u = {k: sum(r["usage"][k] for r in recs) for k in recs[0]["usage"]} if recs else {}
    out.update({
        "name_consistency": _stat([r["name_consistency"] for r in ok]),
        "goal_satisfied": sum(bool(r["goal_satisfied"]) for r in ok),
        "oracle_success": sum(bool(r["oracle_success"]) for r in ok),
        "steps_done": sum(r["steps_done"] or 0 for r in ok), "steps_total": sum(r["steps_total"] or 0 for r in ok),
        "region_missing": [r["id"] for r in recs if r["region_missing"]],
        "goal_unobservable_steps": sum(r.get("goal_unobservable") or 0 for r in ok),
        "goal_unobservable": [r["id"] for r in ok if r.get("goal_unobservable")],
        # goal not satisfied while a remaining step was not observable: "could not see", not "saw wrong"
        "goal_unsatisfied_unobservable": sum(1 for r in ok if not r["goal_satisfied"] and r.get("goal_unobservable")),
        "init_failed": [r["id"] for r in recs if r["init_failed"]],
        "init_requeried": sum(max((r["init_attempts"] or 1) - 1, 0) for r in recs),
        "agent_fallbacks": sum(r["agent_fallbacks"] or 0 for r in recs),
        "usage": u, "calls": _stat([r["usage"]["calls"] for r in recs]),
        "cost_usd": round(sum(r["cost_usd"] for r in recs), 3), "cost_per_episode": _stat([r["cost_usd"] for r in recs]),
    })
    return out


def aggregate(runs_dir: str | Path, price: tuple[float, float] = PRICE) -> dict[str, Any]:
    runs = Path(runs_dir)
    dirs = sorted(d for d in runs.iterdir() if d.is_dir() and ((d / "summary.json").exists() or (d / "eval.json").exists()
                                                             or (d / "batch_episode.json").exists()))
    status = _load(runs / "batch_status.json")
    # with a batch_status.json only episodes whose last batch step succeeded are aggregated: an output folder of a
    # failed (re)run / re-evaluation / not-run episode is listed as failed / excluded, never averaged
    excluded = {}
    if status is not None:
        excluded = {d.name: status.get(d.name, {"status": "not in batch_status.json"}) for d in dirs
                    if status.get(d.name, {}).get("status") not in USABLE}
        dirs = [d for d in dirs if d.name not in excluded]
    recs = [episode_record(d, price) for d in dirs]
    status = status or {}
    failed = {k: v for k, v in status.items() if v.get("status") not in USABLE} | excluded
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        groups.setdefault(r["group"], []).append(r)
    bench: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        bench.setdefault(r["group"].split(" ")[0], []).append(r)
    return {"runs_dir": str(runs), "price_per_1m": {"prompt": price[0], "completion": price[1]},
            "episodes": recs, "failed": failed,
            "groups": {g: summarize(v) for g, v in sorted(groups.items())},
            "benchmarks": {b: summarize(v) for b, v in sorted(bench.items())},
            "all": summarize(recs)}


# --- markdown -----------------------------------------------------------------------------------------

def _f(s: dict[str, Any] | None, nd: int = 3) -> str:
    if not s:
        return "-"
    if s["n"] == 1 or s["min"] == s["max"]:
        return f"{s['mean']:.{nd}f}"
    return f"{s['mean']:.{nd}f} [{s['min']:.{nd}f}, {s['max']:.{nd}f}]"


def to_markdown(res: dict[str, Any]) -> str:
    rows = [("all", res["all"])] + [(f"**{b}**", s) for b, s in res["benchmarks"].items()] + list(res["groups"].items())
    L = [f"# bggen batch — {res['runs_dir']}", "",
         "Mean over episodes [min, max]. Relation accuracy = episode-level one-to-one map (eval.md). Boundary = mean |frames| "
         "(intent: segment starts, contact: oracle inhand changes). Re-obs = a remembered value agrees when seen again.", "",
         "| group | pass | episodes | rel. accuracy | undecided rate | decided | boundary intent | boundary contact | re-obs (n) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        for p in PASSES:
            m = s.get(p) or {}
            L.append(f"| {name} | {'fwd' if p == 'forward' else 'fwd+bwd'} | {s['evaluated']}/{s['episodes']} | {_f(m.get('acc'))} | "
                     f"{_f(m.get('undecided_rate'))} | {m.get('decided')} | {_f(m.get('boundary_intent'), 1)} | "
                     f"{_f(m.get('boundary_contact'), 1)} | {_f(m.get('reobs'), 2)} ({m.get('reobs_n')}) |")
    L += ["", "## Accuracy per relation kind (mean over episodes [min, max]; pooled right / wrong / undecided)", ""]
    for name, s in rows:
        for p in PASSES:
            m = s.get(p) or {}
            if not m.get("per_pred"):
                continue
            cells = [f"{k} {_f(v, 2)} ({m['per_pred_pooled'][k]['right']}/{m['per_pred_pooled'][k]['wrong']}/"
                     f"{m['per_pred_pooled'][k]['undecided']})" for k, v in m["per_pred"].items()]
            L.append(f"- {name} {'fwd' if p == 'forward' else 'fwd+bwd'}: " + "; ".join(cells))
    L += ["", "## Per episode group: names, goal, init, replies, cost", "",
          "| group | name consistency | goal satisfied | oracle success | steps confirmed | init failed | init re-queried | "
          "agent fallbacks | cut / empty / requery / incomplete | calls / episode | prompt / completion tok | cost $ (per episode) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        u = s.get("usage") or {}
        L.append(f"| {name} | {_f(s['name_consistency'])} | {s['goal_satisfied']}/{s['evaluated']} | {s['oracle_success']}/{s['evaluated']} | "
                 f"{s['steps_done']}/{s['steps_total']} | {len(s['init_failed'])} | {s['init_requeried']} | {s['agent_fallbacks']} | "
                 f"{u.get('cut')} / {u.get('empty')} / {u.get('requeries')} / {u.get('incomplete')} | {_f(s['calls'], 1)} | "
                 f"{u.get('prompt_tokens')} / {u.get('completion_tokens')} | {s['cost_usd']} ({_f(s['cost_per_episode'], 3)}) |")
    L += ["", "## Episodes", "",
          "| episode | rel. acc fwd / fwd+bwd | undecided fwd | names | goal (steps) | oracle success | init | calls | $ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["episodes"]:
        fw, fb = r["forward"], r["forward+backward"]
        init = "FAILED" if r["init_failed"] else f"ok ({r['init_attempts']} attempt(s))" if r["init_attempts"] else "-"
        flags = (" REGION MISSING" if r["region_missing"] else "") + (f" missing {r['missing']}" if r["missing"] else "")
        unobs = f", {r['goal_unobservable']} unobservable" if r.get("goal_unobservable") else ""
        L.append(f"| {r['id']}{flags} | {fw['acc']} / {fb['acc']} | {fw['undecided_rate']} | {r['name_consistency']} | "
                 f"{r['goal_satisfied']} ({r['steps_done']}/{r['steps_total']}{unobs}) | {r['oracle_success']} | {init} | "
                 f"{r['usage']['calls']} | {r['cost_usd']} |")
    a = res["all"]
    L += ["", f"Init failed: {a['init_failed'] or 'none'}", f"Region (at center) missing from the goal: {a['region_missing'] or 'none'}",
          f"Goal steps unobservable until the end (object still held / never detected; goal decision unchanged): "
          f"{a['goal_unobservable_steps']} steps in {a['goal_unobservable'] or 'none'}; goal not satisfied with an "
          f"unobservable step: {a['goal_unsatisfied_unobservable']}",
          "Failed / excluded (batch_status.json, not averaged): " + (", ".join(f"{k} ({v.get('status')}: {v.get('step') or v.get('reason')})"
                                                               for k, v in res["failed"].items()) or "none"),
          f"Prices: {res['price_per_1m']} $ per 1M tokens; total ${a['cost_usd']} over {a['usage'].get('calls')} calls"]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aggregate")
    ap.add_argument("runs_dir")
    ap.add_argument("--out", default=None, help="output prefix (default RUNS_DIR/aggregate -> .md / .json)")
    ap.add_argument("--price-in", type=float, default=PRICE[0])
    ap.add_argument("--price-out", type=float, default=PRICE[1])
    a = ap.parse_args(argv)
    res = aggregate(a.runs_dir, (a.price_in, a.price_out))
    prefix = Path(a.out) if a.out else Path(a.runs_dir) / "aggregate"
    prefix.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    md = to_markdown(res)
    prefix.with_suffix(".md").write_text(md)
    print(md)
    return 0
