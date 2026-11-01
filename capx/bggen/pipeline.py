"""Run the generator on one prepared episode and write every output.

    <out>/trace.jsonl              per event: graph, agent prompt / raw reply / plan, tool results, verifier decisions
    <out>/rows_forward.jsonl       forward rows (= mode B), machine-readable
    <out>/rows_backward.jsonl      mode A rows after the backward pass
    <out>/cot_rows_forward.jsonl   Ver1 lines of the forward pass (events + 1 Hz)
    <out>/cot_rows.jsonl           Ver1 lines of the final labels (mode A: forward + backward; mode B: forward)
    <out>/backward_log.json        backward decisions (B1-B4) and the forbidden-list check
    <out>/keyframes/               marked images (PNG)
    <out>/summary.json, report.md  counts, model calls / tokens, goal, final graph
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .backward import backward, check_forbidden, sightings
from .configs import RENAMES
from .generator import GenConfig, Generator, save_trace
from .records import load_episode
from .serialize import cot_rows, row_to_json, write_jsonl
from .vocab import REGIONS


def base_motion(events: list[Any]) -> list[tuple[int, int]]:
    out, start = [], None
    for e in events:
        if e.kind == "base_start":
            start = e.t
        elif e.kind == "base_stop" and start is not None:
            out.append((start, e.t))
            start = None
    if start is not None:
        out.append((start, events[-1].t))
    return out


def run_episode(root: str | Path, out: str | Path, cfg: GenConfig, models: Any, detector: Any,
                benchmark: str = "libero_mem") -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ep = load_episode(root)
    gen = Generator(ep, detector, models, cfg, out)
    fwd = gen.run()
    rename = RENAMES.get(benchmark, {})
    goal_keys = gen.goal.all_keys()
    save_trace(out / "trace.jsonl", gen.trace)
    write_jsonl(out / "rows_forward.jsonl", [row_to_json(r) for r in fwd])
    write_jsonl(out / "cot_rows_forward.jsonl", cot_rows(fwd, ep.fps, rename, offline=False, max_items=cfg.max_belief_items,
                                                         goal_keys=goal_keys, n_frames=ep.n_frames))
    summary = gen.summary()
    summary.update(benchmark=benchmark, mode=cfg.mode, instruction=ep.instruction, n_frames=ep.n_frames, fps=ep.fps,
                   arms=list(gen.arms), init=gen.init_info)
    summary["goal"].update(region_check(ep.instruction, summary["goal"]["branches"], cfg.relations))
    final = fwd
    if cfg.offline:
        cats = {a: t.category for a, t in gen.tracker.tracks.items()}
        for m in gen.tracker.merges:
            if m.get("from"):
                cats.setdefault(m["from"], cats.get(m["into"], ""))
        res = backward(fwd, gen.tracker.merges, cats, base_motion(gen.events), cfg.navigation_subtask,
                       bool(summary["goal"]["final_satisfied"]), sight=sightings(gen.trace), caps=dict(gen.tracker.caps))
        final = res.rows
        violations = check_forbidden(fwd, final, renamed={m["from"]: m["into"] for m in res.merges_backward})
        write_jsonl(out / "rows_backward.jsonl", [row_to_json(r) for r in final])
        summary["track_identity"]["merges_backward"] = len(res.merges_backward)
        (out / "backward_log.json").write_text(json.dumps({"log": res.log, "forbidden_violations": violations,
                                                           # B1b alias map of rows_backward (evaluation renames the trace)
                                                           "merges_backward": res.merges_backward,
                                                           "segments": [s.__dict__ | {"effect": [[str(k), a, b] for k, a, b in s.effect]}
                                                                        for s in res.segments]}, indent=1, default=str))
        summary["backward"] = {"corrections": sum(1 for x in res.log if x["rule"] == "B2"),
                               "unified": sum(1 for x in res.log if x["rule"] == "B1"),
                               "merges_backward": [f"{m['from']} -> {m['into']} ({m['why']})" for m in res.merges_backward],
                               "segments": len(res.segments), "effects": sum(1 for s in res.segments if s.effect),
                               "forbidden_violations": violations}
    write_jsonl(out / "cot_rows.jsonl", cot_rows(final, ep.fps, rename, offline=cfg.offline, max_items=cfg.max_belief_items,
                                                 goal_keys=goal_keys, n_frames=ep.n_frames))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "report.md").write_text(report(summary, final, rename))
    return summary


def region_check(instruction: str, branches: list[Any], relations: tuple[str, ...]) -> dict[str, Any]:
    """The instruction names a table region (LIBERO-Mem 8 / 9 "... in the center"): is an ``at`` relation in the goal?
    ``region_missing`` True: the goal lost that step (no progress / goal satisfaction for it)."""
    words = [r for r in REGIONS if re.search(rf"\b{r}\b", (instruction or "").lower())]
    in_goal = sorted({st for br in branches for step in br for st in step if str(st).startswith("(at ")})
    return {"region_words": words, "region_relations": in_goal,
            "region_missing": bool(words) and bool(branches) and not in_goal, "at_allowed": "at" in relations}


def report(s: dict[str, Any], rows: list[Any], rename: dict[str, str]) -> str:
    from .serialize import row_fields, text

    m = s.get("models", {})
    lines = [f"# BG generator report — {s.get('benchmark')} (mode {s.get('mode')})", ""]
    if s.get("init_failed"):
        lines += ["> **!!! INIT FAILED — NO GOAL !!!** " + str(s["init_failed"]),
                  "> Goal progress, `done` and goal consistency of this run are meaningless; the initial statements",
                  "> came from the default list. See trace line 0 (`init.replies`, `init.attempts`: finish_reason).", ""]
    lines += [f"Instruction: {s.get('instruction')}", f"Frames: {s.get('n_frames')} at {s.get('fps')} fps; rows: {s['rows']}", "",
              "## Events", ", ".join(f"{k} {v}" for k, v in sorted(s["events"].items())), "",
              "## Model calls", "prompt tok = as reported by the server (real APIs include images; the FAKE server counts text only). "
              "cut = finish_reason length; requery = asked again with the larger limit; incomplete = still cut / empty.",
              "| role | model | calls | failures | images sent | prompt tok | completion tok | reasoning tok | cut | empty | requery | incomplete | s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, u in m.items():
        lines.append(f"| {r} | {u.get('model')} | {u.get('calls')} | {u.get('failures')} | {u.get('images')} | {u.get('prompt_tokens')} | "
                     f"{u.get('completion_tokens')} | {u.get('reasoning_tokens')} | {u.get('cut')} | {u.get('empty')} | "
                     f"{u.get('requeries')} | {u.get('incomplete')} | {u.get('seconds')} |")
    ti = s.get("track_identity", {})
    lines += ["", f"Agent format failures -> fallback plan: {s['agent_fallbacks']} {s.get('agent_fallback_reasons', {})}",
              f"Provisional grasps (close events): {s.get('grasp_checks', [])}",
              f"Detector: {s['detector']}; re-detections: {len(s['redetects'])}; track merges: {len(s['merges'])}", "",
              "## Track identity",
              f"splits (new track of an already-tracked category after t=0): {ti.get('new_after_init')}; "
              f"new categories after t=0: {ti.get('new_category_after_init')}; reacquired: {ti.get('reacquired')}; "
              f"merged at release place: {ti.get('merged_on_detect')} on detect + {ti.get('merged_late')} late; "
              f"bound to expected: {ti.get('bound_expected')}; closed-world assigned / dropped: "
              f"{ti.get('closed_assigned')} / {ti.get('closed_dropped')}; duplicate masks dropped: {ti.get('duplicates_dropped')}",
              f"tracks per category: {ti.get('tracks_per_category')}; expected never seen: {ti.get('expected_unbound')}", "",
              "## Verifier decisions by rule", ", ".join(f"{k} {v}" for k, v in sorted(s["verifier_rules"].items())), "",
              "## Goal"] + ([f"> **REGION MISSING FROM THE GOAL**: the instruction names {s['goal']['region_words']} but no "
                             "`at` relation is in the goal branches (that step is not tracked)."]
                            if s["goal"].get("region_missing") else []) + [f"branches: {s['goal']['branches']}", f"done per branch: {s['goal']['done']} at frames {s['goal']['done_t']}",
              f"final graph satisfies the goal: {s['goal']['final_satisfied']}",
              f"steps unobservable until the end: {s['goal'].get('unobservable') or 'none'}", ""]
    if "backward" in s:
        lines += ["## Backward pass", json.dumps(s["backward"]), ""]
    lines += ["## Labels (event rows)", ""]
    for r in rows:
        lines += [f"### t={r.t} {r.kind}{' ' + r.obj if r.obj else ''}", "```", text(row_fields(r, rename)), "```"]
    return "\n".join(lines) + "\n"
