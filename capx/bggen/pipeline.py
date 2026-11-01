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
from pathlib import Path
from typing import Any

from .backward import backward, check_forbidden
from .configs import RENAMES
from .generator import GenConfig, Generator, save_trace
from .records import load_episode
from .serialize import cot_rows, row_to_json, write_jsonl


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
    final = fwd
    if cfg.offline:
        cats = {a: t.category for a, t in gen.tracker.tracks.items()}
        for m in gen.tracker.merges:
            if m.get("from"):
                cats.setdefault(m["from"], cats.get(m["into"], ""))
        res = backward(fwd, gen.tracker.merges, cats, base_motion(gen.events), cfg.navigation_subtask,
                       bool(summary["goal"]["final_satisfied"]))
        final = res.rows
        violations = check_forbidden(fwd, final)
        write_jsonl(out / "rows_backward.jsonl", [row_to_json(r) for r in final])
        (out / "backward_log.json").write_text(json.dumps({"log": res.log, "forbidden_violations": violations,
                                                           "segments": [s.__dict__ | {"effect": [[str(k), a, b] for k, a, b in s.effect]}
                                                                        for s in res.segments]}, indent=1, default=str))
        summary["backward"] = {"corrections": sum(1 for x in res.log if x["rule"] == "B2"),
                               "unified": sum(1 for x in res.log if x["rule"] == "B1"),
                               "segments": len(res.segments), "effects": sum(1 for s in res.segments if s.effect),
                               "forbidden_violations": violations}
    write_jsonl(out / "cot_rows.jsonl", cot_rows(final, ep.fps, rename, offline=cfg.offline, max_items=cfg.max_belief_items,
                                                 goal_keys=goal_keys, n_frames=ep.n_frames))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "report.md").write_text(report(summary, final, rename))
    return summary


def report(s: dict[str, Any], rows: list[Any], rename: dict[str, str]) -> str:
    from .serialize import row_fields, text

    m = s.get("models", {})
    lines = [f"# BG generator report — {s.get('benchmark')} (mode {s.get('mode')})", "",
             f"Instruction: {s.get('instruction')}", f"Frames: {s.get('n_frames')} at {s.get('fps')} fps; rows: {s['rows']}", "",
             "## Events", ", ".join(f"{k} {v}" for k, v in sorted(s["events"].items())), "",
             "## Model calls", "prompt tok = as reported by the server (real APIs include images; the FAKE server counts text only)",
             "| role | model | calls | failures | images sent | prompt tok | completion tok | s |", "|---|---|---|---|---|---|---|---|"]
    for r, u in m.items():
        lines.append(f"| {r} | {u.get('model')} | {u.get('calls')} | {u.get('failures')} | {u.get('images')} | {u.get('prompt_tokens')} | "
                     f"{u.get('completion_tokens')} | {u.get('seconds')} |")
    lines += ["", f"Agent format failures -> fallback plan: {s['agent_fallbacks']} {s.get('agent_fallback_reasons', {})}",
              f"Provisional grasps (close events): {s.get('grasp_checks', [])}",
              f"Detector: {s['detector']}; re-detections: {len(s['redetects'])}; track merges: {len(s['merges'])}", "",
              "## Verifier decisions by rule", ", ".join(f"{k} {v}" for k, v in sorted(s["verifier_rules"].items())), "",
              "## Goal", f"branches: {s['goal']['branches']}", f"done per branch: {s['goal']['done']} at frames {s['goal']['done_t']}",
              f"final graph satisfies the goal: {s['goal']['final_satisfied']}", ""]
    if "backward" in s:
        lines += ["## Backward pass", json.dumps(s["backward"]), ""]
    lines += ["## Labels (event rows)", ""]
    for r in rows:
        lines += [f"### t={r.t} {r.kind}{' ' + r.obj if r.obj else ''}", "```", text(row_fields(r, rename)), "```"]
    return "\n".join(lines) + "\n"
