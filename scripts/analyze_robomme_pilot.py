"""Aggregate a RoboMME pilot run (scripts/run_robomme_pilot.py) into results.tsv and SUMMARY.md.

    python scripts/analyze_robomme_pilot.py outputs/robomme_pilot/<run> [--cond none]

Per trial (from trial_end.json and all_responses.json): reason (success | fail | timeout | error | step_cap |
finish | turn_limit | wall_timeout | missing), success (RoboMME status == success), turns (code executions),
sim_steps (control steps), time_s (wall clock after reset), coding-model and VDM tokens.
RoboMME's own episode ends are success / fail / timeout / error; step_cap is the 1300-step budget, finish the
model's FINISH, turn_limit the multi-turn limit, wall_timeout the trial's wall-clock budget.

Writes <run>/results.tsv (columns: cond task trial reason success turns sim_steps time_s code_prompt_tok
code_completion_tok vdm_prompt_tok vdm_completion_tok cost_usd ens_queries ens_failed_candidates
synth_format_errors) and <run>/SUMMARY.md (per task and suite success rates, reason counts, mean turns / steps /
time, tokens, cost by model, ensemble failures and synthesis format errors), and prints the summary.

Coding tokens include every ensemble candidate and synthesis call when the ensemble is on. Cost uses the per-model
prices in --prices (default env_configs/pricing/nebius.yaml); the coding / VDM model names of non-ensemble calls
come from pilot_manifest.jsonl (``model`` / ``vdm_model``) or --coding-model / --vdm-model (scripts/model_costs.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_robomme_configs import TASKS  # noqa: E402
from model_costs import DEFAULT_PRICES, cost_by_model, cost_table, load_prices, merge_usage, trial_usage  # noqa: E402
from trial_end_fields import success_error_lines, success_of  # noqa: E402

TASK_BY_STEM = {f"robomme_{t.lower()}": t for t, _ in TASKS}
SUITE = dict(TASKS)
SUITES = list(dict.fromkeys(s for _, s in TASKS))
COLUMNS = ["cond", "task", "trial", "reason", "success", "turns", "sim_steps", "time_s",
           "code_prompt_tok", "code_completion_tok", "vdm_prompt_tok", "vdm_completion_tok",
           "cost_usd", "ens_queries", "ens_failed_candidates", "synth_format_errors"]
TOKEN_COLUMNS = COLUMNS[8:12]
DEFAULT_CODING_MODEL = "moonshotai/Kimi-K3"
DEFAULT_VDM_MODEL = "Qwen/Qwen3.8-27B"


def token_counts(
    responses: list[dict[str, Any]],
    coding_model: str = DEFAULT_CODING_MODEL,
    vdm_model: str = DEFAULT_VDM_MODEL,
    prices: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Coding (all ensemble calls included) and VDM tokens over every reply in the trial (retries of cut-off
    replies included), cost and ensemble counts; ``_by_model`` keeps the per-model usage for the summary."""
    u = trial_usage(responses, coding_model, vdm_model)
    code, vdm, ens = u["by_role"]["coding"], u["by_role"]["vdm"], u["ensemble"]
    cost = cost_by_model(u["by_model"], prices or {})
    return {
        "code_prompt_tok": int(code["prompt"]), "code_completion_tok": int(code["completion"]),
        "vdm_prompt_tok": int(vdm["prompt"]), "vdm_completion_tok": int(vdm["completion"]),
        "cost_usd": round(cost["total"], 4),
        "ens_queries": ens["ensemble_queries"], "ens_failed_candidates": ens["failed_candidates"],
        "synth_format_errors": ens["synthesis_format_errors"],
        "_by_model": u["by_model"], "_ensemble": ens,
    }


def collect(
    run_dir: Path, cond: str, trials: int | None, coding_model: str = DEFAULT_CODING_MODEL,
    vdm_model: str = DEFAULT_VDM_MODEL, prices: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    task_dirs = [d for d in run_dir.rglob("robomme_*") if d.is_dir() and d.name in TASK_BY_STEM]
    for task_dir in sorted(task_dirs):
        task = TASK_BY_STEM[task_dir.name]
        ends = sorted(task_dir.glob("trial_*/trial_end.json"), key=lambda p: p.stat().st_mtime)
        for end_path in ends:  # later files win (a resumed trial rewrites its directory)
            end = json.loads(end_path.read_text())
            resp_path = end_path.parent / "all_responses.json"
            responses = json.loads(resp_path.read_text()) if resp_path.exists() else []
            success, success_err = success_of(end, end_path)
            if success_err:
                print(f"WARNING: {success_err} (counted as failure)", file=sys.stderr)
            rows[(task, int(end["trial"]))] = {
                "cond": cond, "task": task, "trial": int(end["trial"]), "reason": end.get("reason"),
                "success": success, "success_error": success_err, "turns": end.get("turns"),
                "sim_steps": end.get("sim_steps"), "time_s": end.get("time_s"),
                **token_counts(responses, coding_model, vdm_model, prices),
            }
        n = trials or max([k for (t, k) in rows if t == task] or [0])
        for k in range(1, n + 1):
            rows.setdefault((task, k), {"cond": cond, "task": task, "trial": k, "reason": "missing", "success": 0,
                                        "turns": None, "sim_steps": None, "time_s": None,
                                        **dict.fromkeys(COLUMNS[8:], 0), "_by_model": {}, "_ensemble": {}})
    order = {t: i for i, (t, _) in enumerate(TASKS)}
    return sorted(rows.values(), key=lambda r: (order[r["task"]], r["trial"]))


def _mean(vals: list[Any]) -> str:
    v = [float(x) for x in vals if x is not None]
    return f"{statistics.mean(v):.1f}" if v else "-"


def summarize(rows: list[dict[str, Any]], prices: dict[str, tuple[float, float]] | None = None) -> str:
    done = [r for r in rows if r["reason"] != "missing"]
    lines = [f"# RoboMME pilot summary ({rows[0]['cond'] if rows else '?'})", "",
             f"Finished trials: {len(done)} / {len(rows)} (missing trials count as failures below).", "", *success_error_lines(rows)]

    def table(groups: dict[str, list[dict[str, Any]]], title: str) -> None:
        lines.extend([f"## {title}", "",
                      "| group | success | rate | turns | sim_steps | time_s | code tok/trial | VDM tok/trial | reasons |",
                      "|---|---|---|---|---|---|---|---|---|"])
        for name, rs in groups.items():
            s = sum(r["success"] for r in rs)
            reasons = ", ".join(f"{k} {v}" for k, v in Counter(r["reason"] for r in rs).most_common())
            fin = [r for r in rs if r["reason"] != "missing"]
            code = [r["code_prompt_tok"] + r["code_completion_tok"] for r in fin]
            vdm = [r["vdm_prompt_tok"] + r["vdm_completion_tok"] for r in fin]
            lines.append(
                f"| {name} | {s}/{len(rs)} | {s / len(rs):.0%} | {_mean([r['turns'] for r in fin])} | "
                f"{_mean([r['sim_steps'] for r in fin])} | {_mean([r['time_s'] for r in fin])} | "
                f"{_mean(code)} | {_mean(vdm)} | {reasons} |"
            )
        lines.append("")

    by_task: dict[str, list] = defaultdict(list)
    by_suite: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
        by_suite[SUITE[r["task"]]].append(r)
    table({s: by_suite[s] for s in SUITES if s in by_suite} | {"all": rows}, "By suite")
    table(dict(by_task), "By task")
    tot = Counter()
    for r in done:
        for k in TOKEN_COLUMNS:
            tot[k] += r[k]
    lines.extend(["## Tokens (all finished trials)", "",
                  f"coding: prompt {tot['code_prompt_tok']:,}, completion {tot['code_completion_tok']:,}; "
                  f"VDM: prompt {tot['vdm_prompt_tok']:,}, completion {tot['vdm_completion_tok']:,}", ""])
    lines.extend(cost_and_ensemble_sections(done, prices or {}))
    return "\n".join(lines)


def cost_and_ensemble_sections(done: list[dict[str, Any]], prices: dict[str, tuple[float, float]]) -> list[str]:
    """Markdown: cost by model (catalogue prices) and ensemble failure / synthesis format statistics."""
    lines = ["## Cost by model (catalogue prices, finished trials)", ""]
    lines += cost_table(merge_usage([r.get("_by_model") or {} for r in done]), prices, len(done))
    lines.append("")
    ens = [r.get("_ensemble") or {} for r in done]
    queries = sum(e.get("ensemble_queries", 0) for e in ens)
    if queries:
        cands = sum(e.get("candidate_calls", 0) for e in ens)
        failed = sum(e.get("failed_candidates", 0) for e in ens)
        synth = sum(e.get("synthesis_calls", 0) for e in ens)
        ferr = sum(e.get("synthesis_format_errors", 0) for e in ens)
        issues: Counter = Counter()
        for e in ens:
            issues.update(e.get("format_issues") or {})
        wall = sum(e.get("ensemble_wall_s", 0.0) for e in ens)
        cut = sum(e.get("cut_off_candidates", 0) for e in ens)
        synth_cut = sum(e.get("synthesis_cut_off", 0) for e in ens)
        synth_failed = sum(e.get("synthesis_failed", 0) for e in ens)
        lines += [
            "## Ensemble", "",
            f"- ensemble queries: {queries} ({queries / len(done):.1f} per trial), mean wall {wall / queries:.1f} s",
            f"- failed candidates: {failed} / {cands} ({failed / cands:.1%})" if cands else "- failed candidates: 0 / 0",
            f"- candidates cut off at max_tokens (still synthesized): {cut}; synthesis replies cut off: {synth_cut}; "
            f"synthesis calls failed (error): {synth_failed}",
            f"- synthesis format errors: {ferr} / {synth} ({ferr / synth:.1%})"
            + (f": {', '.join(f'{k} {v}' for k, v in issues.most_common())}" if issues else "") if synth else
            "- synthesis format errors: 0 / 0",
            "",
        ]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cond", default=None, help="condition label (default: memory_mode from pilot_manifest.jsonl)")
    ap.add_argument("--trials", type=int, default=None, help="expected trials per task (default: from manifest)")
    ap.add_argument("--out", type=Path, default=None, help="results TSV path (default: <run_dir>/results.tsv)")
    ap.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="pricing YAML ($ per 1M tokens per model)")
    ap.add_argument("--coding-model", default=None, help="model of non-ensemble coding calls (default: manifest)")
    ap.add_argument("--vdm-model", default=None, help="VDM model (default: manifest)")
    a = ap.parse_args()
    manifest: dict[str, Any] = {}
    mpath = a.run_dir / "pilot_manifest.jsonl"
    if mpath.exists():
        manifest = json.loads(mpath.read_text().strip().splitlines()[-1])
    cond = a.cond or manifest.get("memory_mode", "none")
    prices = load_prices(a.prices)
    rows = collect(a.run_dir, cond, a.trials or manifest.get("trials"),
                   a.coding_model or manifest.get("model", DEFAULT_CODING_MODEL),
                   a.vdm_model or manifest.get("vdm_model", DEFAULT_VDM_MODEL), prices)
    if not rows:
        print(f"no RoboMME task directories under {a.run_dir}")
        return 1
    out = a.out or a.run_dir / "results.tsv"
    with out.open("w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join("" if r[c] is None else str(r[c]) for c in COLUMNS) + "\n")
    summary = summarize(rows, prices)
    (a.run_dir / "SUMMARY.md").write_text(summary + "\n")
    print(summary)
    print(f"wrote {out} and {a.run_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
