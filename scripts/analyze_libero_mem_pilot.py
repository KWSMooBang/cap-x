"""Aggregate a LIBERO-Mem pilot run (scripts/run_ensemble_pilot.py --bench libero_mem, or any run directory with
task_XX_*/trial_*/trial_end.json) into results.tsv and SUMMARY.md, with cost by model.

    python scripts/analyze_libero_mem_pilot.py outputs/capbg_ensemble/<run> [--cond ensemble]

The repo copy of the LIBERO-Mem counterpart of scripts/analyze_robomme_pilot.py (the older analyze_pilot.py on the
cluster volume is not in the repo; it can import scripts/model_costs.py for the same cost columns).

Per trial (trial_end.json, all_responses.json): reason (finish | turn_limit | step_cap | wall_timeout | missing),
success (task_completed), turns, sim_steps, time_s, init_source, coding / VDM tokens (every ensemble candidate
and synthesis call included), cost_usd (catalogue prices, --prices), ensemble queries, failed candidates and
synthesis format errors. SUMMARY.md: success by task and init source, reasons, means, cost by model, ensemble.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_robomme_pilot import (  # noqa: E402
    DEFAULT_CODING_MODEL,
    DEFAULT_VDM_MODEL,
    _mean,
    cost_and_ensemble_sections,
    token_counts,
)
from model_costs import DEFAULT_PRICES, load_prices  # noqa: E402

TASK_DIR = re.compile(r"^task_(\d{2})_")
COLUMNS = ["cond", "task", "trial", "reason", "success", "turns", "sim_steps", "time_s", "init_source",
           "code_prompt_tok", "code_completion_tok", "vdm_prompt_tok", "vdm_completion_tok",
           "cost_usd", "ens_queries", "ens_failed_candidates", "synth_format_errors"]


def collect(run_dir: Path, cond: str, trials: int | None, coding_model: str, vdm_model: str,
            prices: dict[str, tuple[float, float]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    task_dirs = sorted(d for d in run_dir.rglob("task_*") if d.is_dir() and TASK_DIR.match(d.name)
                       and "_incomplete" not in d.parts)
    for task_dir in task_dirs:
        task = task_dir.name
        ends = sorted(task_dir.glob("trial_*/trial_end.json"), key=lambda p: p.stat().st_mtime)
        for end_path in ends:  # later files win (a resumed trial rewrites its directory)
            end = json.loads(end_path.read_text())
            resp_path = end_path.parent / "all_responses.json"
            responses = json.loads(resp_path.read_text()) if resp_path.exists() else []
            init = next((r.get("init_source") for r in responses if r.get("decision") == "initial"), None)
            rows[(task, int(end["trial"]))] = {
                "cond": cond, "task": task, "trial": int(end["trial"]), "reason": end.get("reason"),
                "success": int(bool(end.get("task_completed"))), "turns": end.get("turns"),
                "sim_steps": end.get("sim_steps"), "time_s": end.get("time_s"), "init_source": init,
                **token_counts(responses, coding_model, vdm_model, prices),
            }
        n = trials or max([k for (t, k) in rows if t == task] or [0])
        for k in range(1, n + 1):
            rows.setdefault((task, k), {"cond": cond, "task": task, "trial": k, "reason": "missing", "success": 0,
                                        "turns": None, "sim_steps": None, "time_s": None, "init_source": None,
                                        **dict.fromkeys(COLUMNS[9:], 0), "_by_model": {}, "_ensemble": {}})
    return sorted(rows.values(), key=lambda r: (r["task"], r["trial"]))


def summarize(rows: list[dict[str, Any]], prices: dict[str, tuple[float, float]]) -> str:
    done = [r for r in rows if r["reason"] != "missing"]
    lines = [f"# LIBERO-Mem pilot summary ({rows[0]['cond'] if rows else '?'})", "",
             f"Finished trials: {len(done)} / {len(rows)} (missing trials count as failures below).", ""]

    def table(groups: dict[str, list[dict[str, Any]]], title: str) -> None:
        lines.extend([f"## {title}", "",
                      "| group | success | rate | turns | sim_steps | time_s | code tok/trial | $/trial | reasons |",
                      "|---|---|---|---|---|---|---|---|---|"])
        for name, rs in groups.items():
            s = sum(r["success"] for r in rs)
            fin = [r for r in rs if r["reason"] != "missing"]
            reasons = ", ".join(f"{k} {v}" for k, v in Counter(r["reason"] for r in rs).most_common())
            code = [r["code_prompt_tok"] + r["code_completion_tok"] for r in fin]
            lines.append(
                f"| {name} | {s}/{len(rs)} | {s / len(rs):.0%} | {_mean([r['turns'] for r in fin])} | "
                f"{_mean([r['sim_steps'] for r in fin])} | {_mean([r['time_s'] for r in fin])} | {_mean(code)} | "
                f"{_mean([r['cost_usd'] for r in fin]) if fin else '-'} | {reasons} |"
            )
        lines.append("")

    by_task: dict[str, list] = defaultdict(list)
    by_init: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
        by_init[str(r["init_source"])].append(r)
    table(dict(by_task) | {"all": rows}, "By task")
    table(dict(by_init), "By init source")
    lines.extend(cost_and_ensemble_sections(done, prices))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cond", default=None, help="condition label (default: from pilot_manifest.jsonl)")
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
    cond = a.cond or ("ensemble" if manifest.get("use_parallel_ensemble") else manifest.get("memory_mode", "none"))
    prices = load_prices(a.prices)
    rows = collect(a.run_dir, cond, a.trials or manifest.get("trials"),
                   a.coding_model or manifest.get("model", DEFAULT_CODING_MODEL),
                   a.vdm_model or manifest.get("vdm_model", DEFAULT_VDM_MODEL), prices)
    if not rows:
        print(f"no LIBERO-Mem task directories under {a.run_dir}")
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
