"""Aggregate a RoboMME pilot run (scripts/run_robomme_pilot.py) into results.tsv and SUMMARY.md.

    python scripts/analyze_robomme_pilot.py outputs/robomme_pilot/<run> [--cond none]

Per trial (from trial_end.json and all_responses.json): reason (success | fail | timeout | error | step_cap |
finish | turn_limit | wall_timeout | missing), success (RoboMME status == success), turns (code executions),
sim_steps (control steps), time_s (wall clock after reset), coding-model and VDM tokens.
RoboMME's own episode ends are success / fail / timeout / error; step_cap is the 1300-step budget, finish the
model's FINISH, turn_limit the multi-turn limit, wall_timeout the trial's wall-clock budget.

Writes <run>/results.tsv (columns: cond task trial reason success turns sim_steps time_s code_prompt_tok
code_completion_tok vdm_prompt_tok vdm_completion_tok) and <run>/SUMMARY.md (per task and suite success rates,
reason counts, mean turns / steps / time, tokens), and prints the summary.
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

TASK_BY_STEM = {f"robomme_{t.lower()}": t for t, _ in TASKS}
SUITE = dict(TASKS)
SUITES = list(dict.fromkeys(s for _, s in TASKS))
COLUMNS = ["cond", "task", "trial", "reason", "success", "turns", "sim_steps", "time_s",
           "code_prompt_tok", "code_completion_tok", "vdm_prompt_tok", "vdm_completion_tok"]


def _tok(usage: Any, key: str) -> int:
    return int(usage.get(key) or 0) if isinstance(usage, dict) else 0


def token_counts(responses: list[dict[str, Any]]) -> dict[str, int]:
    """Coding-model and VDM tokens over every reply in the trial (retries of cut-off replies included)."""
    out = dict.fromkeys(("code_prompt_tok", "code_completion_tok", "vdm_prompt_tok", "vdm_completion_tok"), 0)

    def add_code(entry: dict[str, Any]) -> None:
        for u in (entry.get("usage"), entry.get("truncated_first_usage")):
            out["code_prompt_tok"] += _tok(u, "prompt_tokens")
            out["code_completion_tok"] += _tok(u, "completion_tokens")

    for r in responses:
        add_code(r)
        if isinstance(r.get("s5_first_reply"), dict):
            add_code(r["s5_first_reply"])
        for u in r.get("vdm_usage") or []:
            out["vdm_prompt_tok"] += _tok(u, "prompt_tokens")
            out["vdm_completion_tok"] += _tok(u, "completion_tokens")
    return out


def collect(run_dir: Path, cond: str, trials: int | None) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    task_dirs = [d for d in run_dir.rglob("robomme_*") if d.is_dir() and d.name in TASK_BY_STEM]
    for task_dir in sorted(task_dirs):
        task = TASK_BY_STEM[task_dir.name]
        ends = sorted(task_dir.glob("trial_*/trial_end.json"), key=lambda p: p.stat().st_mtime)
        for end_path in ends:  # later files win (a resumed trial rewrites its directory)
            end = json.loads(end_path.read_text())
            resp_path = end_path.parent / "all_responses.json"
            responses = json.loads(resp_path.read_text()) if resp_path.exists() else []
            rows[(task, int(end["trial"]))] = {
                "cond": cond, "task": task, "trial": int(end["trial"]), "reason": end.get("reason"),
                "success": int(bool(end.get("task_completed"))), "turns": end.get("turns"),
                "sim_steps": end.get("sim_steps"), "time_s": end.get("time_s"), **token_counts(responses),
            }
        n = trials or max([k for (t, k) in rows if t == task] or [0])
        for k in range(1, n + 1):
            rows.setdefault((task, k), {"cond": cond, "task": task, "trial": k, "reason": "missing", "success": 0,
                                        "turns": None, "sim_steps": None, "time_s": None,
                                        **dict.fromkeys(COLUMNS[8:], 0)})
    order = {t: i for i, (t, _) in enumerate(TASKS)}
    return sorted(rows.values(), key=lambda r: (order[r["task"]], r["trial"]))


def _mean(vals: list[Any]) -> str:
    v = [float(x) for x in vals if x is not None]
    return f"{statistics.mean(v):.1f}" if v else "-"


def summarize(rows: list[dict[str, Any]]) -> str:
    done = [r for r in rows if r["reason"] != "missing"]
    lines = [f"# RoboMME pilot summary ({rows[0]['cond'] if rows else '?'})", "",
             f"Finished trials: {len(done)} / {len(rows)} (missing trials count as failures below).", ""]

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
        for k in COLUMNS[8:]:
            tot[k] += r[k]
    lines.extend(["## Tokens (all finished trials)", "",
                  f"coding: prompt {tot['code_prompt_tok']:,}, completion {tot['code_completion_tok']:,}; "
                  f"VDM: prompt {tot['vdm_prompt_tok']:,}, completion {tot['vdm_completion_tok']:,}", ""])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cond", default=None, help="condition label (default: memory_mode from pilot_manifest.jsonl)")
    ap.add_argument("--trials", type=int, default=None, help="expected trials per task (default: from manifest)")
    ap.add_argument("--out", type=Path, default=None, help="results TSV path (default: <run_dir>/results.tsv)")
    a = ap.parse_args()
    manifest: dict[str, Any] = {}
    mpath = a.run_dir / "pilot_manifest.jsonl"
    if mpath.exists():
        manifest = json.loads(mpath.read_text().strip().splitlines()[-1])
    cond = a.cond or manifest.get("memory_mode", "none")
    rows = collect(a.run_dir, cond, a.trials or manifest.get("trials"))
    if not rows:
        print(f"no RoboMME task directories under {a.run_dir}")
        return 1
    out = a.out or a.run_dir / "results.tsv"
    with out.open("w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join("" if r[c] is None else str(r[c]) for c in COLUMNS) + "\n")
    summary = summarize(rows)
    (a.run_dir / "SUMMARY.md").write_text(summary + "\n")
    print(summary)
    print(f"wrote {out} and {a.run_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
