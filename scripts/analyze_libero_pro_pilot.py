"""Aggregate a LIBERO-PRO pilot run (scripts/run_ensemble_pilot.py --bench libero_pro) into results.tsv and
SUMMARY.md: success by suite in the layout of the CaP-X paper's Table 2 (Pos = *_swap, Task = *_task for
Object / Goal / Spatial), by suite, by task, reasons, cost by model and ensemble statistics.

    python scripts/analyze_libero_pro_pilot.py outputs/capbg_ensemble/<run> [more run dirs ...]

A run split over pods (same --run-name, different --tasks suites) is merged: every franka_libero_*_<task>/ directory
under the given run directories counts, and the tasks listed in every pilot_manifest.jsonl line that have no
directory yet show up as missing trials. Copy the run directories of pods without a shared volume next to each
other and pass them all; outputs go to the first one. Per-trial columns as in scripts/analyze_libero_mem_pilot.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_libero_mem_pilot import COLUMNS, collect, group_table  # noqa: E402
from analyze_robomme_pilot import DEFAULT_CODING_MODEL, DEFAULT_VDM_MODEL, cost_and_ensemble_sections  # noqa: E402
from model_costs import DEFAULT_PRICES, load_prices  # noqa: E402

TASK_DIR = re.compile(r"^franka_(libero_(?:object|goal|spatial)_(?:swap|task))_(\d+)$")
FAMILIES = ["object", "goal", "spatial"]
PERTURBATIONS = [("swap", "Pos"), ("task", "Task")]


def suite_of(task: str) -> str:
    m = TASK_DIR.match(task)
    return m.group(1) if m else "?"


def read_manifests(run_dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in run_dirs:
        path = d / "pilot_manifest.jsonl"
        if path.exists():
            out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def collect_runs(run_dirs: list[Path], cond: str, trials: int | None, coding_model: str, vdm_model: str,
                 prices: dict[str, tuple[float, float]], expected_tasks: list[str]) -> list[dict[str, Any]]:
    """Rows of all run directories; a (task, trial) finished in several of them keeps the last one given."""
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for i, d in enumerate(run_dirs):
        for r in collect(d, cond, trials, coding_model, vdm_model, prices, dir_glob="franka_libero_*",
                         dir_re=TASK_DIR, expected_tasks=expected_tasks if i == len(run_dirs) - 1 else ()):
            key = (r["task"], r["trial"])
            if key not in rows or r["reason"] != "missing":
                rows[key] = r
    return sorted(rows.values(), key=lambda r: (_task_order(r["task"]), r["trial"]))


def _task_order(task: str) -> tuple[int, int, int]:
    m = TASK_DIR.match(task)
    if not m:
        return (99, 99, 0)
    fam, pert = m.group(1).split("_")[1:]
    return (FAMILIES.index(fam), [p for p, _ in PERTURBATIONS].index(pert), int(m.group(2)))


def _rate(rs: list[dict[str, Any]]) -> str:
    if not rs:
        return "-"
    s = sum(r["success"] for r in rs)
    return f"{s / len(rs):.1%} ({s}/{len(rs)})"


def summarize(rows: list[dict[str, Any]], prices: dict[str, tuple[float, float]]) -> str:
    done = [r for r in rows if r["reason"] != "missing"]
    by_suite: dict[str, list] = defaultdict(list)
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_suite[suite_of(r["task"])].append(r)
        by_task[r["task"]].append(r)
    lines = [f"# LIBERO-PRO pilot summary ({rows[0]['cond'] if rows else '?'})", "",
             f"Finished trials: {len(done)} / {len(rows)} (missing trials count as failures below).", "",
             "## Table 2 layout (success rate; Pos = *_swap, Task = *_task)", "",
             "| | " + " | ".join(f"{f.capitalize()} {name}" for f in FAMILIES for _, name in PERTURBATIONS)
             + " | Pos | Task | All |",
             "|---|" + "---|" * (len(FAMILIES) * len(PERTURBATIONS) + 3)]
    cells = [_rate(by_suite.get(f"libero_{f}_{p}", [])) for f in FAMILIES for p, _ in PERTURBATIONS]
    pos = [r for r in rows if suite_of(r["task"]).endswith("_swap")]
    task = [r for r in rows if suite_of(r["task"]).endswith("_task")]
    lines += ["| success | " + " | ".join(cells) + f" | {_rate(pos)} | {_rate(task)} | {_rate(rows)} |", ""]
    group_table(lines, dict(sorted(by_suite.items(), key=lambda kv: _task_order(kv[1][0]["task"])))
                | {"all": rows}, "By suite")
    group_table(lines, dict(by_task), "By task")
    lines.extend(cost_and_ensemble_sections(done, prices))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--cond", default=None, help="condition label (default: from pilot_manifest.jsonl)")
    ap.add_argument("--trials", type=int, default=None, help="expected trials per task (default: from manifests)")
    ap.add_argument("--out", type=Path, default=None, help="results TSV path (default: <first run_dir>/results.tsv)")
    ap.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="pricing YAML ($ per 1M tokens per model)")
    ap.add_argument("--coding-model", default=None, help="model of non-ensemble coding calls (default: manifest)")
    ap.add_argument("--vdm-model", default=None, help="VDM model (default: manifest)")
    a = ap.parse_args()
    manifests = read_manifests(a.run_dirs)
    last = manifests[-1] if manifests else {}
    trials = a.trials or max([m.get("trials") or 0 for m in manifests] or [0]) or None
    expected = list(dict.fromkeys(t for m in manifests for t in m.get("tasks", []) if TASK_DIR.match(t)))
    cond = a.cond or ("ensemble" if last.get("use_parallel_ensemble") else last.get("memory_mode", "none"))
    prices = load_prices(a.prices)
    rows = collect_runs(a.run_dirs, cond, trials, a.coding_model or last.get("model", DEFAULT_CODING_MODEL),
                        a.vdm_model or last.get("vdm_model", DEFAULT_VDM_MODEL), prices, expected)
    if not rows:
        print(f"no LIBERO-PRO task directories under {', '.join(map(str, a.run_dirs))}")
        return 1
    out_dir = a.run_dirs[0]
    out = a.out or out_dir / "results.tsv"
    with out.open("w") as f:
        f.write("\t".join(["suite"] + COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join([suite_of(r["task"])] + ["" if r[c] is None else str(r[c]) for c in COLUMNS]) + "\n")
    summary = summarize(rows, prices)
    (out_dir / "SUMMARY.md").write_text(summary + "\n")
    print(summary)
    print(f"wrote {out} and {out_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
