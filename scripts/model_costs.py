"""Token usage and USD cost of one trial, per model, from its all_responses.json.

Used by scripts/analyze_robomme_pilot.py and scripts/analyze_libero_mem_pilot.py (and importable from other
aggregators: ``from model_costs import trial_usage, load_prices``).

What is counted, per all_responses.json entry:
- Ensemble on (the entry has ``ensemble_calls``): every candidate call and every synthesis call listed there,
  under its own model. These include cut-off / empty retries and S5 re-asks, so the entry's ``usage`` /
  ``truncated_first_usage`` (the synthesis reply) are not counted again.
- Ensemble off: ``usage``, ``truncated_first_usage`` and ``s5_first_reply.usage`` under the coding model.
- ``vdm_usage`` under the VDM model (turn VDM calls and, for condition_video_mode vdm_text, the conditioning
  video description).
Cost = prompt tokens x input price + completion tokens x output price (reasoning tokens are part of the
completion tokens). Models without a price get cost None and are listed as unpriced.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "env_configs" / "pricing" / "nebius.yaml"


def load_prices(path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    """``{model: (input $/1M, output $/1M)}`` from a pricing YAML (env_configs/pricing/nebius.yaml)."""
    data = yaml.safe_load(Path(path or DEFAULT_PRICES).read_text())
    unit = float(data.get("unit_tokens", 1_000_000))
    scale = 1_000_000 / unit
    return {m: (float(p["input"]) * scale, float(p["output"]) * scale) for m, p in data["models"].items()}


def _tokens(usage: Any) -> tuple[int, int, int]:
    if not isinstance(usage, dict):
        return 0, 0, 0
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens")
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), int(reasoning or 0)


class _Acc:
    def __init__(self) -> None:
        def zero() -> dict[str, float]:
            return {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0, "latency_s": 0.0, "failed": 0}

        self.by_model: dict[str, dict[str, float]] = defaultdict(zero)
        self.by_role: dict[str, dict[str, float]] = {"coding": zero(), "vdm": zero()}

    def add(self, role: str, model: str, usage: Any, latency_s: float | None = None, failed: bool = False) -> None:
        p, c, r = _tokens(usage)
        for m in (self.by_model[model], self.by_role[role]):
            m["calls"] += 1
            m["prompt"] += p
            m["completion"] += c
            m["reasoning"] += r
            m["latency_s"] += float(latency_s or 0.0)
            m["failed"] += int(failed)


def trial_usage(responses: list[dict[str, Any]], coding_model: str, vdm_model: str) -> dict[str, Any]:
    """Per-model and per-role (coding / vdm) calls, tokens and latency, and ensemble statistics, for one trial."""
    acc = _Acc()
    ens = {"ensemble_queries": 0, "candidate_calls": 0, "failed_candidates": 0, "cut_off_candidates": 0,
           "synthesis_calls": 0, "synthesis_cut_off": 0,
           "synthesis_format_errors": 0, "format_issues": defaultdict(int), "ensemble_wall_s": 0.0}

    def add_single(entry: dict[str, Any]) -> None:
        for key in ("usage", "truncated_first_usage"):
            if entry.get(key) is not None:
                acc.add("coding", coding_model, entry.get(key))

    for r in responses:
        calls = r.get("ensemble_calls")
        if calls is not None:
            for call in calls:
                ens["ensemble_queries"] += 1
                ens["ensemble_wall_s"] += float(call.get("wall_s") or 0.0)
                for cand in call.get("candidates", []):
                    ens["candidate_calls"] += 1
                    ens["failed_candidates"] += int(not cand.get("ok", True))
                    ens["cut_off_candidates"] += int(cand.get("finish_reason") == "length")
                    acc.add("coding", cand["model"], cand.get("usage"), cand.get("latency_s"), failed=not cand.get("ok", True))
                synth = call.get("synthesis")
                if synth:
                    ens["synthesis_calls"] += 1
                    ens["synthesis_cut_off"] += int(synth.get("finish_reason") == "length")
                    acc.add("coding", synth["model"], synth.get("usage"), synth.get("latency_s"))
                    issue = synth.get("format_issue")
                    if issue:
                        ens["synthesis_format_errors"] += 1
                        ens["format_issues"][issue] += 1
        else:
            add_single(r)
            if isinstance(r.get("s5_first_reply"), dict):
                add_single(r["s5_first_reply"])
        for u in r.get("vdm_usage") or []:
            acc.add("vdm", vdm_model, u)
    ens["format_issues"] = dict(ens["format_issues"])
    return {"by_model": {m: dict(v) for m, v in acc.by_model.items()}, "by_role": acc.by_role, "ensemble": ens}


def cost_by_model(by_model: dict[str, dict[str, float]], prices: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """``{model: cost USD or None}`` plus ``total`` (priced models only) and ``unpriced`` model names."""
    out: dict[str, Any] = {}
    total = 0.0
    unpriced = []
    for model, t in by_model.items():
        if model not in prices:
            out[model] = None
            if t["prompt"] or t["completion"]:
                unpriced.append(model)
            continue
        pin, pout = prices[model]
        c = (t["prompt"] * pin + t["completion"] * pout) / 1_000_000
        out[model] = c
        total += c
    out["total"] = total
    out["unpriced"] = sorted(unpriced)
    return out


def merge_usage(items: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for by_model in items:
        for model, t in by_model.items():
            m = merged.setdefault(model, dict.fromkeys(t, 0))
            for k, v in t.items():
                m[k] = m.get(k, 0) + v
    return merged


def cost_table(by_model: dict[str, dict[str, float]], prices: dict[str, tuple[float, float]], n_trials: int) -> list[str]:
    """Markdown lines: one row per model (calls, tokens, mean latency, failed calls, cost) and the total."""
    costs = cost_by_model(by_model, prices)
    lines = ["| model | calls | failed | prompt tok | completion tok | reasoning tok | mean latency s | cost $ | $/trial |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model in sorted(by_model, key=lambda m: -(costs.get(m) or 0)):
        t = by_model[model]
        c = costs.get(model)
        lat = t["latency_s"] / t["calls"] if t["calls"] and t["latency_s"] else None
        lines.append(
            f"| {model} | {int(t['calls'])} | {int(t['failed'])} | {int(t['prompt']):,} | {int(t['completion']):,} | "
            f"{int(t['reasoning']):,} | {'-' if lat is None else f'{lat:.1f}'} | "
            f"{'unpriced' if c is None else f'{c:.2f}'} | "
            f"{'-' if c is None or not n_trials else f'{c / n_trials:.3f}'} |"
        )
    lines.append(f"| **total** |  |  |  |  |  |  | {costs['total']:.2f} | "
                 f"{costs['total'] / n_trials if n_trials else 0:.3f} |")
    if costs["unpriced"]:
        lines.append(f"\nUnpriced models (not in the pricing file): {', '.join(costs['unpriced'])}")
    return lines
