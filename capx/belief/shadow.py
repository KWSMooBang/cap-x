"""Oracle shadow for ``bg_est``: the estimated graph compared with simulator ground truth.

EVALUATION LOGS ONLY. Nothing here goes into a prompt or back into the estimator (except the
``goal_source: bddl`` diagnostic, which is logged as such). Per turn it records:

* the alias -> simulator object mapping (per category, Hungarian on position) and identity swaps
  (an alias mapped to a different object than at turn 0 / the previous turn);
* accuracy of the goal predicates (estimate P > 0.5 vs the oracle graph, aliases translated with the
  turn-0 mapping, since the goal was parsed with turn-0 aliases);
* the step-count error (estimated done steps - LIBERO-Mem tracker's done steps);
* whether S1 / S4 / S5 that fired (or S5 that was suppressed) were right.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .goal import GoalResult, goal_from_bddl, goal_preds
from .oracle import GroundTruthSource, OracleEstimator
from .progress import Progress, canonical_step, format_pred, goal_branches

TABLE = "table"


class OracleShadow:
    def __init__(self, source: GroundTruthSource) -> None:
        self.source = source
        self.oracle = OracleEstimator(source)
        self.oracle_branches = goal_branches(source.goal_state())
        self.t0_map: dict[str, str] = {}
        self.prev_map: dict[str, str] = {}
        self.progress: Progress | None = None
        self.totals: dict[str, Any] = {
            "turns": 0, "identity_swap_events": 0, "aliases_swapped_at_end": 0,
            "goal_pred_correct": 0, "goal_pred_total": 0, "step_error_abs_sum": 0,
            "S1": [0, 0], "S2": [0, 0], "S4": [0, 0], "S5b": [0, 0], "S5a": [0, 0],  # [fired, right]
            "s5_suppressed_while_complete": 0,
        }

    # --- identities -----------------------------------------------------------------------------
    def mapping(self, est: Any) -> dict[str, str]:
        from scipy.optimize import linear_sum_assignment

        positions = self.source.positions()
        out: dict[str, str] = {}
        for cat in est.categories:
            sims = [n for n in self.source.movable_objects() if n in positions and any(k in n for k in cat.sim_keys)]
            tracks = [t for t in est.tracks.values() if t.category.key == cat.key]
            if not sims or not tracks:
                continue
            cost = np.array([[np.linalg.norm(t.pos - np.asarray(positions[s])) for s in sims] for t in tracks])
            for r, c in zip(*linear_sum_assignment(cost), strict=True):
                out[tracks[r].alias] = sims[c]
        return out

    def to_sim(self, pred: tuple, est: Any) -> tuple | None:
        """Alias predicate -> simulator edge key with the turn-0 mapping (None if unmapped)."""
        name, subj = pred[0], self.t0_map.get(pred[1])
        if subj is None:
            return None
        if name == "inhand":
            return ("inhand", subj, None)
        target = pred[2]
        if target == TABLE:
            return (name, subj, TABLE)
        sim_t = self.t0_map.get(target)
        if sim_t is None:
            return (name, subj, target)  # e.g. a region name: never in the oracle graph
        if name == "in":
            sim_t += "_contain_region"
        return (name, subj, sim_t)

    def truth(self, key: tuple | None) -> bool | None:
        if key is None:
            return None
        edge = self.oracle.graph.edges.get(key)
        return None if edge is None else edge.holds

    def bddl_goal(self, est: Any) -> GoalResult:
        """``goal_source: bddl`` (diagnostics): the BDDL goal with sim names mapped to turn-0 aliases."""
        inverse = {sim: alias for alias, sim in self.mapping(est).items()}
        return goal_from_bddl(self.source.goal_state(), inverse)

    # --- per turn -------------------------------------------------------------------------------
    def start(self, est: Any) -> dict[str, Any]:
        self.t0_map = self.mapping(est)
        self.prev_map = dict(self.t0_map)
        _, self.progress = self.oracle.observe(0)
        log = {"turn": 0, "mapping": dict(self.t0_map), "goal_match": None,
               "oracle_goal": [[[" ".join(p) for p in s] for s in b] for b in self.oracle_branches]}
        if est.goal.branches is not None:
            mapped = []
            for b in est.goal.branches:
                steps = []
                for s in b:
                    keys = [self.to_sim(p, est) for p in s]
                    steps.append(canonical_step([tuple(x for x in k if x is not None) for k in keys if k]))
                mapped.append(steps)
            norm = sorted(repr(b) for b in mapped)
            log["goal_match"] = norm == sorted(repr([canonical_step(s) for s in b]) for b in self.oracle_branches)
        log.update(self._accuracy(est))
        return log

    def _accuracy(self, est: Any) -> dict[str, Any]:
        correct, total, wrong = 0, 0, []
        for p in goal_preds(est.goal.branches or []):
            truth = self.truth(self.to_sim(p, est))
            if truth is None:
                continue
            belief = est.pred_belief(p)[0]
            total += 1
            if (belief > 0.5) == truth:
                correct += 1
            else:
                wrong.append({"pred": format_pred(p), "est_p": round(belief, 3), "oracle": truth})
        return {"goal_pred_correct": correct, "goal_pred_total": total, "goal_pred_wrong": wrong}

    def compare(self, turn: int, est: Any, r: Any, fired: list[Any], prev_oracle: Progress | None) -> dict[str, Any]:
        _, self.progress = self.oracle.observe(turn)
        mapping = self.mapping(est)
        swapped = sorted(a for a, s in mapping.items() if self.t0_map.get(a) not in (None, s))
        changed = sorted(a for a, s in mapping.items() if self.prev_map.get(a) not in (None, s))
        self.prev_map = mapping
        est_done, oracle_done = len(r.progress.done), len(self.progress.done)
        log: dict[str, Any] = {
            "turn": turn, "mapping": mapping, "aliases_swapped_vs_t0": swapped, "identity_swap_events": changed,
            "est_done": est_done, "oracle_done": oracle_done, "step_error": est_done - oracle_done,
            "oracle_complete": self.progress.complete, "oracle_overshot": self.progress.overshot,
            "est_suppress_completion": r.suppress_completion,
        }
        log.update(self._accuracy(est))
        sig_logs = []
        for sig in fired:
            entry: dict[str, Any] = {"id": sig.id}
            if sig.id == "S1":
                items = []
                for v in r.violations:
                    truth = self.truth(self.to_sim(v.key, est)) if v.key else None
                    items.append({"expected": v.expected, "oracle": truth,
                                  "right": None if truth is None else truth != v.holds})
                entry["items"] = items
                rights = [i["right"] for i in items if i["right"] is not None]
                entry["right"] = bool(rights) and all(rights)
            elif sig.id == "S4":
                items = []
                for u in r.uncertain:
                    truth = self.truth(self.to_sim(u.key, est)) if u.key else None
                    items.append({"pred": u.pred, "reason": u.reason, "est_p": round(u.p, 3), "oracle": truth,
                                  "est_wrong": None if truth is None else (u.p > 0.5) != truth})
                entry["items"] = items
                entry["right"] = any(i["est_wrong"] for i in items) or any(i["reason"] == "unconfirmed" for i in items)
            elif sig.id == "S5b":
                entry["right"] = self.progress.complete
            elif sig.id == "S2":
                entry["right"] = prev_oracle is None or len(self.progress.done) <= len(prev_oracle.done)
            sig_logs.append(entry)
        log["signals"] = sig_logs
        t = self.totals
        t["turns"] += 1
        t["identity_swap_events"] += len(changed)
        t["aliases_swapped_at_end"] = len(swapped)
        t["goal_pred_correct"] += log["goal_pred_correct"]
        t["goal_pred_total"] += log["goal_pred_total"]
        t["step_error_abs_sum"] += abs(log["step_error"])
        for e in sig_logs:
            if e["id"] in t:
                t[e["id"]][0] += 1
                t[e["id"]][1] += int(bool(e.get("right")))
        if r.suppress_completion and self.progress.complete:
            t["s5_suppressed_while_complete"] += 1
        return log

    def finish_check(self, rejected: bool) -> dict[str, Any]:
        complete = bool(self.progress and self.progress.complete)
        if rejected:
            self.totals["S5a"][0] += 1
            self.totals["S5a"][1] += int(not complete)
        return {"s5a_rejected": rejected, "oracle_complete": complete, "right": (not complete) if rejected else None}
