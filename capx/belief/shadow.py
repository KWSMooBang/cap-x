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


# --- v2 (bg_est2) ------------------------------------------------------------------------------------

REGION_SIM = {"center": "kitchen_table_the_center"}  # region word -> LIBERO-Mem BDDL region (logs only)
PLACE_OPS = ("place", "put_in", "place_region")


class OracleShadowV2(OracleShadow):
    """``bg_est2`` shadow (EVALUATION LOGS ONLY): v1's metrics plus Observe accuracy (per predicate and
    answer, ? rate), declaration compliance (declared share, declared Effects vs truth), subtask
    judgement accuracy, and goal_match on goals normalized to newly-true predicates per step."""

    def __init__(self, source: GroundTruthSource) -> None:
        super().__init__(source)
        self.totals.update({
            "S3": [0, 0],
            "observe": {},  # pred -> {"1": [n, right], "0": [n, right], "?": n}
            "observe_fail": 0, "observe_calls": 0,
            "turns_with_code": 0, "declared": 0, "declared_valid": 0, "declared_dropped": 0,
            "effects_applied": 0, "effects_right": 0,
            "subtasks_judged": 0, "subtasks_right": 0,
        })

    def to_sim(self, pred: tuple, est: Any) -> tuple | None:
        if pred and pred[0] == "at":
            subj = self.t0_map.get(pred[1])
            return None if subj is None else ("on", subj, REGION_SIM.get(pred[2], pred[2]))
        if pred and pred[0] in ("handempty", "above", "lifted"):
            return (pred[0],) + tuple(pred[1:])
        return super().to_sim(pred, est)

    def truth(self, key: tuple | None) -> bool | None:
        if key is None:
            return None
        if key[0] == "handempty":
            try:
                return not any(self.source.grasped(n) for n in self.source.movable_objects())
            except Exception:
                return None
        if key[0] in ("above", "lifted"):
            return None
        return super().truth(key)

    def _oracle_init(self) -> dict[str, frozenset]:
        out: dict[str, set] = {}
        for e in self.oracle.graph.edges.values():
            if e.holds and e.predicate in ("on", "in"):
                out.setdefault(e.subject, set()).add((e.predicate, e.target))
        return {k: frozenset(v) for k, v in out.items()}

    def start(self, est: Any) -> dict[str, Any]:
        from .goal import normalize_steps

        log = super().start(est)
        oracle_norm = normalize_steps(self.oracle_branches, self._oracle_init())
        log["goal_match_normalized"] = None
        if est.goal.branches is not None:
            mapped = []
            for b in est.goal.branches:
                steps = []
                for s in b:
                    keys = [self.to_sim(p, est) for p in s]
                    steps.append(canonical_step([tuple(x for x in k if x is not None) for k in keys if k]))
                mapped.append(steps)
            log["goal_match_normalized"] = (sorted(repr(b) for b in mapped)
                                            == sorted(repr([canonical_step(s) for s in b]) for b in oracle_norm))
            log["est_goal_mapped"] = [[[" ".join(p) for p in s] for s in b] for b in mapped]
        log["oracle_goal_normalized"] = [[[" ".join(p) for p in s] for s in b] for b in oracle_norm]
        log["observe"] = self._observe_accuracy(est)
        return log

    def _observe_accuracy(self, est: Any) -> dict[str, Any]:
        res = getattr(est, "last_observe", None)
        t = self.totals
        if res is None:
            return {"asked": 0}
        t["observe_calls"] += 1
        if not res.ok:
            t["observe_fail"] += 1
            return {"asked": len(res.statements), "ok": False}
        items = []
        for key in res.statements:
            v = res.parse.values.get(key)
            truth = self.truth(self.to_sim(key, est))
            entry = t["observe"].setdefault(key[0], {"1": [0, 0], "0": [0, 0], "?": 0, "missing": 0})
            if v is None:
                entry["missing"] += 1
            elif v == "?":
                entry["?"] += 1
            elif truth is not None:
                entry[v][0] += 1
                entry[v][1] += int((v == "1") == truth)
            items.append({"key": " ".join(str(x) for x in key if x is not None), "answer": v, "oracle": truth})
        return {"asked": len(res.statements), "ok": True, "items": items}

    def compare(self, turn: int, est: Any, r: Any, fired: list[Any], prev_oracle: Progress | None,
                had_code: bool = True) -> dict[str, Any]:
        log = super().compare(turn, est, r, fired, prev_oracle)
        t = self.totals
        log["observe"] = self._observe_accuracy(est)
        d = getattr(est, "decl", None)
        decl_log: dict[str, Any] = {"declared": False}
        if had_code:
            t["turns_with_code"] += 1
        if d is not None and d.declared:
            t["declared"] += 1
            t["declared_valid"] += int(d.valid)
            t["declared_dropped"] += int(d.dropped is not None)
            effects = []
            if d.valid and d.applied:
                for e in d.effects:
                    truth = self.truth(self.to_sim(e.key, est))
                    if truth is None:
                        continue
                    t["effects_applied"] += 1
                    t["effects_right"] += int(truth == e.after)
                    effects.append({"effect": e.text, "oracle": truth, "right": truth == e.after})
            decl_log = {"declared": True, "valid": d.valid, "dropped": d.dropped, "effects": effects}
        log["declaration"] = decl_log
        subs = []
        for s in getattr(r, "judged", []):
            if s["status"] not in ("done", "failed"):
                continue
            o, args = s["op"], s["args"]
            if o in PLACE_OPS:
                pred = {"place": "on", "put_in": "in", "place_region": "at"}[o]
                key = self.to_sim((pred, args[0], args[1]), est)
            elif o == "pick" and est.robot.get("holding") == args[0]:
                key = self.to_sim(("inhand", args[0]), est)
            else:
                continue
            truth = self.truth(key)
            if truth is None:
                continue
            right = truth == (s["status"] == "done")
            t["subtasks_judged"] += 1
            t["subtasks_right"] += int(right)
            subs.append({"subtask": f"{o}({', '.join(args)})", "status": s["status"], "oracle": truth, "right": right})
        log["subtasks"] = subs
        return log
