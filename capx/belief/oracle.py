"""Oracle Belief Graph estimator: ground truth read from a LIBERO-Mem simulator.

Everything here is privileged sim state (object poses, BDDL predicates, the subgoal tracker). It
is meant for the ``bg_oracle`` upper-bound condition, where the graph is put into the prompt on
purpose. It must never be exposed to the generated-code sandbox.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .graph import BeliefGraph, EdgeKey
from .progress import Pred, Progress, Step, _pred, _step, goal_branches

TABLE = "table"  # robosuite table body; contact with it stands in for On(x, table)


class GroundTruthSource(Protocol):
    """What the oracle needs from a simulator. ``LiberoMemSource`` implements it for LIBERO-Mem;
    tests use a fake."""

    def movable_objects(self) -> list[str]: ...

    def on_targets(self) -> list[str]: ...

    def contain_regions(self) -> list[str]: ...

    def goal_state(self) -> list[Any]: ...

    def positions(self) -> dict[str, tuple[float, float, float]]: ...

    def eval_predicate(self, pred: Pred) -> bool: ...

    def grasped(self, name: str) -> bool: ...

    def completed_steps(self) -> list[Step]: ...

    def overshot(self) -> bool: ...


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LiberoMemSource:
    """Reads ground truth from a ``FrankaLiberoMemEnv`` (``low_level_env``).

    Positions are in the robot base frame, the frame the generated code works in.
    """

    def __init__(self, low_level_env: Any) -> None:
        self.low = low_level_env

    @property
    def problem(self) -> Any:  # the BDDL problem env (robosuite env subclass)
        return self.low.handle.env.env

    def movable_objects(self) -> list[str]:
        return list(self.problem.objects_dict)

    def on_targets(self) -> list[str]:
        """Objects and fixtures, goal regions used with On (e.g. kitchen_table_the_center), and
        ``table`` (the table has no BDDL object state; see ``eval_predicate``)."""
        p = self.problem
        names = list(p.objects_dict) + list(p.fixtures_dict)
        for pred in _goal_preds(self.goal_state()):
            if pred[0] == "on" and len(pred) == 3 and pred[2] not in names:
                names.append(pred[2])
        names = [n for n in names if n in p.object_states_dict]
        if self._table_body() is not None:
            names.append(TABLE)
        return names

    def _table_body(self) -> int | None:
        try:
            return int(self.problem.sim.model.body_name2id(TABLE))
        except Exception:
            return None

    def _touches_table(self, name: str) -> bool:
        p = self.problem
        model, data = p.sim.model, p.sim.data
        table = self._table_body()
        geoms = {model.geom_name2id(g) for g in p.objects_dict[name].contact_geoms}
        for c in data.contact[: data.ncon]:
            pair = (c.geom1, c.geom2)
            for a, b in (pair, pair[::-1]):
                if a in geoms and model.geom_bodyid[b] == table:
                    return True
        return False

    def contain_regions(self) -> list[str]:
        p = self.problem
        names = [n for n in p.object_states_dict if n.endswith("_contain_region")]
        for pred in _goal_preds(self.goal_state()):
            if pred[0] == "in" and len(pred) == 3 and pred[2] not in names and pred[2] in p.object_states_dict:
                names.append(pred[2])
        return names

    def goal_state(self) -> list[Any]:
        return self.problem.parsed_problem["goal_state"]

    def positions(self) -> dict[str, tuple[float, float, float]]:
        p = self.problem
        sim = p.sim
        base_id = sim.model.body_name2id("robot0_base")
        rot = _quat_wxyz_to_mat(np.asarray(sim.data.xquat[base_id], dtype=np.float64))
        base = np.asarray(sim.data.xpos[base_id], dtype=np.float64)
        out = {}
        for name in p.objects_dict:
            body_id = p.obj_body_id[name]
            world = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
            out[name] = tuple(float(v) for v in rot.T @ (world - base))
        return out

    def eval_predicate(self, pred: Pred) -> bool:
        if pred[0] == "on" and len(pred) == 3 and pred[2] == TABLE:
            return self._touches_table(pred[1])
        return bool(self.problem._eval_predicate(list(pred)))

    def grasped(self, name: str) -> bool:
        p = self.problem
        return bool(p._check_grasp(gripper=p.robots[0].gripper, object_geoms=p.objects_dict[name]))

    def completed_steps(self) -> list[Step]:
        return [_step(s) for s in self.problem._satisfied_subgoals]

    def overshot(self) -> bool:
        return bool(getattr(self.problem, "_overshot", False))


def _goal_preds(goal_state: list[Any]) -> list[Pred]:
    seen: list[Pred] = []
    for branch in goal_branches(goal_state):
        for step in branch:
            for p in step:
                if p not in seen:
                    seen.append(p)
    return seen


class OracleEstimator:
    """Builds and updates a Belief Graph and task progress from a ground-truth source.

    All values are direct observations (source=obs, P=0.97/0.03).
    ``visibility="visible"`` (only what the cameras see) is reserved for a later variant.
    """

    def __init__(self, source: GroundTruthSource, visibility: str = "full") -> None:
        if visibility != "full":
            raise NotImplementedError("Only visibility='full' is implemented for the oracle estimator.")
        self.source = source
        self.visibility = visibility
        self.graph = BeliefGraph()
        self.branches = goal_branches(source.goal_state())
        self.candidates = self._candidates()
        self.goal_preds = _goal_preds(source.goal_state())
        self._broken: set[EdgeKey] = set()

    def _candidates(self) -> list[EdgeKey]:
        movable = self.source.movable_objects()
        on_targets = self.source.on_targets()
        regions = self.source.contain_regions()
        keys: list[EdgeKey] = []
        for s in movable:
            keys += [("on", s, t) for t in on_targets if t != s]
            keys += [("in", s, r) for r in regions if not r.startswith(f"{s}_")]
            keys.append(("inhand", s, None))
        for p in _goal_preds(self.source.goal_state()):
            key: EdgeKey = (p[0], p[1], p[2] if len(p) > 2 else None)
            if key not in keys:
                keys.append(key)
        return keys

    def _eval(self, key: EdgeKey) -> bool | None:
        if key in self._broken:
            return None
        try:
            if key[0] == "inhand":
                return self.source.grasped(key[1])
            return self.source.eval_predicate(_pred([key[0], key[1]] + ([key[2]] if key[2] else [])))
        except Exception:  # predicate undefined for this pair (e.g. no contain geometry)
            self._broken.add(key)
            return None

    def observe(self, turn: int) -> tuple[list, Progress]:
        """Read ground truth, update the graph, and return (new events, progress)."""
        obs = {}
        for key in self.candidates:
            v = self._eval(key)
            if v is not None:
                obs[key] = v
        events = self.graph.update(turn, obs, self.source.positions())
        progress = Progress.from_completed(self.branches, self.source.completed_steps(), self.source.overshot())
        return events, progress

    def current_truths(self) -> dict[Pred, bool]:
        """Truth of each goal predicate in the current graph (for the progress line)."""
        out = {}
        for p in self.goal_preds:
            edge = self.graph.edges.get((p[0], p[1], p[2] if len(p) > 2 else None))
            out[p] = bool(edge and edge.holds)
        return out
