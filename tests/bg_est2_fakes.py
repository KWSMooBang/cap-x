"""Fakes for bg_est2 tests: an Observe VDM that answers statements from the fake world's truth, a
basket world (cream cheese + 2 baskets, LIBERO-Mem task_08 / 09 layout) and a generic detector.

The fake VDM resolves aliases to world objects through the estimator's tracks (nearest object of the
same category), the way a real VDM resolves marks in the image; tests can make it answer ? / the
wrong value / add Notes / reply garbage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from capx.belief.perception import VOCABULARY, Detection
from tests.belief_fakes import FakeWorld

CAT = {c.key: c for c in VOCABULARY}


class AnyWorld(FakeWorld):
    """FakeWorld whose gripper grasps any non-support object (cream cheese, baskets, bottles ...)."""

    graspable = ("bowl", "cream_cheese", "basket", "bottle")
    prefer: str | None = None  # grasp this object when it is within reach (a basket by its rim)

    def close_gripper(self) -> None:
        self.steps += 30
        near = [(np.linalg.norm(o["pos"] - self.eef), n) for n, o in self.objects.items() if o["cat"] in self.graspable]
        near = [x for x in near if x[0] < 0.05]
        if near:
            self.held = self.prefer if self.prefer in [n for _, n in near] else min(near)[1]
            self.gripper = 0.3
        else:
            self.gripper = 0.0

    def open_gripper(self) -> None:
        self.steps += 30
        if self.held is not None:
            obj = self.objects[self.held]
            supports = [o for n, o in self.objects.items() if n != self.held and o["cat"] in ("plate", "basket")
                        and np.linalg.norm(o["pos"][:2] - self.eef[:2]) < 0.08 and o["pos"][2] < self.eef[2]]
            z = max(s["pos"][2] for s in supports) + 0.02 if supports else 0.0
            obj["pos"] = np.array([self.eef[0], self.eef[1], z])
        self.held = None
        self.gripper = 1.0


def basket_world() -> AnyWorld:
    """task_08 / 09: cream cheese on the table, two baskets (basket_1 nearer the robot's right)."""
    return AnyWorld({
        "cream_cheese_1": ("cream_cheese", (0.45, 0.05, 0.0)),
        "basket_1": ("basket", (0.55, -0.15, 0.0)), "basket_2": ("basket", (0.55, 0.20, 0.0)),
    })


class AnyDetector:
    """Category-level detections with the vocabulary's nominal sizes, center lifted by half the height."""

    name = "fake"

    def __init__(self, world: FakeWorld, seed: int = 0) -> None:
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.miss: set[str] = set()
        self.calls = 0

    def detect(self, obs, categories):
        self.calls += len(categories)
        keys = {c.key for c in categories}
        out = []
        for n, o in self.world.objects.items():
            if o["cat"] not in keys or n in self.miss:
                continue
            size = np.array(CAT[o["cat"]].size)
            out.append(Detection(o["cat"], o["pos"].copy(), size, 0.9))
        self.miss = set()
        return [out[i] for i in self.rng.permutation(len(out))]


def support_of(world: FakeWorld, name: str) -> str | None:
    o = world.objects[name]
    for m, q in world.objects.items():
        if m == name or q["cat"] not in ("plate", "basket"):
            continue
        if np.linalg.norm(q["pos"][:2] - o["pos"][:2]) < 0.08 and 0 < o["pos"][2] - q["pos"][2] < 0.05:
            return m
    return None


def world_truth(world: FakeWorld, key: tuple, name_of: Callable[[str], str | None]) -> bool | None:
    pred, s, t = key
    if pred == "handempty":
        return world.held is None
    sn = name_of(s)
    if sn is None:
        return None
    if pred == "inhand":
        return world.held == sn
    if pred in ("on", "in"):
        if world.held == sn:
            return False
        sup = support_of(world, sn)
        if t == "table":
            return sup is None and world.objects[sn]["pos"][2] < 0.01
        return sup is not None and sup == name_of(t)
    if pred == "at":
        return bool(np.linalg.norm(world.objects[sn]["pos"][:2] - np.array([0.5, 0.0])) < 0.1)
    return None


class FakeObserveVdm:
    """Answers the Statements of an M2 / M5 prompt from the world truth.

    ``override``: statement text -> forced answer (``"?"``, ``"0"``, ...) for the next call only, or
    every call if ``sticky``; ``notes``: Notes line for the next call; ``garbage``: next reply unreadable."""

    def __init__(self, world: FakeWorld, est_fn: Callable[[], Any]) -> None:
        self.world = world
        self.est_fn = est_fn
        self.prompts: list[Any] = []
        self.replies: list[str] = []
        self.override: dict[str, str] = {}
        self.sticky: dict[str, str] = {}
        self.notes: str | None = None
        self.garbage = False

    def name_of(self, alias: str) -> str | None:
        est = self.est_fn()
        t = est.tracks.get(alias) if est is not None else None
        if t is None:
            return None
        same = [n for n, o in self.world.objects.items() if o["cat"] == t.category.key]
        return min(same, key=lambda n: np.linalg.norm(self.world.objects[n]["pos"] - t.pos)) if same else None

    def __call__(self, prompt):
        self.prompts.append(prompt)
        text = prompt[-1]["content"][-1]["text"]
        stmts = re.findall(r"^\(.*\)$", text.split("Statements:")[1], re.M)
        from capx.belief.observe import statement_key

        parts = []
        for s in stmts:
            toks = s.strip("()").split()
            key = statement_key(toks[0], toks[1:])
            v = self.override.get(s, self.sticky.get(s))
            if v is None:
                truth = world_truth(self.world, key, self.name_of) if key else None
                v = "?" if truth is None else ("1" if truth else "0")
            parts.append(f"{s} {v}")
        reply = "Observe: " + " | ".join(parts)
        if self.notes:
            reply += "\nNotes: " + self.notes
        if self.garbage:
            reply = "I cannot comply."
        self.override, self.notes, self.garbage = {}, None, False
        self.replies.append(reply)
        return {"content": reply, "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def goal_json(*branches: list[list[tuple[str, str, str]]], repeat: int | None = None) -> str:
    bs = [{"steps": [{"preds": [list(p) for p in step], **({"repeat": repeat} if repeat else {})} for step in b]}
          for b in branches]
    return json.dumps({"goal": {"type": "or", "branches": bs}})


SWAP2_GOAL = goal_json(
    [[("on", "bowl_A", "plate_C")], [("on", "bowl_B", "plate_A")], [("on", "bowl_A", "plate_B")]],
    [[("on", "bowl_B", "plate_C")], [("on", "bowl_A", "plate_B")], [("on", "bowl_B", "plate_A")]],
)
# The same goal written as cumulative states (what v1's goal model did): normalized to SWAP2_GOAL.
SWAP2_GOAL_CUMULATIVE = goal_json(
    [[("on", "bowl_A", "plate_C")], [("on", "bowl_A", "plate_C"), ("on", "bowl_B", "plate_A")],
     [("on", "bowl_B", "plate_A"), ("on", "bowl_A", "plate_B")]],
    [[("on", "bowl_B", "plate_C")], [("on", "bowl_B", "plate_C"), ("on", "bowl_A", "plate_B")],
     [("on", "bowl_A", "plate_B"), ("on", "bowl_B", "plate_A")]],
)
REPEAT2_GOAL = goal_json([[("on", "bowl_A", "plate_A")]], repeat=3)
BASKET_GOAL = goal_json(
    [[("in", "cream_cheese_A", "basket_1")], [("on", "basket_1", "center")]],
    [[("in", "cream_cheese_A", "basket_2")], [("on", "basket_2", "center")]],
).replace("basket_1", "basket_A").replace("basket_2", "basket_B")


def declared(subtask: str, effect: str, body: str) -> str:
    return f"# Subtask: {subtask}\n# Effect: {effect}\n{body}"
