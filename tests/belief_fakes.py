"""Fakes for the estimated Belief Graph (bg_est) tests: a tiny tabletop world with a gripper, a
detector over it with scripted misses and noise, and scripted LLM / VDM replies.

The world knows simulator-style names (akita_black_bowl_1 ...); the estimator only ever gets
category-level detections, the robot observation and hook records from it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from capx.belief.perception import Detection

BOWL_SIZE = (0.13, 0.13, 0.05)
PLATE_SIZE = (0.20, 0.20, 0.02)


class FakeWorld:
    """Objects on a table (z=0), a gripper that grasps the object within 5 cm of the EEF."""

    def __init__(self, objects: dict[str, tuple[str, tuple[float, float, float]]]) -> None:
        self.objects = {n: {"cat": c, "pos": np.array(p, float)} for n, (c, p) in objects.items()}
        self.eef = np.array([0.4, 0.0, 0.4])
        self.gripper = 1.0
        self.held: str | None = None
        self.steps = 0
        self.slip_next_lift = False  # the next lift after a grasp drops the object

    # --- API (what generated code calls) ---
    def close_gripper(self) -> None:
        self.steps += 30
        near = [(np.linalg.norm(o["pos"] - self.eef), n) for n, o in self.objects.items()
                if o["cat"] == "bowl"]
        near = [x for x in near if x[0] < 0.05]
        if near:
            self.held = min(near)[1]
            self.gripper = 0.3
        else:
            self.gripper = 0.0

    def open_gripper(self) -> None:
        self.steps += 30
        if self.held is not None:
            obj = self.objects[self.held]
            supports = [o for n, o in self.objects.items() if n != self.held
                        and np.linalg.norm(o["pos"][:2] - self.eef[:2]) < 0.08 and o["pos"][2] < self.eef[2]]
            z = max(s["pos"][2] for s in supports) + 0.02 if supports else 0.0
            obj["pos"] = np.array([self.eef[0], self.eef[1], z])
        self.held = None
        self.gripper = 1.0

    def goto_pose(self, position, quaternion_wxyz=None, z_approach: float = 0.0) -> None:
        self.steps += 50
        self.eef = np.asarray(position, float).copy()
        if self.held is not None:
            if self.slip_next_lift:
                self.slip_next_lift = False
                obj = self.objects[self.held]
                obj["pos"] = np.array([obj["pos"][0], obj["pos"][1], 0.0])
                self.held, self.gripper = None, 0.0
                return
            self.objects[self.held]["pos"] = self.eef + np.array([0.0, 0.0, -0.02])
        return None

    def functions(self) -> dict[str, Any]:
        return {"close_gripper": self.close_gripper, "open_gripper": self.open_gripper, "goto_pose": self.goto_pose}

    # --- robot observation (what get_observation() gives) ---
    def get_observation(self) -> dict[str, Any]:
        return {
            "robot_cartesian_pos": np.concatenate([self.eef, [0.0, 1.0, 0.0, 0.0], [self.gripper]]),
            "agentview": {
                "images": {"rgb": np.zeros((8, 8, 3), np.uint8), "depth": np.ones((8, 8), np.float32)},
                "intrinsics": np.array([[4.0, 0, 4.0], [0, 4.0, 4.0], [0, 0, 1]]),
                "pose_mat": np.array([[1.0, 0, 0, 0.5], [0, -1.0, 0, 0], [0, 0, -1.0, 1.5], [0, 0, 0, 1]]),
            },
        }

    def proprio(self):
        return self.eef.copy(), self.gripper

    def pick_place(self, name: str, target: str, lift: float = 0.15) -> None:
        """Code a model would write: grasp ``name`` and put it on ``target``."""
        p = self.objects[name]["pos"]
        self.api["goto_pose"](p + [0, 0, 0.02])
        self.api["close_gripper"]()
        self.api["goto_pose"](p + [0, 0, lift])
        t = self.objects[target]["pos"]
        self.api["goto_pose"](t + [0, 0, lift])
        self.api["goto_pose"](t + [0, 0, 0.04])
        self.api["open_gripper"]()
        self.api["goto_pose"](t + [0, 0, lift])


class FakeDetector:
    """Category-level detections of the world. ``miss`` (names) are not detected on the next call;
    ``noise`` is a per-axis std in m."""

    name = "fake"

    def __init__(self, world: FakeWorld, noise: float = 0.0, seed: int = 0) -> None:
        self.world = world
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.miss: set[str] = set()
        self.miss_always: set[str] = set()
        self.calls = 0

    def detect(self, obs, categories):
        self.calls += len(categories)
        keys = {c.key for c in categories}
        out = []
        for n, o in self.world.objects.items():
            if o["cat"] not in keys or n in self.miss or n in self.miss_always:
                continue
            size = BOWL_SIZE if o["cat"] == "bowl" else PLATE_SIZE
            out.append(Detection(o["cat"], o["pos"] + self.rng.normal(0, self.noise, 3) if self.noise else o["pos"].copy(),
                                 np.array(size), 0.9))
        self.miss = set()
        return [out[i] for i in self.rng.permutation(len(out))]


def swap_world() -> FakeWorld:
    """task_06 layout: plates at y=-0.2 / 0 / 0.2 (plate_A/B/C by y), bowls on the first two."""
    return FakeWorld({
        "plate_1": ("plate", (0.5, -0.2, 0.0)), "plate_2": ("plate", (0.5, 0.0, 0.0)),
        "plate_3": ("plate", (0.5, 0.2, 0.0)),
        "akita_black_bowl_1": ("bowl", (0.5, -0.2, 0.02)), "akita_black_bowl_2": ("bowl", (0.5, 0.0, 0.02)),
    })


def repeat_world() -> FakeWorld:
    """task_02 layout: one bowl on plate_1 (plate_A), plates_2/3 empty."""
    return FakeWorld({
        "plate_1": ("plate", (0.5, -0.2, 0.0)), "plate_2": ("plate", (0.5, 0.0, 0.0)),
        "plate_3": ("plate", (0.5, 0.2, 0.0)),
        "akita_black_bowl_1": ("bowl", (0.5, -0.2, 0.02)),
    })


SWAP_GOAL_JSON = json.dumps({"goal": {"type": "or", "branches": [
    {"steps": [{"preds": [["on", "bowl_A", "plate_C"]]}, {"preds": [["on", "bowl_B", "plate_A"]]},
               {"preds": [["on", "bowl_A", "plate_B"]]}]},
    {"steps": [{"preds": [["on", "bowl_B", "plate_C"]]}, {"preds": [["on", "bowl_A", "plate_B"]]},
               {"preds": [["on", "bowl_B", "plate_A"]]}]},
]}})
REPEAT_GOAL_JSON = json.dumps({"goal": {"type": "sequence", "steps": [{"preds": [["on", "bowl_A", "plate_A"]], "repeat": 3}]}})


class ScriptedModel:
    """Returns the scripted replies in order (the last one repeats) and records prompts."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[Any] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return {"content": text, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
