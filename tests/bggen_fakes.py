"""Synthetic fixtures for capx.bggen tests: a scripted "swap the 2 bowls" episode (the design's example
tab), written in the real on-disk episode format, a FAKE detector that reads the synthetic world, and a
FAKE VDM brain that answers from it. Ground truth goes to <root>/oracle/ (evaluation) and stays in this
module; the generator only gets the episode directory.

World (robot base frame, m): plates at x 0.6, y +0.2 / 0.0 / -0.2. The camera looks straight down, image
left = +y, so aliases from the image left are plate_A (y +0.2, empty), plate_B (y 0), plate_C (y -0.2);
bowl_A starts on plate_B, bowl_B on plate_C. Moves: bowl_A -> plate_A, bowl_B -> plate_B (released with a
6.6 cm offset that slides to the center within 15 frames: the rim case), bowl_A -> plate_C.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from capx.bggen.fake_models import FakeBrain
from capx.bggen.oracle import write_oracle
from capx.bggen.perception import Detection
from capx.bggen.records import write_episode, write_image

FPS = 20.0
W, H, F = 160, 120, 150.0
PLATES = {"plate_1": (0.6, 0.2), "plate_2": (0.6, 0.0), "plate_3": (0.6, -0.2)}  # sim name -> xy
ALIAS = {"plate_A": "plate_1", "plate_B": "plate_2", "plate_C": "plate_3", "bowl_A": "bowl_1", "bowl_B": "bowl_2"}
SIM_CAT = {"plate_1": "plate", "plate_2": "plate", "plate_3": "plate", "bowl_1": "bowl", "bowl_2": "bowl"}
PLATE_Z, BOWL_Z, HOME = 0.01, 0.038, np.array([0.45, 0.0, 0.30])  # dz 0.028: bg_est2's measured bowl/plate
OPEN_W, HELD_W = 0.08, 0.03
CAM_POS = np.array([0.6, 0.0, 1.0])
R_CAM = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]).T  # columns: camera axes in base


def T_base_cam() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_CAM, CAM_POS
    return T


def K() -> np.ndarray:
    return np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])


def project(p: np.ndarray) -> tuple[float, float]:
    c = R_CAM.T @ (np.asarray(p) - CAM_POS)
    return float(F * c[0] / c[2] + W / 2), float(F * c[1] / c[2] + H / 2)


@dataclass
class SwapWorld:
    eef: list[np.ndarray] = field(default_factory=list)
    width: list[float] = field(default_factory=list)
    bowls: list[dict[str, np.ndarray]] = field(default_factory=list)
    held: list[str | None] = field(default_factory=list)
    grasp_t: list[int] = field(default_factory=list)
    release_t: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.eef)

    def truth(self, t: int) -> dict[tuple, bool]:
        """(pred, sim_a, sim_b) -> truth for on(bowl, plate), on(bowl, table), inhand, handempty."""
        out: dict[tuple, bool] = {}
        for b, p in self.bowls[t].items():
            held = self.held[t] == b
            for pl, (x, y) in PLATES.items():
                out[("on", b, pl)] = (not held) and np.hypot(p[0] - x, p[1] - y) < 0.06 and abs(p[2] - BOWL_Z) < 0.01
            out[("on", b, "table")] = False
            out[("inhand", b, "main")] = held
        out[("handempty", "main", None)] = self.held[t] is None
        return out


def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    return [a + (b - a) * (i + 1) / n for i in range(n)]


def make_world(rim_slide: bool = True) -> SwapWorld:
    w = SwapWorld()
    bowls = {"bowl_1": np.array([0.6, 0.0, BOWL_Z]), "bowl_2": np.array([0.6, -0.2, BOWL_Z])}
    eef, width, held = HOME.copy(), OPEN_W, None

    def push(n: int = 1, slide: tuple[str, np.ndarray, np.ndarray, int] | None = None) -> None:
        for _ in range(n):
            if held is not None:
                bowls[held] = eef.copy()
            w.eef.append(eef.copy())
            w.width.append(width)
            w.bowls.append({k: v.copy() for k, v in bowls.items()})
            w.held.append(held)

    push(20)
    moves = [("bowl_1", (0.6, 0.2)), ("bowl_2", (0.6, 0.0)), ("bowl_1", (0.6, -0.2))]
    for k, (name, (tx, ty)) in enumerate(moves):
        b = bowls[name]
        for p in _lerp(eef, np.array([b[0], b[1], 0.20]), 40) + _lerp(np.array([b[0], b[1], 0.20]), b.copy(), 20):
            eef = p
            push()
        for v in np.linspace(OPEN_W, HELD_W, 10):  # close
            width = float(v)
            push()
        held = name
        w.grasp_t.append(len(w.eef) - 1)
        push(6)
        for p in _lerp(eef, eef + np.array([0, 0, 0.15]), 20) + _lerp(eef + np.array([0, 0, 0.15]), np.array([tx, ty, 0.20]), 40):
            eef = p
            push()
        offset = np.array([0.0, 0.066, 0.0]) if (k == 1 and rim_slide) else np.zeros(3)
        for p in _lerp(eef, np.array([tx, ty, BOWL_Z]) + offset, 20):
            eef = p
            push()
        held = None
        w.release_t.append(len(w.eef))
        for i, v in enumerate(np.linspace(HELD_W, OPEN_W, 10)):  # open
            width = float(v)
            push()
        if k == 1 and rim_slide:  # the bowl rests on the rim, then slides to the center
            for i in range(15):
                bowls[name] = np.array([tx, ty + 0.066 * (1 - (i + 1) / 15), BOWL_Z + 0.012 * (1 - (i + 1) / 15)])
                push()
        for p in _lerp(eef, eef + np.array([0, 0, 0.2]), 20):
            eef = p
            push()
    for p in _lerp(eef, HOME, 30):
        eef = p
        push()
    push(20)
    return w


def render(w: SwapWorld, t: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (150, 120, 90))
    d = ImageDraw.Draw(img)
    for pl, (x, y) in PLATES.items():
        u, v = project(np.array([x, y, PLATE_Z]))
        d.ellipse((u - 15, v - 15, u + 15, v + 15), fill=(240, 240, 240))
    for b, p in w.bowls[t].items():
        u, v = project(p)
        d.ellipse((u - 9, v - 9, u + 9, v + 9), fill=(20, 20, 20))
    u, v = project(w.eef[t])
    d.rectangle((u - 3, v - 3, u + 3, v + 3), fill=(200, 0, 0))
    arr = np.asarray(img).copy()
    arr[0, 0] = (t // 256, t % 256, 7)  # the frame index, read by the fake VDM
    return arr


def write_swap_episode(root: Path, w: SwapWorld | None = None, image_stride: int = 2) -> SwapWorld:
    w = w or make_world()
    n = w.n
    proprio = {"t": np.arange(n), "eef_pos_main": np.stack(w.eef), "gripper_main": np.array(w.width)}
    ts = list(range(0, n, image_stride))
    if ts[-1] != n - 1:
        ts.append(n - 1)
    meta = {"benchmark": "synthetic", "episode_id": "swap", "instruction": "swap the 2 bowls on their plates using the empty plate",
            "fps": FPS, "n_frames": n, "arms": ["main"], "cameras": {"agentview": {"depth": True, "width": W, "height": H}}}
    write_episode(root, meta, proprio, {"agentview": {"t": ts, "K": K(), "T": np.stack([T_base_cam()] * len(ts))}})
    depth = np.full((H, W), 0.9, np.float32)
    for t in ts:
        write_image(root, "agentview", t, render(w, t), depth)
    states = []
    for t in range(0, n, 2):
        tr = w.truth(t)
        pos = {**{k: [float(x) for x in v] for k, v in w.bowls[t].items()},
               **{k: [x, y, PLATE_Z] for k, (x, y) in PLATES.items()}}
        states.append({"t": t, "pos": pos, "true": [list(k) for k, v in tr.items() if v and k[0] == "on"],
                       "inhand": {"main": w.held[t]}})
    segs = []
    prev = 0
    for g, r in zip(w.grasp_t, w.release_t):
        segs += [{"start": prev, "end": g, "label": "pick"}, {"start": g, "end": r, "label": "place"}]
        prev = r
    write_oracle(root, {"source": "synthetic", "frame": "base", "objects": SIM_CAT, "relations": ["on", "inhand", "handempty"],
                        "arms": ["main"]}, states, segs, {"success": True})
    return w


class FakeDetector:
    """FAKE: synthetic world positions (+ small deterministic noise); an object under the gripper (not held)
    is missed; ``fail_prompts`` return nothing (to test prompt candidates)."""

    name = "synthetic_fake"

    def __init__(self, world: SwapWorld, fail_prompts: tuple[str, ...] = ()) -> None:
        self.w = world
        self.fail = set(fail_prompts)
        self.calls = 0
        self.rng = np.random.default_rng(0)

    def detect(self, cam: Any, categories: Any) -> list[Detection]:
        t = cam.t
        out = []
        for cat in categories:
            self.calls += 1
            if cat.sam3_text in self.fail:
                continue
            if cat.key == "plate":
                objs = {k: np.array([x, y, PLATE_Z]) for k, (x, y) in PLATES.items()}
                size = np.array([0.12, 0.12, 0.02])
            elif cat.key == "bowl":
                objs, size = self.w.bowls[t], np.array([0.13, 0.13, 0.05])
            else:
                continue
            eef = self.w.eef[t]
            for name, p in objs.items():
                held = self.w.held[t] == name
                if (not held and cat.key == "bowl" and np.hypot(*(p[:2] - eef[:2])) < 0.04 and eef[2] > p[2] + 0.03):
                    continue  # hidden under the gripper
                c = p + self.rng.normal(0, 0.002, 3)
                u, v = project(c)
                out.append(Detection(cat.key, 0.9, (u, v), (u - 10, v - 10, u + 10, v + 10), c, size, prompt=cat.sam3_text))
        self.rng.shuffle(out)
        return out


class TruthBrain(FakeBrain):
    """FAKE VDM answering from the synthetic world at the frame encoded in the image; ``flip`` = {(stmt,
    frame range)} answers inverted (VDM mistakes); init returns the design's goal branches."""

    def __init__(self, world: SwapWorld, flip: list[tuple[str, int, int]] | None = None, bad_every: int = 0,
                 agent_reply: Any = None) -> None:
        super().__init__(bad_every=bad_every, goal=[
            [["on", "bowl_A", "plate_A"], ["on", "bowl_B", "plate_B"], ["on", "bowl_A", "plate_C"]],
            [["on", "bowl_B", "plate_A"], ["on", "bowl_A", "plate_C"], ["on", "bowl_B", "plate_B"]]])
        self.w = world
        self.flip = flip or []
        self.agent_reply = agent_reply
        self.asked: list[tuple[int | None, str, str]] = []

    def agent(self, messages: list[dict[str, Any]]) -> str:
        if self.agent_reply is not None:
            return self.agent_reply(messages) if callable(self.agent_reply) else self.agent_reply
        return super().agent(messages)

    def answer(self, stmt: str, t: int | None) -> str:
        m = re.match(r"\((\w+)\s*([\w ]*)\)", stmt)
        if m is None or t is None:
            return "?"
        pred, args = m.group(1), m.group(2).split()
        sim = [ALIAS.get(a, a) for a in args]
        tr = self.w.truth(t)
        if pred == "on" and len(sim) == 2:
            v = tr.get(("on", sim[0], sim[1]))
            # a bowl on the rim still looks "on" to the VDM (the bg_est2 failure mode)
            p = self.w.bowls[t].get(sim[0])
            if p is not None and sim[1] in PLATES and np.hypot(p[0] - PLATES[sim[1]][0], p[1] - PLATES[sim[1]][1]) < 0.08 \
                    and self.w.held[t] != sim[0]:
                v = True
        else:
            v = None
        if v is None:
            return "?"
        ans = "1" if v else "0"
        if any(stmt == s and a <= t <= b for s, a, b in self.flip):
            ans = "0" if ans == "1" else "1"
        self.asked.append((t, stmt, ans))
        return ans
