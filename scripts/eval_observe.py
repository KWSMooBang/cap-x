"""Offline accuracy of the VDM Observe (bg_est2, prompt M5) on LIBERO-Mem scenes.

Two separate steps, so the scenes can be made on a CPU box without API keys and queried elsewhere:

1. ``scenes`` (CPU, needs the LIBERO-Mem simulator): reset LIBERO-Mem tasks, teleport objects into
   labelled situations, let physics settle, and save per scene the marked front-camera image
   (``image.png``, marks at the true object centers, aliases / marks as bg_est2 assigns them) and
   ``scene.json`` (statements in M5 order, the simulator truth of each, marks, condition).
   Conditions: ``init`` (task start), ``normal`` (a bowl put on another plate), ``tilted`` (a bowl
   rotated and dropped on a plate rim), ``table`` (an object on the table), ``in_basket`` /
   ``basket_center`` (task 8 / 9), ``arm`` (the arm low over an object, possibly hiding it).

2. ``query`` (any box, needs the VDM): send each scene as an M5 prompt (one image, no declaration) to
   the VDM and score the parsed answers -> ``results.jsonl`` and ``table.md`` (accuracy per
   predicate x condition x truth, ``?`` rate, unreadable replies). Server from the environment:
   ``OBSERVE_VDM_URL`` (chat completions URL), ``OBSERVE_VDM_MODEL``, ``OBSERVE_VDM_API_KEY``.

   ``table`` re-scores existing results without querying.

Examples::

    # CPU (OSMesa), LIBERO-Mem venv
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 \\
      python scripts/eval_observe.py scenes --out outputs/eval_observe/scenes --tasks 2 6 7 8 9 --per-task 8
    # with the VDM server
    OBSERVE_VDM_URL=$NEBIUS_SERVER_URL OBSERVE_VDM_MODEL=Qwen/Qwen3.8-27B OBSERVE_VDM_API_KEY=$NEBIUS_API_KEY \\
      python scripts/eval_observe.py query --scenes outputs/eval_observe/scenes --out outputs/eval_observe/qwen
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import string
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capx.belief.observe import parse_observe, statement_key, statement_text  # noqa: E402
from capx.belief.perception import VOCABULARY, project_to_pixel  # noqa: E402
from capx.belief.prompts_v2 import marks_now_text, observe_turn_prompt  # noqa: E402
from capx.belief.som import MarkTable, draw_marks, is_support, spread_marks  # noqa: E402

CENTER_REGION = "kitchen_table_the_center"
CONDITIONS = ("init", "normal", "tilted", "table", "in_basket", "basket_center", "arm")
# Joint configurations that put the hand low over the table (arm in front of the objects).
ARM_POSES = (
    (0.0, 0.55, 0.0, -1.95, 0.0, 2.5, 0.785),
    (-0.35, 0.6, 0.0, -1.85, 0.0, 2.45, 0.785),
    (0.35, 0.6, 0.0, -1.85, 0.0, 2.45, 0.785),
)


# --- scene generation (simulator) -----------------------------------------------------------------

def _category(sim_name: str):
    for cat in VOCABULARY:
        if any(k in sim_name for k in cat.sim_keys):
            return cat
    return None


def _quat_mat(q: np.ndarray) -> np.ndarray:
    from capx.belief.oracle import _quat_wxyz_to_mat

    return _quat_wxyz_to_mat(np.asarray(q, dtype=np.float64))


class SceneMaker:
    """Teleports LIBERO-Mem objects (simulator access: this is evaluation tooling, not the estimator)."""

    def __init__(self, task_id: int, seed: int = 0) -> None:
        from capx.belief.oracle import LiberoMemSource
        import capx.integrations.libero_mem as lm
        from capx.envs.simulators.libero_mem import FrankaLiberoMemEnv

        if os.environ.get("LIBERO_MEM_ROOT"):  # a LIBERO-Mem checkout elsewhere (its init states)
            lm.LIBERO_MEM_ROOT = os.environ["LIBERO_MEM_ROOT"]

        self.low = FrankaLiberoMemEnv(task_id=task_id, privileged=False, max_steps=100000, control_freq=20)
        self.low.reset(seed=seed)
        self.src = LiberoMemSource(self.low)
        self.task_id = task_id
        self.instruction = getattr(self.low.handle, "task_language", "")

    @property
    def problem(self):
        return self.low.handle.env.env

    @property
    def sim(self):
        return self.problem.sim

    def reset(self, seed: int) -> None:
        self.low.reset(seed=seed)

    def _base(self) -> tuple[np.ndarray, np.ndarray]:
        bid = self.sim.model.body_name2id("robot0_base")
        return np.asarray(self.sim.data.xpos[bid], float), _quat_mat(self.sim.data.xquat[bid])

    def positions(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, float) for k, v in self.src.positions().items()}

    def teleport(self, name: str, pos_base: Sequence[float], tilt_deg: float = 0.0) -> None:
        """Put ``name``'s body origin at ``pos_base`` (robot base frame), optionally tilted about x."""
        base, rot = self._base()
        world = base + rot @ np.asarray(pos_base, float)
        a = np.deg2rad(tilt_deg) / 2.0
        quat = np.array([np.cos(a), np.sin(a), 0.0, 0.0])
        obj = self.problem.objects_dict[name]
        joint = obj.joints[0]
        self.sim.data.set_joint_qpos(joint, np.concatenate([world, quat]))
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()

    def settle(self, steps: int = 40) -> None:
        for _ in range(steps):
            self.low._step_once()

    def arm(self, joints: Sequence[float]) -> None:
        self.low.move_to_joints_blocking(np.asarray(joints, float), max_steps=200)

    def truth(self, key: tuple) -> bool | None:
        pred, s, t = key
        try:
            if pred == "handempty":
                return not any(self.src.grasped(n) for n in self.src.movable_objects())
            if pred == "inhand":
                return self.src.grasped(s)
            if pred == "at":
                return self.src.eval_predicate(("on", s, CENTER_REGION))
            if t == "table":
                return self.src.eval_predicate(("on", s, "table"))
            return self.src.eval_predicate((pred, s, t + ("_contain_region" if pred == "in" else "")))
        except Exception:
            return None

    def center_xy(self) -> np.ndarray | None:
        """Middle of the BDDL center region in the base frame (scene making only)."""
        try:
            site = self.sim.model.site_name2id(CENTER_REGION)
            base, rot = self._base()
            return (rot.T @ (np.asarray(self.sim.data.site_xpos[site]) - base))[:2]
        except Exception:
            return None


def _aliases(names: Sequence[str], positions: dict[str, np.ndarray]) -> dict[str, str]:
    """sim name -> alias the way bg_est2 assigns them at turn 0: per category, by base y (image left first)."""
    out = {}
    by_cat: dict[str, list[str]] = defaultdict(list)
    for n in names:
        cat = _category(n)
        if cat is not None and n in positions:
            by_cat[cat.key].append(n)
    for key, ns in by_cat.items():
        for i, n in enumerate(sorted(ns, key=lambda n: (round(float(positions[n][1]), 3), float(positions[n][0])))):
            out[n] = f"{key}_{string.ascii_uppercase[i]}"
    return out


def scene_statements(alias_of: dict[str, str]) -> list[tuple]:
    """sim-name keys in M5 order: movable x support relations, on table, inhand, handempty, at center."""
    names = list(alias_of)
    cats = {n: _category(n) for n in names}
    movable = [n for n in names if cats[n].key != "plate"]
    supports = [n for n in names if is_support(cats[n].key)]
    keys: list[tuple] = []
    for m in movable:
        for s in supports:
            if s == m or (cats[m].container and cats[s].container):
                continue
            keys.append(("in" if cats[s].container else "on", m, s))
        keys.append(("on", m, "table"))
    for m in movable:
        if cats[m].container:
            keys.append(("at", m, "center"))
    keys += [("inhand", m, None) for m in movable[:1]] + [("handempty", "gripper", None)]
    return keys


def _alias_key(key: tuple, alias_of: dict[str, str]) -> tuple:
    pred, s, t = key
    return (pred, alias_of.get(s, s), alias_of.get(t, t) if t is not None else None)


def save_scene(maker: SceneMaker, out: Path, sid: str, condition: str, detail: str, alias_of: dict[str, str]) -> dict:
    obs = maker.low.get_observation()
    cam = obs["agentview"]
    rgb = np.asarray(cam["images"]["rgb"])
    K, T = np.asarray(cam["intrinsics"]), np.asarray(cam["pose_mat"])
    pos = maker.positions()
    marks = MarkTable()
    for n in sorted(alias_of, key=lambda n: alias_of[n]):
        marks.assign(alias_of[n], _category(n).key)
    shown, missing = [], []
    for m, a in marks.ordered():
        n = next(k for k, v in alias_of.items() if v == a)
        center = pos[n] + np.array([0.0, 0.0, 0.5 * _category(n).size[2]])
        px = project_to_pixel(center, K, T)
        if px is not None and 0 <= px[0] < rgb.shape[1] and 0 <= px[1] < rgb.shape[0]:
            shown.append((m, a, px))
        else:
            missing.append(m)
    spread = spread_marks([(m, px) for m, _, px in shown], rgb.shape)  # readable
    shown = [(m, a, p) for (m, a, _), (_, p) in zip(shown, spread, strict=True)]
    img = draw_marks(rgb, spread)
    d = out / sid
    d.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(img).save(d / "image.png")
    keys = scene_statements(alias_of)
    rec = {
        "id": sid, "task_id": maker.task_id, "instruction": maker.instruction, "condition": condition, "detail": detail,
        "marks_now": marks_now_text([(m, a, px) for m, a, px in shown], missing),
        "statements": [statement_text(_alias_key(k, alias_of)) for k in keys],
        "truth": {statement_text(_alias_key(k, alias_of)): maker.truth(k) for k in keys},
        "aliases": {a: n for n, a in alias_of.items()},
        "positions": {alias_of[n]: [round(float(v), 4) for v in p] for n, p in pos.items() if n in alias_of},
    }
    (d / "scene.json").write_text(json.dumps(rec, indent=1))
    return rec


def make_scenes(out: Path, tasks: Sequence[int], per_task: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    recs = []
    for task in tasks:
        maker = SceneMaker(task, seed)
        made = 0
        trial = 0
        while made < per_task:
            maker.reset(seed + trial)
            maker.settle(5)
            pos = maker.positions()
            names = [n for n in maker.src.movable_objects() if _category(n) is not None]
            alias_of = _aliases(names, pos)
            cats = {n: _category(n).key for n in names}
            bowls = [n for n in names if cats[n] in ("bowl", "bottle")]
            plates = [n for n in names if cats[n] == "plate"]
            cheese = [n for n in names if cats[n] == "cream_cheese"]
            baskets = [n for n in names if cats[n] == "basket"]
            plan = ["init", "arm"]
            if bowls and plates:
                plan += ["normal", "tilted", "table"]
            if cheese and baskets:
                plan += ["in_basket", "table", "basket_center"]
            cond = plan[trial % len(plan)]
            detail = ""
            if cond == "normal":
                b, p = rng.choice(bowls), rng.choice(plates)
                maker.teleport(b, pos[p] + [0.0, 0.0, 0.03])
                detail = f"{alias_of[b]} put on {alias_of[p]}"
            elif cond == "tilted":
                b, p = rng.choice(bowls), rng.choice(plates)
                maker.teleport(b, pos[p] + [0.0, 0.045 * rng.choice([-1, 1]), 0.03], tilt_deg=float(rng.uniform(12, 22)))
                detail = f"{alias_of[b]} dropped tilted on the rim of {alias_of[p]}"
            elif cond == "table":
                o = rng.choice(bowls or cheese)
                xy = np.array([rng.uniform(0.40, 0.70), rng.uniform(-0.30, 0.30)])
                maker.teleport(o, [xy[0], xy[1], float(pos[o][2]) + 0.01])
                detail = f"{alias_of[o]} on the table at ({xy[0]:.2f}, {xy[1]:.2f})"
            elif cond == "in_basket":
                c, b = cheese[0], rng.choice(baskets)
                maker.teleport(c, pos[b] + [0.0, 0.0, 0.05])
                detail = f"{alias_of[c]} dropped into {alias_of[b]}"
            elif cond == "basket_center":
                b = rng.choice(baskets)
                c = maker.center_xy()
                if c is not None:
                    maker.teleport(b, [c[0], c[1], float(pos[b][2]) + 0.01])
                    detail = f"{alias_of[b]} moved to the table center"
            elif cond == "arm":
                maker.arm(ARM_POSES[trial % len(ARM_POSES)])
                detail = f"arm pose {trial % len(ARM_POSES)}"
            maker.settle(60)
            recs.append(save_scene(maker, out, f"t{task:02d}_{made:03d}_{cond}", cond, detail, alias_of))
            made += 1
            trial += 1
            print(f"scene {recs[-1]['id']}: {detail or cond}", flush=True)
    (out / "index.json").write_text(json.dumps([r["id"] for r in recs]))
    return recs


# --- query (VDM) ----------------------------------------------------------------------------------

def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def scene_prompt(rec: dict, image_url: str) -> list[dict[str, Any]]:
    """M5 with one image (image 2 = now), no declaration."""
    return observe_turn_prompt(rec["instruction"], None, image_url, rec["marks_now"], None, rec["statements"])


def default_query_fn() -> Callable[[list[dict[str, Any]]], Any]:
    from capx.llm.client import ModelQueryArgs, query_model

    url, model = os.environ.get("OBSERVE_VDM_URL"), os.environ.get("OBSERVE_VDM_MODEL")
    if not url or not model:
        raise SystemExit("set OBSERVE_VDM_URL and OBSERVE_VDM_MODEL (and OBSERVE_VDM_API_KEY)")
    args = ModelQueryArgs(model=model, server_url=url, api_key=os.environ.get("OBSERVE_VDM_API_KEY"),
                          temperature=float(os.environ.get("OBSERVE_VDM_TEMPERATURE", "0.2")),
                          max_tokens=int(os.environ.get("OBSERVE_VDM_MAX_TOKENS", "32768")))
    return lambda prompt: query_model(args, prompt)


def query_scenes(scenes: Path, out: Path, query_fn: Callable[[list[dict[str, Any]]], Any], limit: int | None = None) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    ids = json.loads((scenes / "index.json").read_text())[:limit]
    done = {}
    res_path = out / "results.jsonl"
    if res_path.exists():  # resume
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    with res_path.open("a") as f:
        for sid in ids:
            if sid in done:
                continue
            rec = json.loads((scenes / sid / "scene.json").read_text())
            prompt = scene_prompt(rec, _data_url(scenes / sid / "image.png"))
            keys = [statement_key(s.strip("()").split()[0], s.strip("()").split()[1:]) for s in rec["statements"]]
            try:
                reply = query_fn(prompt)
                text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
                usage = reply.get("usage") if isinstance(reply, dict) else None
                error = None
            except Exception as exc:  # noqa: BLE001 (keep going; recorded)
                text, usage, error = "", None, f"query failed: {exc}"
            parsed = parse_observe(text, keys, list(rec["aliases"]))
            row = {"id": sid, "condition": rec["condition"], "task_id": rec["task_id"], "reply": text, "error": error,
                   "ok": parsed.ok, "notes": parsed.notes, "usage": usage,
                   "answers": {statement_text(k): parsed.values.get(k) for k in keys}, "truth": rec["truth"]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[sid] = row
            print(f"{sid}: ok={parsed.ok}", flush=True)
    rows = [done[i] for i in ids if i in done]
    write_table(rows, out)
    return rows


def score(rows: Sequence[dict]) -> dict[str, Any]:
    """Counts per (predicate, condition, truth): n, right, '?', missing; plus unreadable replies."""
    cells: dict[tuple, dict[str, int]] = defaultdict(lambda: {"n": 0, "right": 0, "q": 0, "missing": 0})
    bad = 0
    for r in rows:
        if not r["ok"]:
            bad += 1
            continue
        for stmt, truth in r["truth"].items():
            if truth is None:
                continue
            pred = stmt.strip("()").split()[0]
            for cond in (r["condition"], "all"):
                c = cells[(pred, cond, int(truth))]
                c["n"] += 1
                a = r["answers"].get(stmt)
                if a is None:
                    c["missing"] += 1
                elif a == "?":
                    c["q"] += 1
                elif (a == "1") == bool(truth):
                    c["right"] += 1
    return {"cells": {f"{p}|{c}|{t}": v for (p, c, t), v in sorted(cells.items())}, "unreadable": bad, "replies": len(rows)}


def write_table(rows: Sequence[dict], out: Path) -> str:
    s = score(rows)
    lines = [f"# Observe accuracy ({s['replies']} replies, {s['unreadable']} unreadable)", "",
             "Accuracy = right / (answered 1 or 0); ? and missing are counted apart.", "",
             "| predicate | condition | truth | n | accuracy | ? rate | missing |", "|---|---|---|---|---|---|---|"]
    for k, v in s["cells"].items():
        p, c, t = k.split("|")
        answered = v["n"] - v["q"] - v["missing"]
        acc = f"{v['right'] / answered:.2f}" if answered else "-"
        lines.append(f"| {p} | {c} | {t} | {v['n']} | {acc} | {v['q'] / v['n']:.2f} | {v['missing']} |")
    text = "\n".join(lines) + "\n"
    (out / "table.md").write_text(text)
    (out / "score.json").write_text(json.dumps(s, indent=1))
    return text


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scenes")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--tasks", type=int, nargs="+", default=[2, 6, 7, 8, 9])
    a.add_argument("--per-task", type=int, default=8)
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("query")
    b.add_argument("--scenes", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--limit", type=int, default=None)
    c = sub.add_parser("table")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "scenes":
        make_scenes(args.out, args.tasks, args.per_task, args.seed)
    elif args.cmd == "query":
        query_scenes(args.scenes, args.out, default_query_fn(), args.limit)
        print((args.out / "table.md").read_text())
    else:
        rows = [json.loads(x) for x in (args.out / "results.jsonl").read_text().splitlines()]
        print(write_table(rows, args.out))


if __name__ == "__main__":
    main()
