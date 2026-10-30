"""Generate fixed init states for LIBERO-Mem tasks that ship none upstream.

Upstream builds its .pruned_init files by resetting until no two plates touch and then keeping the
scenes a person approves (scripts/mem_1_develop_init_states.py in the LIBERO-Mem submodule). This script
automates that check. A sampled scene is kept when:
  1. no plate touches another plate right after spawn,
  2. after SETTLE_STEPS idle steps every object is tilted less than MAX_TILT_DEG and moved less than
     MAX_DRIFT_M over the last SETTLE_TAIL steps,
  3. every BDDL :init predicate holds (evaluated by the env).
Attempt k is sampled after np.random.seed(seed + k), so reruns produce the same states. Output goes to
capx/integrations/libero_mem/init_states/<task>.pruned_init (same format as upstream, loaded by
_load_init_states when upstream has no file) plus <task>.json metadata, and an agentview gallery of the
kept scenes for review.

Usage (CPU works):
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 \\
  python scripts/gen_libero_mem_init_states.py --tasks 1 2 3 4 5 9 --num 50 --gallery-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from capx.integrations.libero_mem import GENERATED_INIT_DIR, load_libero_mem_task

SETTLE_STEPS = 50
SETTLE_TAIL = 10
MAX_TILT_DEG = 10.0
MAX_DRIFT_M = 0.01
HOLD_ACTION = np.concatenate([np.zeros(7), [-1.0]])  # JOINT_POSITION: hold the arm, gripper open


def _plate_contact(sim) -> bool:
    for c in sim.data.contact[: sim.data.ncon]:
        g1, g2 = sim.model.geom_id2name(c.geom1) or "", sim.model.geom_id2name(c.geom2) or ""
        if "plate" in g1 and "plate" in g2 and sim.model.geom_bodyid[c.geom1] != sim.model.geom_bodyid[c.geom2]:
            return True
    return False


def _object_poses(inner) -> dict[str, tuple[np.ndarray, float]]:
    """Position and tilt (degrees from upright) of every movable object."""
    sim = inner.sim
    out = {}
    for name, body_id in inner.obj_body_id.items():
        if name not in inner.objects_dict:
            continue  # fixtures
        tilt = float(np.degrees(np.arccos(np.clip(sim.data.body_xmat[body_id].reshape(3, 3)[2, 2], -1.0, 1.0))))
        out[name] = (sim.data.body_xpos[body_id].copy(), tilt)
    return out


def _check(env) -> tuple[bool, str]:
    inner = env.env
    if _plate_contact(inner.sim):
        return False, "plate_contact"
    for _ in range(SETTLE_STEPS - SETTLE_TAIL):
        env.step(HOLD_ACTION)
    before = _object_poses(inner)
    for _ in range(SETTLE_TAIL):
        env.step(HOLD_ACTION)
    after = _object_poses(inner)
    if any(tilt >= MAX_TILT_DEG for _, tilt in after.values()):
        return False, "tilted"
    if any(np.linalg.norm(after[k][0] - before[k][0]) >= MAX_DRIFT_M for k in after):
        return False, "moving"
    if not all(inner._eval_predicate(p) for p in inner.parsed_problem["initial_state"]):
        return False, "init_predicate"
    return True, "ok"


def _gallery(images: list[np.ndarray], path: str, cols: int = 10) -> None:
    from PIL import Image, ImageDraw

    h, w = images[0].shape[:2]
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    for i, img in enumerate(images):
        tile = Image.fromarray(img)
        ImageDraw.Draw(tile).text((4, 2), str(i), fill=(255, 255, 0))
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    sheet.save(path, quality=85)


def generate(task_id: int, num: int, seed: int, max_attempts: int, gallery_dir: str | None) -> dict:
    import torch

    handle = load_libero_mem_task(task_id, cam_w=64, cam_h=64, controller="JOINT_POSITION", horizon=100000)
    env = handle.env
    from libero.libero import benchmark  # type: ignore[import-not-found]

    task_name = benchmark.get_benchmark_dict()["libero_mem"]().get_task(task_id).name
    states, kept_attempts, images = [], [], []
    rejected = {"plate_contact": 0, "tilted": 0, "moving": 0, "init_predicate": 0}
    attempts = 0
    start = time.time()
    while len(states) < num and attempts < max_attempts:
        np.random.seed(seed + attempts)
        env.reset()
        ok, why = _check(env)
        attempts += 1
        if not ok:
            rejected[why] += 1
            continue
        states.append(env.env.sim.get_state().flatten())
        kept_attempts.append(attempts - 1)
        images.append(env.env.sim.render(width=256, height=160, camera_name="agentview")[::-1])
        print(f"[task {task_id}] kept {len(states)}/{num} after {attempts} attempts", flush=True)

    os.makedirs(GENERATED_INIT_DIR, exist_ok=True)
    torch.save(np.array(states), os.path.join(GENERATED_INIT_DIR, f"{task_name}.pruned_init"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    meta = {
        "task_id": task_id, "task_name": task_name, "num_states": len(states), "seed": seed,
        "attempts": attempts, "pass_rate": round(len(states) / attempts, 3) if attempts else 0.0,
        "kept_attempts": kept_attempts, "rejected": rejected,
        "criteria": {"settle_steps": SETTLE_STEPS, "settle_tail": SETTLE_TAIL, "max_tilt_deg": MAX_TILT_DEG,
                     "max_drift_m": MAX_DRIFT_M, "no_plate_plate_contact_at_spawn": True,
                     "all_bddl_init_predicates": True, "controller": "JOINT_POSITION"},
        "generated_by_commit": commit, "seconds": round(time.time() - start),
    }
    with open(os.path.join(GENERATED_INIT_DIR, f"{task_name}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if gallery_dir and images:
        os.makedirs(gallery_dir, exist_ok=True)
        _gallery(images, os.path.join(gallery_dir, f"task_{task_id:02d}_{task_name}.jpg"))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 9])
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=2000)
    p.add_argument("--gallery-dir", default=None)
    a = p.parse_args()
    for t in a.tasks:
        m = generate(t, a.num, a.seed, a.max_attempts, a.gallery_dir)
        print(json.dumps({k: m[k] for k in ("task_id", "num_states", "attempts", "pass_rate", "rejected")}))


if __name__ == "__main__":
    main()
