"""Smoke test for a benchmark env: build the low-level env from a YAML, reset, take random
actions, and save camera images + memory state.

Usage (CPU-only rendering):
    CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    uv run --no-sync python scripts/bench_smoke.py \
        --config-path env_configs/libero_mem/franka_libero_mem_6_swap_bowls.yaml \
        --out-dir outputs/bench_smoke/libero_mem_6
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import yaml
from PIL import Image

from capx.envs.configs.instantiate import instantiate


@dataclass
class Args:
    config_path: str
    out_dir: str = "outputs/bench_smoke"
    num_steps: int = 40
    """Number of random actions."""
    seed: int = 1
    save_every: int = 10


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else f"ndarray{x.shape}"
    if isinstance(x, np.generic):
        return x.item()
    return x


def _save_images(obs: dict[str, Any], out: Path, tag: str) -> list[str]:
    saved = []
    for cam, v in obs.items():
        if isinstance(v, dict) and "images" in v and "rgb" in v["images"]:
            path = out / f"{tag}_{cam}.png"
            Image.fromarray(np.ascontiguousarray(v["images"]["rgb"]).astype(np.uint8)).save(path)
            saved.append(path.name)
    return saved


def _random_step(env: Any, rng: np.random.Generator) -> None:
    """One random action through the cap-x env's control interface."""
    if hasattr(env, "move_to_joints_blocking"):  # Franka/MuJoCo envs (LIBERO, LIBERO-Mem)
        joints = np.asarray(env.get_observation()["robot_joint_pos"][:7], dtype=np.float64)
        env._set_gripper(float(rng.uniform()))
        env.move_to_joints_blocking(joints + rng.normal(scale=0.05, size=7), max_steps=5)
    else:
        env.step(env.action_space.sample())


def main(args: Args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config_path).read_text())
    low_level_cfg = cfg["env"]["cfg"]["low_level"]

    t0 = time.time()
    env = instantiate(low_level_cfg)
    obs, info = env.reset(seed=args.seed)
    log: dict[str, Any] = {
        "config_path": args.config_path,
        "low_level": low_level_cfg,
        "reset_info": _jsonable(info),
        "reset_s": round(time.time() - t0, 2),
        "images": _save_images(obs, out, "step000"),
        "steps": [],
    }
    rng = np.random.default_rng(args.seed)
    for i in range(1, args.num_steps + 1):
        _random_step(env, rng)
        obs = env.get_observation()
        rec = {"step": i, "reward": _jsonable(env.compute_reward()), "memory": _jsonable(obs.get("memory"))}
        if i % args.save_every == 0 or i == args.num_steps:
            rec["images"] = _save_images(obs, out, f"step{i:03d}")
        log["steps"].append(rec)
    log["task_completed"] = bool(env.task_completed())
    log["total_s"] = round(time.time() - t0, 2)
    (out / "smoke_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: log[k] for k in ("reset_info", "reset_s", "total_s", "task_completed")}, default=str))
    print(f"last memory: {log['steps'][-1]['memory']}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
