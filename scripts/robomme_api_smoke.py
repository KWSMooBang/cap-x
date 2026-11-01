"""RoboMME API smoke test (GPU + NVIDIA Vulkan): scripted pick-and-place through the CaP-X API.

Runs fixed code (no LLM) in the same sandbox the coding model uses: SAM3 text segmentation of the target cube,
Contact-GraspNet grasp (top-down centroid grasp if none is usable), goto_pose with approach, close, lift 15 cm,
lower, open. Saves the front-camera video, before/after images, the sandbox stdout/stderr and RoboMME's own
progress (oracle subgoal / status, logs only) to --out-dir.

    python scripts/robomme_api_smoke.py --config-path env_configs/robomme/robomme_pickxtimes.yaml \
        --out-dir outputs/robomme_smoke/api_pickxtimes --seed 1

What to check: smoke_log.json "checks" (all true), the video, and that RoboMME's subgoal moved past
"pick up the ... cube" after the lift (``memory_after_lift.simple_subgoal_online``).
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

SCRIPTED_CODE = r'''
import re
import numpy as np

goal = GOAL
m = re.search(r"\b(red|green|blue|yellow|purple|orange|white|black|pink|cyan)\b", goal)
color = m.group(1) if m else "red"
obs = get_observation()
cam = obs["agentview"]
rgb, depth, K, T = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
results = segment_sam3_text_prompt(rgb, f"{color} cube")
print("sam3 results:", len(results), [round(float(r["score"]), 3) for r in results])
best = max(results, key=lambda r: r["score"])
pts = mask_to_world_points(best["mask"], depth, K, T)
center = np.median(pts, axis=0)
print("cube center (base frame):", np.round(center, 4), "n_points:", len(pts))

top_down = np.array([0.0, 1.0, 0.0, 0.0])
grasp_pos, grasp_quat, source = center.copy(), top_down, "centroid"
try:
    poses, scores = plan_grasp(depth, K, best["mask"].astype(np.int32))
    g_world, g_score = select_top_down_grasp(poses, scores, T, vertical_threshold=0.8)
    if g_world is not None:
        grasp_pos, grasp_quat = decompose_transform(g_world)
        source = f"graspnet (score {float(g_score):.3f})"
except Exception as e:
    print("plan_grasp failed:", repr(e))
print("grasp source:", source, "pos:", np.round(grasp_pos, 4), "quat:", np.round(grasp_quat, 3))

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
print("tcp at grasp:", np.round(get_observation()["robot_cartesian_pos"][:3], 4))
close_gripper()
width_closed = float(get_observation()["robot_joint_pos"][7])
print("gripper after close (0 closed .. 1 open):", round(width_closed, 3))
lift = grasp_pos + np.array([0.0, 0.0, 0.15])
goto_pose(lift, grasp_quat)
LIFT_DONE = True
print("tcp after lift:", np.round(get_observation()["robot_cartesian_pos"][:3], 4))
'''

PLACE_CODE = r'''
goto_pose(grasp_pos + np.array([0.0, 0.0, 0.01]), grasp_quat)
open_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, 0.12]), grasp_quat)
print("released; tcp:", np.round(get_observation()["robot_cartesian_pos"][:3], 4))
'''


@dataclass
class Args:
    config_path: str = "env_configs/robomme/robomme_pickxtimes.yaml"
    out_dir: str = "outputs/robomme_smoke/api_pickxtimes"
    seed: int = 1
    """Trial seed: seed k plays test episode k-1."""
    start_servers: bool = True
    """Start the config's api_servers (SAM3, Contact-GraspNet, pyroki) unless already listening."""


def _save(img: np.ndarray, path: Path) -> None:
    Image.fromarray(np.ascontiguousarray(img).astype(np.uint8)).save(path)


def main(args: Args) -> None:
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.runner import _start_api_servers, _stop_api_servers
    from capx.utils.video_utils import _write_video

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config_path).read_text())
    procs = _start_api_servers(cfg.get("api_servers")) if args.start_servers else []
    log: dict[str, Any] = {"config_path": args.config_path, "seed": args.seed}
    try:
        t0 = time.time()
        env = instantiate(cfg["env"])
        obs, info = env.reset(seed=args.seed)
        low = env.low_level_env
        log["reset_s"] = round(time.time() - t0, 2)
        log["episode_idx"] = info.get("episode_idx", low.handle.episode_idx)
        log["goal"] = low.handle.task_goal()
        log["tcp_offset_hand"] = low.tcp_offset_hand.tolist()
        log["num_conditioning_frames"] = low.num_conditioning_frames()
        log["memory_reset"] = low.memory_state()
        _save(obs["agentview"]["images"]["rgb"], out / "before_front.png")
        _save(obs["robot0_eye_in_hand"]["images"]["rgb"], out / "before_wrist.png")
        env.enable_video_capture(True, clear=True)

        code = f"GOAL = {log['goal']!r}\n" + SCRIPTED_CODE
        _, _, _, truncated, step_info = env.step(code)
        log["pick"] = {k: step_info.get(k) for k in ("sandbox_rc", "stdout", "stderr")}
        log["memory_after_lift"] = low.memory_state()
        lifted = bool(env._exec_globals.get("LIFT_DONE"))
        g = env._exec_globals
        if not truncated and lifted:
            _, _, _, truncated, step_info = env.step(PLACE_CODE)
            log["place"] = {k: step_info.get(k) for k in ("sandbox_rc", "stdout", "stderr")}
        log["memory_end"] = low.memory_state()
        log["sim_steps"] = low._sim_step_count
        final = env.low_level_env.get_observation()
        _save(final["agentview"]["images"]["rgb"], out / "after_front.png")

        width = float(np.asarray(g.get("width_closed", 1.0)))
        center = np.asarray(g.get("center", [np.nan] * 3), dtype=float)
        log["checks"] = {
            "pick_code_ran_without_error": log["pick"]["sandbox_rc"] == 0,
            "cube_center_on_table": bool(np.all(np.isfinite(center)) and -0.01 < center[2] < 0.1),
            "gripper_stopped_on_object": 0.05 < width < 0.95,
            "lift_done": lifted,
            "robomme_subgoal_advanced": log["memory_after_lift"].get("simple_subgoal_online")
            != log["memory_reset"].get("simple_subgoal_online"),
            "place_code_ran_without_error": log.get("place", {}).get("sandbox_rc") == 0,
        }
        frames = env.get_video_frames(clear=True)
        if frames:
            _write_video(frames, str(out), suffix="smoke")
        log["total_s"] = round(time.time() - t0, 2)
    finally:
        (out / "smoke_log.json").write_text(json.dumps(log, indent=2, default=str))
        _stop_api_servers(procs)
    print(json.dumps({k: log.get(k) for k in ("goal", "episode_idx", "sim_steps", "checks", "memory_end")},
                     indent=2, default=str))
    print(f"saved to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
