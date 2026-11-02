"""Per-trial memory of one launch.py task process, reproduced without model servers.

Runs the real trial loop (capx.envs.runner._run_trial_batch: one env reused for all trials, video on) on a
real task config, with the model calls faked at ``capx.llm.client.query_model`` (so the multimodel ensemble
still fans out in its thread pool) and the generated code moving the arm through the real simulator. After
every trial it logs RSS, the Python-heap size (tracemalloc, optional) and the largest live numpy arrays.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python scripts/trial_mem_repro.py \
        --config-path env_configs/capbg_ensemble/libero_mem/task_07_rotate_3_bowls.yaml \
        --trials 5 --output-dir /tmp/trial_mem --rss-log /tmp/trial_mem/rss.jsonl

The fake replies are deterministic (same trial -> same code, decisions and texts), so two runs of the same
version write the same text artifacts; that is how the fix was checked to leave the output files unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return float("nan")


# Arm waypoints (joint offsets from home) for one code block; each move_to_joints is up to 120 sim steps.
MOVE_CODE = """import numpy as np
obs = get_observation()
home = np.array(obs["robot_joint_pos"][:7], dtype=np.float64)
for k in range({n_moves}):
    offset = np.zeros(7)
    offset[k % 7] = {amp} * (1 if k % 2 == 0 else -1)
    move_to_joints(home + offset)
    if k % 3 == 0:
        close_gripper() if k % 2 == 0 else open_gripper()
    obs = get_observation()
    print("move", k, float(obs["robot_joint_pos"][0]))
goto_home_joint_position()
"""


class FakeModels:
    """Deterministic stand-in for every model call (coding candidates, synthesis, VDM, goal parse)."""

    def __init__(self, n_moves: int, regenerations: int, vdm_model: str) -> None:
        self.vdm_model = vdm_model
        self.n_moves = n_moves
        self.regenerations = regenerations

    def code(self, salt: int) -> str:
        return "```python\n" + MOVE_CODE.format(n_moves=self.n_moves, amp=0.15 + 0.01 * (salt % 5)) + "```"

    def __call__(self, args, prompt) -> dict:
        text = json.dumps(prompt, default=str)
        model = getattr(args, "model", "")
        if model == self.vdm_model:
            content = ("The robot arm moved; the bowls stay on the table. " * 40).strip()
        elif "Please respond with EXACTLY ONE" in text or "multi-turn robot control task" in text:
            # Multi-turn decision (candidate or synthesis). The turn is read from the amplitude of the code
            # that just ran (no shared counter: the ensemble calls this from several threads at once).
            ran = text.split("<candidate_solutions>")[0]
            amps = re.findall(r"offset\[k % 7\] = ([0-9.]+)", ran)
            k = round((float(amps[-1]) - 0.15) / 0.01) if amps else 0
            if k >= self.regenerations:
                content = "The task appears complete.\nFINISH"
            else:
                content = "Not done yet.\nREGENERATE\n" + self.code(k + 1)
        else:
            content = "Plan: move the arm through waypoints.\n" + self.code(0)
        return {"content": content, "reasoning": "fake reasoning " * 50, "finish_reason": "stop",
                "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": len(content) // 4, "model": model}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rss-log", required=True)
    p.add_argument("--n-moves", type=int, default=12)
    p.add_argument("--regenerations", type=int, default=2)
    p.add_argument("--memory-mode", default="none")
    p.add_argument("--tracemalloc", action="store_true")
    p.add_argument("--vdm-model", default="Qwen/Qwen3.8-27B")
    a = p.parse_args()

    import capx.envs.runner as runner
    import capx.envs.trial as trial_mod
    import capx.llm.client as client
    from capx.envs.launch import LaunchArgs
    from capx.utils.launch_utils import _load_config

    fake = FakeModels(a.n_moves, a.regenerations, a.vdm_model)
    client.query_model = fake  # ensemble candidates / synthesis (through client._timed_query, _synthesize)
    trial_mod._query_model = fake  # VDM, single-model queries, goal parse

    if a.tracemalloc:
        import tracemalloc

        tracemalloc.start(1)

    args = LaunchArgs(
        config_path=a.config_path, model="fake-coder", visual_differencing_model=a.vdm_model,
        total_trials=a.trials, num_workers=1, output_dir=a.output_dir, record_video=True,
        memory_mode=a.memory_mode, memory_est_perception="sim_fake_for_tests" if a.memory_mode == "bg_est" else None,
        trial_timeout_s=100000, max_trial_retries=1,
    )
    env_factory, config, _api_servers = _load_config(args)  # API servers are not started
    runner._setup_output_dir(args, config)
    log_path = Path(a.rss_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    orig = runner._run_trial_with_retries
    state = {"t0": time.time()}

    def wrapped(env, trial, *rest, **kw):
        if trial == 1:
            gc.collect()
            rec = {"trial": 0, "rss_mb": round(rss_mb(), 1), "t": 0.0}
            print("[mem]", rec, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        summary = orig(env, trial, *rest, **kw)
        gc.collect()
        rec = {"trial": trial, "rss_mb": round(rss_mb(), 1), "t": round(time.time() - state["t0"], 1),
               "sim_steps": getattr(getattr(env, "low_level_env", None), "_sim_step_count", None),
               "frame_buffer": len(getattr(getattr(env, "low_level_env", None), "_frame_buffer", []) or [])}
        if a.tracemalloc:
            import tracemalloc

            rec["py_heap_mb"] = round(tracemalloc.get_traced_memory()[0] / 2**20, 1)
            snap = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
            if "snap" in state:
                rec["py_growth_top"] = [
                    f"{d.size_diff / 2**20:+.1f}MB {d.traceback[0].filename.split('/')[-1]}:{d.traceback[0].lineno}"
                    for d in snap.compare_to(state["snap"], "lineno")[:6]
                ]
            state["snap"] = snap
        print("[mem]", rec, flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return summary

    runner._run_trial_with_retries = wrapped
    runner._run_trial_batch(list(range(1, a.trials + 1)), args=args, env_factory=env_factory, config=config)


if __name__ == "__main__":
    main()
