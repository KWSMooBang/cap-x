"""LIBERO-Mem (arXiv 2511.11478) task loading.

LIBERO-Mem is a LIBERO fork that ships under the same ``libero`` distribution name as
LIBERO-PRO, so the two live in mutually exclusive uv extras (``libero`` vs ``libero-mem``).
Its ``setup.py`` does not expose an importable package to PEP 660 editable installs, so —
like the vendored-LIBERO fallback in ``capx.integrations.libero`` — we put the submodule
root on ``sys.path`` ourselves.

Memory-relevant state: tasks use BDDL ``(Sequence ...)`` / ``(Or ...)`` goals that are
tracked by the env (``_satisfied_subgoals``, ``_overshot``). Upstream evaluation advances
that tracker by calling ``env._check_success(inc=True)`` after every step
(``scripts/mem_5_run_evaluation_env_gt.py``); :class:`LiberoMemHandle` does the same.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from capx.integrations.libero import LiberoHandle, _extract_language_from_bddl

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

LIBERO_MEM_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "LIBERO-Mem")
)
SUITE_NAME = "libero_mem"


def ensure_libero_mem_importable() -> None:
    """Make ``libero.libero`` (LIBERO-Mem) importable, or raise ImportError.

    Only touches ``sys.path`` when the installed ``libero`` distribution is LIBERO-Mem, so
    the LIBERO-PRO install (``--extra libero``) is never shadowed by the submodule checkout.
    """
    try:
        direct_url = metadata.distribution("libero").read_text("direct_url.json") or ""
    except metadata.PackageNotFoundError as e:
        raise ImportError("LIBERO-Mem not installed; run `uv sync --extra libero-mem`.") from e
    if "LIBERO-Mem" not in direct_url:
        raise ImportError("Installed `libero` is not LIBERO-Mem; run `uv sync --extra libero-mem`.")

    # libero/libero/__init__.py reads (and interactively creates) $LIBERO_CONFIG_PATH/config.yaml
    # at import time. Use a dedicated config so we never read another LIBERO install's paths.
    config_dir = os.environ.get(
        "LIBERO_MEM_CONFIG_PATH", os.path.expanduser("~/.cache/capx/libero_mem")
    )
    os.environ["LIBERO_CONFIG_PATH"] = config_dir
    config_file = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_file):
        import yaml

        bench_root = os.path.join(LIBERO_MEM_ROOT, "libero", "libero")
        os.makedirs(config_dir, exist_ok=True)
        with open(config_file, "w") as f:
            yaml.safe_dump(
                {
                    "benchmark_root": bench_root,
                    "bddl_files": os.path.join(bench_root, "bddl_files"),
                    "init_states": os.path.join(bench_root, "init_files"),
                    "datasets": os.path.join(LIBERO_MEM_ROOT, "libero", "datasets"),
                    "assets": os.path.join(bench_root, "assets"),
                },
                f,
            )

    if LIBERO_MEM_ROOT not in sys.path:
        sys.path.append(LIBERO_MEM_ROOT)


@dataclass
class LiberoMemHandle(LiberoHandle):
    num_subgoals: int = 1

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed)
        self.reset_progress()
        return obs, info

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        # Advance the sequential-subgoal tracker exactly like upstream evaluation.
        self.env.env._check_success(inc=True)
        return obs, reward, done, info

    def reset_progress(self) -> None:
        self.env.env.reset_subgoal_progress()
        self.env.env._overshot = False  # not cleared by reset_subgoal_progress()

    def memory_state(self) -> dict[str, Any]:
        """Task-progress state tracked by LIBERO-Mem (exposed for memory modules)."""
        problem = self.env.env
        try:
            satisfied = problem.get_satisfied_subgoals(self.task_language)
        except KeyError:  # language not in upstream's SIMPLIFY table
            satisfied = problem.get_satisfied_subgoals()
        return {
            "satisfied_subgoals": satisfied,
            "num_satisfied_subgoals": len(problem._satisfied_subgoals),
            "num_subgoals": self.num_subgoals,
            "overshot": bool(problem._overshot),
        }


def _load_init_states(task_name: str) -> Any:
    """Pregenerated init states. Upstream only ships these for a few tasks (scripts/__init_data);
    for the rest, BDDL placement sampling on reset is used."""
    import torch

    path = os.path.join(LIBERO_MEM_ROOT, "scripts", "__init_data", f"{task_name}.pruned_init")
    if not os.path.exists(path):
        print(f"Warning: no LIBERO-Mem init states for {task_name}; using random BDDL placement")
        return None
    return torch.load(path)


def load_libero_mem_task(
    task_id: int,
    cam_w: int = 128,
    cam_h: int = 128,
    controller: str = "OSC_POSE",
    horizon: int = 1000,
    control_freq: int = 20,
    camera_depths: bool = True,
) -> LiberoMemHandle:
    ensure_libero_mem_importable()
    from libero.libero import benchmark, get_libero_path  # type: ignore[import-not-found]
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]

    task_suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    task = task_suite.get_task(task_id)
    bddl_file_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=cam_h,
        camera_widths=cam_w,
        controller=controller,
        horizon=horizon,
        control_freq=control_freq,
        camera_depths=camera_depths,
        # Upstream default passes camera names as segmentation levels; request instance
        # masks so FrankaLiberoEnv exposes them as images["segmentation"].
        camera_segmentations="instance",
    )
    env.seed(0)

    return LiberoMemHandle(
        env=env,
        suite_name=SUITE_NAME,
        task_id=task_id,
        task_language=_extract_language_from_bddl(bddl_file_path) or task.language,
        init_states=_load_init_states(task.name),
        num_subgoals=env.env.get_goal_sequence_len(),
    )


def libero_mem_num_tasks() -> int:
    ensure_libero_mem_importable()
    from libero.libero import benchmark  # type: ignore[import-not-found]

    return benchmark.get_benchmark_dict()[SUITE_NAME]().n_tasks
