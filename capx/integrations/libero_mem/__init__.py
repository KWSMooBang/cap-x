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
    # Where init_states came from: upstream | capbg_generated | random (BDDL placement on reset).
    init_source: str = "random"

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed)
        self.reset_progress()
        return obs, info

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        obs, _, _, info = self.env.step(action)
        # Advance the sequential-subgoal tracker exactly like upstream evaluation, then
        # compute reward/done from the updated progress (the env's own values predate it).
        problem = self.env.env
        problem._check_success(inc=True)
        done = self.task_success()
        reward = 1.0 if done else 0.0  # LIBERO's sparse completion reward
        return obs, reward, done, info

    def task_success(self) -> bool:
        """Full-task success. Upstream ``_check_success()`` also returns True on the step an
        intermediate subgoal is appended, so require the whole sequence for Sequence/Or goals."""
        problem = self.env.env
        is_sequential = problem.parsed_problem["goal_state"][0][0] in ("sequence", "or")
        if is_sequential and len(problem._satisfied_subgoals) < self.num_subgoals:
            return False
        return bool(problem._check_success())

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


def patch_robosuite_egl_context() -> None:
    """Make LIBERO-Mem's robosuite 1.4.1 EGL context work on Python 3.12 with PyOpenGL 3.1.6.

    Its ``EGLGLContext.__init__`` passes ``ctypes.byref(config)`` to ``eglChooseConfig``, which
    PyOpenGL 3.1.6 rejects (``No array-type handler for type _ctypes.CArgObject``). Pass the
    ``EGLConfig`` itself instead, as the robosuite fork used for LIBERO-PRO does. No-op unless
    ``MUJOCO_GL=egl``.
    """
    if os.environ.get("MUJOCO_GL") != "egl":
        return
    import atexit
    import ctypes

    from robosuite.renderers.context import egl_context

    EGL = egl_context.EGL

    def __init__(self, max_width, max_height, device_id=0):
        del max_width, max_height  # unused
        num_configs = ctypes.c_long()
        config = EGL.EGLConfig()
        EGL.eglReleaseThread()
        if egl_context.EGL_DISPLAY is None:
            display = egl_context.create_initialized_egl_device_display(device_id=device_id)
            if display == EGL.EGL_NO_DISPLAY:
                raise ImportError("Cannot initialize an EGL device display.")
            egl_context.EGL_DISPLAY = display
            atexit.register(EGL.eglTerminate, display)
        display = egl_context.EGL_DISPLAY
        EGL.eglChooseConfig(display, egl_context.EGL_ATTRIBUTES, config, 1, num_configs)
        if num_configs.value < 1:
            raise RuntimeError(f"EGL found no framebuffer config for {egl_context.EGL_ATTRIBUTES}")
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        self._context = EGL.eglCreateContext(display, config, EGL.EGL_NO_CONTEXT, None)
        if not self._context:
            raise RuntimeError("Cannot create an EGL context.")

    def __del__(self):
        # At interpreter exit eglTerminate (atexit) may run first; freeing then raises EGLError.
        try:
            self.free()
        except Exception:
            pass

    egl_context.EGLGLContext.__init__ = __init__
    egl_context.EGLGLContext.__del__ = __del__


# Init states generated in cap-x (scripts/gen_libero_mem_init_states.py) for tasks upstream ships none for.
GENERATED_INIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_states")


def _load_init_states(task_name: str) -> tuple[Any, str]:
    """Pregenerated init states and their source.

    Upstream ships these for a few tasks only (scripts/__init_data); cap-x generates the rest with the
    same acceptance idea (GENERATED_INIT_DIR). Without either, reset uses BDDL placement sampling,
    which often spawns overlapping plates that topple objects.
    """
    import torch

    upstream = os.path.join(LIBERO_MEM_ROOT, "scripts", "__init_data", f"{task_name}.pruned_init")
    # Both are pickled numpy arrays from trusted sources (submodule, this repo); torch>=2.6 defaults
    # to weights_only=True, which rejects them unless something else has patched torch.load.
    if os.path.exists(upstream):
        return torch.load(upstream, weights_only=False), "upstream"
    generated = os.path.join(GENERATED_INIT_DIR, f"{task_name}.pruned_init")
    if os.path.exists(generated):
        return torch.load(generated, weights_only=False), "capbg_generated"
    print(f"Warning: no LIBERO-Mem init states for {task_name}; using random BDDL placement")
    return None, "random"


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
        # Upstream's default passes camera names as segmentation levels, so set it explicitly.
        # No segmentation: with privileged=False a GT instance mask in images["segmentation"]
        # would leak privileged state (LIBERO-PRO has none), and robosuite's per-pixel ID
        # remapping made each control step ~20x slower.
        camera_segmentations=None,
    )
    env.seed(0)

    init_states, init_source = _load_init_states(task.name)
    return LiberoMemHandle(
        env=env,
        suite_name=SUITE_NAME,
        task_id=task_id,
        task_language=_extract_language_from_bddl(bddl_file_path) or task.language,
        init_states=init_states,
        num_subgoals=env.env.get_goal_sequence_len(),
        init_source=init_source,
    )


def libero_mem_num_tasks() -> int:
    ensure_libero_mem_importable()
    from libero.libero import benchmark  # type: ignore[import-not-found]

    return benchmark.get_benchmark_dict()[SUITE_NAME]().n_tasks
