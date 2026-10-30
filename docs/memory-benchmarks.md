# Memory Benchmarks (LIBERO-Mem, RoboMME)

## LIBERO-Mem

[LIBERO-Mem](https://libero-mem.github.io) ([arXiv 2511.11478](https://arxiv.org/abs/2511.11478)) adds 10
non-Markovian tasks (`libero_mem` suite) to LIBERO. Each task's goal is a BDDL `(Sequence ...)` or `(Or ...)`, for
example "lift the bowl and place it back on the plate 3 times".

- Submodule: `capx/third_party/LIBERO-Mem` → https://github.com/libero-mem/libero-mem
- It ships its own `libero` and `robosuite` (1.4.1), so the `libero-mem` extra is **mutually exclusive** with
  `libero` and `robosuite`. Use a separate venv:

```bash
uv venv .venv-libero-mem --python 3.12
source .venv-libero-mem/bin/activate
uv sync --active --extra libero-mem --extra contactgraspnet
```

- No `~/.libero/config.yaml` setup is needed. On import, cap-x writes a LIBERO-Mem config to
  `~/.cache/capx/libero_mem/config.yaml`; set `LIBERO_MEM_CONFIG_PATH` to override the location. It then points
  `LIBERO_CONFIG_PATH` at that config, so a LIBERO-PRO install's config is never read.
- Envs: `capx.envs.simulators.libero_mem.FrankaLiberoMemEnv(task_id=0..9)`, registered as
  `franka_libero_mem_<task_id>_low_level`. It uses the same API as LIBERO (`FrankaLiberoApi`).
- Memory state: `env.memory_state()` (and `info["memory"]` on reset and after each code step) holds the
  env's sequential-subgoal tracker:
  `satisfied_subgoals`, `num_satisfied_subgoals`, `num_subgoals` and `overshot`. As in upstream evaluation, the
  tracker advances on every sim step.
  It is ground-truth progress, so it is not in `get_observation()` and generated code cannot read it (with
  `privileged: false` the sandbox also has no `env` global). Trials log it per turn in `all_responses.json`.
- Init states: upstream ships fixed init states for tasks 0, 6, 7 and 8 only (`scripts/__init_data`). Raw BDDL
  placement for the other tasks often spawns overlapping plates that topple the bowl or bottle (8 of 10 resets in
  task 2), so cap-x ships generated sets for tasks 1, 2, 3, 4, 5 and 9 in `capx/integrations/libero_mem/init_states/`
  (50 each, made by `scripts/gen_libero_mem_init_states.py`; criteria and pass rates in the `.json` next to each
  file). `_load_init_states` prefers upstream, then the generated file; `info["init_source"]` (and the initial entry
  of `all_responses.json`) records `upstream`, `capbg_generated` or `random`. Tasks 1-5 share one scene definition,
  so their generated sets are identical; report the generated tasks as their own group.

Smoke test (CPU rendering):

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 \
uv run --no-sync python scripts/bench_smoke.py \
    --config-path env_configs/libero_mem/franka_libero_mem_6_swap_bowls.yaml --out-dir outputs/bench_smoke/libero_mem_6
```

`NUMBA_DISABLE_JIT=1` is needed when `libOSMesa` comes from apt (Ubuntu 24.04): it links the system `libLLVM`,
which clashes with llvmlite's bundled LLVM and segfaults when robosuite JIT-compiles on `reset`. It makes the
first reset slow (~50 s). On GPU nodes, use `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` instead. LIBERO-Mem's bundled
robosuite 1.4.1 cannot create an EGL context with the locked PyOpenGL 3.1.6 on Python 3.12 (`No array-type
handler for type _ctypes.CArgObject`), so `capx.envs.simulators.libero_mem` patches its `EGLGLContext` on import.

Full run: `python capx/envs/launch.py --config-path env_configs/libero_mem/franka_libero_mem_6_swap_bowls.yaml`.

## RoboMME

[RoboMME](https://robomme.github.io) ([arXiv 2603.04639](https://arxiv.org/abs/2603.04639)) has 16 ManiSkill3/SAPIEN
Panda tasks in 4 suites: Counting, Permanence, Reference and Imitation.

- Submodule: `capx/third_party/RoboMME` → https://github.com/RoboMME/robomme_benchmark. Its own nested submodules
  are baseline policies and are not needed: run `git submodule update --init capx/third_party/RoboMME` without
  `--recursive`.
- Requires Python >= 3.11, `torch==2.9.1` and the ManiSkill fork pinned by RoboMME (`YinpeiDai/ManiSkill@07be6fb`).
  The `robomme` extra is exclusive with `libero`, `libero-mem`, `robosuite`, `molmo` and `verl`:

```bash
uv venv .venv-robomme --python 3.12
source .venv-robomme/bin/activate
uv sync --active --extra robomme
```

- **SAPIEN needs a Vulkan ICD even to build a scene.** On a GPU host, the NVIDIA Vulkan ICD must be present
  (`/usr/share/vulkan/icd.d/nvidia_icd.json`, and `NVIDIA_DRIVER_CAPABILITIES` must include `graphics`). Without any
  ICD, env creation segfaults in `sapien_env._setup_scene`.
- **CPU (lavapipe, `mesa-vulkan-drivers`): state only, no camera images.** With
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`, raw ManiSkill envs work with
  `obs_mode="state", render_backend="none"`. Images do not work. SAPIEN exposes lavapipe only as
  `pci:0000:00:00.0`, and ManiSkill's `parse_backend_device_id` splits that string on every `:` (ValueError).
  ManiSkill's `"cpu"` fallback device cannot render. `BenchmarkEnvBuilder` hard-codes
  `obs_mode="rgb+depth+segmentation"`, so `RoboMMEEnv` needs the NVIDIA ICD until ManiSkill parses `pci:` device
  ids (e.g. `split(":", 1)`).
- Envs: `capx.envs.simulators.robomme.RoboMMEEnv(task_id="PickXtimes", episode_idx=0, dataset="test")`, registered
  as `robomme_<TaskID>_low_level`. Actions use RoboMME's `joint_angle` space: absolute 7 joints plus a gripper value
  (+1 open, -1 closed). `reset(seed=k)` picks episode `(k-1) % n` of the split.
- Memory state: `obs["memory"]` holds `task_goal`, `simple_subgoal_online`, `grounded_subgoal_online` and `status`,
  plus `reset_obs_lists`: all frames returned at reset, including the conditioning video for Video* tasks.
  `obs["obs_lists"]` holds every frame from the latest step.
- No cap-x control API is registered for RoboMME yet (`apis: []` in `env_configs/robomme/*.yaml`).

Smoke test (needs Vulkan): `uv run --no-sync python scripts/bench_smoke.py --config-path env_configs/robomme/robomme_pickxtimes.yaml`
