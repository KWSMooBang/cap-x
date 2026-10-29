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
- Memory state: `obs["memory"]` (and `info["memory"]` on reset) holds the env's sequential-subgoal tracker:
  `satisfied_subgoals`, `num_satisfied_subgoals`, `num_subgoals` and `overshot`. As in upstream evaluation, the
  tracker advances on every sim step.
- Init states: upstream ships fixed init states for tasks 0, 6, 7 and 8 only (`scripts/__init_data`). The other
  tasks use random BDDL placement.

Smoke test (CPU rendering):

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
uv run --no-sync python scripts/bench_smoke.py \
    --config-path env_configs/libero_mem/franka_libero_mem_6_swap_bowls.yaml --out-dir outputs/bench_smoke/libero_mem_6
```

Full run: `python capx/envs/launch.py --config-path env_configs/libero_mem/franka_libero_mem_6_swap_bowls.yaml`.
