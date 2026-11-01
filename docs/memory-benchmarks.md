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
- Envs: `capx.envs.simulators.robomme.RoboMMEEnv(task_id="PickXtimes", dataset="test", max_steps=1300,
  condition_frames=32)`, registered as `robomme_<TaskID>_low_level`; code env `RoboMMECodeEnv`.
- The GPU pod also needs `--extra contactgraspnet` in the same venv for the Contact-GraspNet server
  (`uv sync --active --extra robomme --extra contactgraspnet`); Molmo2-8B runs from `.venv-molmo` as for LIBERO.

### RoboMME conventions (checked in the RoboMME and ManiSkill source)

| What | Convention |
|---|---|
| Action (`joint_angle`) | Absolute `[7 joint angles (rad), gripper]`, gripper `+1` open / `-1` closed. PatternLock and RouteStick hold a stick: 7-D, and the wrapper drops an 8th value. One `env.step` = one ManiSkill control step (`pd_joint_pos`, 20 Hz). |
| Frames | World: +x forward, +y left, +z up, table top `z = 0`. Panda base (`panda_link0`) at world `(-0.615, 0, 0)`, identity rotation. cap-x reports everything in the **base frame** (world + 0.615 m in x), as on LIBERO. |
| Cameras | `front` (256×256, fov 90°) and `wrist`. `*_camera_extrinsic_list` is ManiSkill `extrinsic_cv`: 3×4 **world → camera**, OpenCV axes. `*_camera_intrinsic` is `intrinsic_cv`. Depth is `int16` **millimetres**. |
| Robot state | `joint_state_list` has the 7 joints. `gripper_state_list` has the two finger positions (0.04 m = open). `eef_state_list` is the TCP (`panda_hand_tcp`, 0.1034 m in front of `panda_hand`) as world xyz + extrinsic-XYZ rpy (unwrapped). |
| Success | `info["status"]`: `success` (the task's `evaluate()` succeeds), `fail` (terminated without success, e.g. the wrong object was picked), `timeout` (step budget), `ongoing` or `error`. RoboMME's own evaluation stops at `terminated or truncated` and scores `status == "success"`. |
| Step budget | Not per task: the paper's MME-VLA runs use `max_steps=1300` for every task (`scripts/evaluation.py`), and the CVPR 2026 challenge uses 1500. We use 1300. |
| Episodes | `test` split: 50 episodes per task, each with a fixed seed and difficulty from `env_metadata`. |
| Goal | `info["task_goal"]` is a list of paraphrases. `[0]` is the main goal, which is what RoboMME's evaluation uses. |

### CaP-X control API (`FrankaRoboMMEApiReducedSkillLibrary`)

`capx.integrations.robomme.api.FrankaRoboMMEApiReducedSkillLibrary` subclasses the LIBERO CaP-Agent0 API
(`FrankaLiberoApiReducedSkillLibrary`). `functions()` has the same names and signatures, so the prompt's API section
is the same text except for the docstrings listed below. `tests/test_robomme_api.py` checks the names and signatures.
Perception (SAM3, Contact-GraspNet, Molmo) and IK (pyroki, `panda_description`, `panda_hand`) use the same servers.
ManiSkill's Panda has the same kinematics: pyroki's IK agrees with the Panda DH model in
`capx/integrations/robomme/geometry.py` (tested). The IK result is sent to RoboMME as a `joint_angle` action.

Differences from LIBERO (each one is in the docstring the model sees):

- `get_observation`: same keys. `agentview` is RoboMME's **front** camera and `robot0_eye_in_hand` its wrist camera.
  `pose_mat` is camera → base (OpenCV axes), and depth is in metres, shape `(H, W)`. `robot_cartesian_pos` is the
  **tool centre point** (between the fingertips, or the stick tip) with the `panda_hand` orientation, so
  `goto_pose(p[:3], p[3:7])` keeps the robot where it is. There is no segmentation image, because it would leak
  object identities.
- `solve_ik` / `goto_pose`: the target position is the TCP. The TCP offset from `panda_hand` is measured from the
  simulator at reset (`RoboMMEEnv.tcp_offset_hand`), instead of LIBERO's fixed 0.1 m. It is about `[0, 0, 0.103]` m
  for the gripper and the stick tip for the stick tasks. Positions are clamped to `x ∈ [-0.1, 0.95]`, `|y| ≤ 0.6`,
  `z ∈ [0, 0.9]`.
- `move_to_joints`: interpolates in joint space by at most 0.06 rad per control step, then waits for the joints
  to settle within 0.01 rad, up to 120 steps in total. Every control step counts toward the 1300-step budget.
- `open_gripper` / `close_gripper`: 15 control steps (LIBERO: 30 sim steps). For PatternLock and RouteStick they
  print that the robot holds a stick and do nothing.
- `goto_home_joint_position`: goes to the pose at the start of the episode. That is RoboMME's reset pose
  `[0, 0, 0, -π/2, 0, π/2, π/4]` (the swing pose for the stick tasks), not LIBERO's home pose.

### Episodes, budget and logs

- `reset(seed=k)` plays test episode `(k-1) % 50`, so trial k is the same episode in every run (trials 1–5 =
  episodes 0–4).
- Every `reset` loads the episode env again if it was already reset (including the reset at construction), so
  RoboMME `env.reset()` runs exactly once per trial. A second `reset()` of the same RoboMME env keeps the previous
  success state (`status == "success"` right after reset); before this fix trial 1 (episode 0, already loaded and
  reset at construction) started "successful" and ended after one control step. `reset` raises
  `RoboMMEResetError` (and logs it) if the reset info is not `status == "ongoing"` with `success` false.
- The episode ends at the first control step whose status is terminal (`success`, `fail`, `timeout`, `error`), or
  at the step that would exceed `max_steps` (`step_cap`). The env records `episode_end_reason` and raises
  `EpisodeTruncated`, so the generated code stops there. `RoboMMECodeEnv.step` then reports `truncated`, and the
  trial ends with `{"decision": "truncated", "reason": <episode_end_reason>}` in `all_responses.json`. Success is
  the recorded `status == "success"` (`task_completed`), so later motion cannot undo it.
- Configs set `trial_timeout_s: 3600` and `max_trial_retries: 1`. A trial that runs out of time ends as
  `wall_timeout`.
- **Privileged information**: `get_observation()` (and the sandbox's `obs` / `INPUTS`) holds only cameras and robot
  state. RoboMME's `simple_subgoal_online`, `grounded_subgoal_online`, `status`, the goal list and the frame lists
  never reach a prompt or the sandbox. `env.memory_state()` / `info["memory"]` keep them (JSON-safe, without frames)
  for the trial logs only (`reset_memory` and the per-turn `memory` in `all_responses.json`). The goal line is
  `info["task_goal"][0]`, filled into `{libero_environment_goal}` as on LIBERO.
- Configs: `env_configs/robomme/robomme_<task>.yaml` (16 files). They are generated by
  `scripts/gen_robomme_configs.py`, so do not edit them by hand. They use the LIBERO-Mem baseline settings:
  CaP-Agent0 prompts, the same API servers, ensemble off, VDM on, video on, `max_tokens: 32768` and
  `memory_mode: none`. Pass the models on the CLI as for LIBERO-Mem
  (`--model moonshotai/Kimi-K3 --visual-differencing-model Qwen/Qwen3.8-27B`, see `docs/capbg-baseline.md`).

### Conditioning video (Video* and Imitation tasks)

Nine tasks show a demonstration before the robot gets control: VideoUnmask, VideoUnmaskSwap, VideoRepick,
VideoPlaceButton, VideoPlaceOrder, MoveCube, InsertPeg, PatternLock and RouteStick. For these, `reset()` returns all
of the video's frames, and RoboMME treats every reset frame except the last as the video. For CaP-X:

- The **first** prompt gets `condition_frames` front-camera frames (32; set it in the low-level config).
  They are sampled uniformly over the video, including the first and last frame. Each one is a PNG image with a
  caption `Video frame n/K (t = … s, frame i of N)`, after one line that says how long the video was. Wrist frames
  are not included (MME-VLA also uses one view).
- `get_observation()` returns only the current frame. Decision prompts are rebuilt from the env's own prompt and
  never hold the video, so the memory-less baseline sees it once. The VDM's initial scene description uses only the
  current frame.
- `all_responses.json` keeps the first prompt with the images, and `reset_memory.condition_frame_indices` lists the
  sampled frame indices. The trial video starts with the conditioning video (every 2nd frame, with a red border, as
  RoboMME's own recorder marks demonstration frames).
- Why 32: RoboMME's best MME-VLA policy (`perceptual-framesamp-modul`, config `budget: 512`,
  `token_per_image: 16`, `num_views: 1`) keeps a 512-token memory at 16 tokens per front frame, i.e. 32 sampled
  frames. 32 frames give the coding model a similar number of frames (at full resolution). With 8 frames, a video
  with several swaps (e.g. VideoUnmaskSwap) can fall between samples, so a failure would come from sampling, not
  memory. For comparison, the Embodied-Astra VLM baseline gets every 3rd frame, up to 750 images.
- Fallback if the model rejects that many images in one request: set `CAPX_ROBOMME_CONDITION_FRAMES=16` (overrides
  the configs without regenerating them), or pass `--condition-frames 16` to `scripts/run_robomme_pilot.py`, which
  sets it for every task and records it in `pilot_manifest.jsonl`. Report runs with different frame counts as
  different conditions.
- Tasks without a video (e.g. StopCube, PickHighlight, ButtonUnmask*) can still show brief events while the robot
  moves. Generated code only sees what it asks `get_observation()` for, and the VDM sees one image per turn.

### Harness additions used here (all benchmarks)

- `trial_end.json` in every finished trial directory: `reason` (`finish`, `turn_limit`, `step_cap`,
  `wall_timeout`, or the env's own episode end, e.g. RoboMME `success` / `fail` / `timeout`), `task_completed`,
  `turns`, `sim_steps` and `time_s`. It is written last, so it also marks a finished trial for resuming.
- `vdm_usage` (a list of token usages) on the initial entry and every decision entry of `all_responses.json`.
- `--resume-idx k` on the CLI (it was YAML only) runs trials `k..total_trials`.

### Conditioning video for text-only models

`condition_video_mode` (`images` default, `vdm_text`, `auto`) can replace the frames with one VDM description of
them; `auto` does so when a coding-side model is text-only. See `docs/capbg-ensemble.md`.

### Smoke tests and pilot (GPU pod with NVIDIA Vulkan)

```bash
source .venv-robomme/bin/activate
# (1) API smoke: scripted segment -> plan_grasp -> goto_pose -> close -> lift -> place on PickXtimes, no LLM
python scripts/robomme_api_smoke.py --config-path env_configs/robomme/robomme_pickxtimes.yaml \
    --out-dir outputs/robomme_smoke/api_pickxtimes --seed 1
# (2) CaP-X, one task per suite x 1 trial, real models
python scripts/run_robomme_pilot.py --run-name robomme_smoke_suites --trials 1 --workers 2 \
    --tasks PickXtimes VideoUnmaskSwap VideoRepick MoveCube
# Pilot: 16 tasks x 5 trials (episodes 0-4), resumable (rerun the same command)
python scripts/run_robomme_pilot.py --run-name 20261101_robomme_pilot --trials 5 --workers 2
python scripts/analyze_robomme_pilot.py outputs/robomme_pilot/20261101_robomme_pilot
```

`run_robomme_pilot.py` reads the endpoint from `$NEBIUS_SERVER_URL` and the key from `$NEBIUS_API_KEY`. It starts the
SAM3, Contact-GraspNet and pyroki servers once and runs `--workers` tasks at a time. The aggregator writes
`results.tsv` (`cond task trial reason success turns sim_steps time_s`, plus coding and VDM tokens) and `SUMMARY.md`
(success by task and suite, end reasons, mean turns / steps / time, tokens).

Raw env smoke (no API): `uv run --no-sync python scripts/bench_smoke.py --config-path env_configs/robomme/robomme_pickxtimes.yaml`
