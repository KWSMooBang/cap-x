# cap-bg Nebius ensemble (CaP-Agent0-style, text-only models)

The baseline CaP-X loop with the CaP-Agent0 **multimodel parallel ensemble** on, using Nebius models:

| Role | Model | Temperatures |
| --- | --- | --- |
| Candidates (9 replies per query) | `openai/gpt-oss-120b`, `Qwen/Qwen3-235B-A22B-Instruct-2507`, `deepseek-ai/DeepSeek-V4-Flash-0731` | 0.1, 0.5, 0.9 each |
| Synthesis | `deepseek-ai/DeepSeek-V4-Pro` | 0.2 (upstream) |
| VDM | `Qwen/Qwen3.8-27B` | as the baseline |
| Conditioning-video description (RoboMME VideoUnmaskSwap, VideoPlaceOrder, MoveCube, PatternLock) | `moonshotai/Kimi-K3` | one call, 32 sampled frames |

Only the conditioning-video description uses Kimi-K3 (`condition_video_model`, below); the turn VDM stays
Qwen3.8-27B. Reason: on kubectl, episode 0 of the four video tasks, judged by eye, Qwen3.8-27B's descriptions often
got cup swaps, placement order and the pattern wrong, and Kimi-K3's single 32-frame call got the four tasks right most
evenly. All four tasks use the same method; there is no per-task choice.

The third candidate was `moonshotai/Kimi-K2.7-Code` in the first smoke run. The lead replaced it with
DeepSeek-V4-Flash-0731 after measuring latency on kubectl: Flash replied in 14–56 s with 4/4 well-formed replies;
Kimi-K2.7-Code took 194–812 s. The other options failed: GLM-5.3 spent the whole 32768-token budget on reasoning and
returned empty replies (4/4); Nemotron-3-Ultra dropped the REGENERATE keyword (2/2); Qwen3.5-397B would have made two
of the three candidates Qwen models. All four coding-side models are **text-only**. RoboMME's conditioning video therefore goes into the first prompt as
a VDM description (`condition_video_mode: auto` → `vdm_text`, below).

## Configuration

| Key (YAML) | CLI | Default (unset) |
| --- | --- | --- |
| `ensemble.candidates` (list of `{model, temperatures}`) | `--ensemble-candidates "m1@0.1,0.5,0.9;m2@0.5"` | upstream `ENSEMBLE_CONFIGS` (`openai/gpt-5.4` × 0.1/0.5/0.9) |
| `ensemble.synthesis_model` | `--ensemble-synthesis-model M` | upstream `openai/gpt-5.4` |
| `condition_video_mode`: `images` \| `vdm_text` \| `auto` | `--condition-video-mode` | `images` |
| `condition_video_model` | `--condition-video-model M` | the VDM (`--visual-differencing-model`) |

- The candidates and synthesis model are used by the multimodel ensemble only (`use_parallel_ensemble: true`,
  `use_multimodel: true`). Setting them without both is an error, not ignored.
- Unset, everything is as before: the requests are byte for byte the upstream ones (checked against the base branch
  with a fake server; candidate order in the synthesis prompt is completion order, as upstream), the baseline configs
  resolve to `ensemble_candidates: None`, `ensemble_synthesis_model: None`, `condition_video_mode: images`.
- Candidates are queried in parallel (`max(9, n)` threads). A candidate that raises (HTTP error, timeout after the
  retries) is left out and the synthesis uses the rest; only if all fail does the query fail. The synthesis prompt is
  the upstream one (`capx/llm/client.py`). Upstream, the synthesis prompt carries only the **text** of the original
  prompt (images are not passed to the synthesizer), so with `images` the synthesizer never sees the video.
- When candidates or the synthesis model are configured, the ensemble result also carries the synthesis reply's
  `finish_reason` and `usage`, so the harness's cut-off rule applies: a synthesis cut off at `max_tokens` is not
  executed and the whole ensemble is queried once more (10 more calls). Unset, the result has no `finish_reason`
  (upstream behaviour).
- All models share `--server-url` / `--api-key` (one Nebius endpoint).

### Text-only models

`TEXT_ONLY_MODELS` in `capx/llm/client.py` lists the four models above. `query_model` raises `TextOnlyModelError`
when a prompt with images is about to go to one of them (`query_model_streaming`, used by the web UI, too), and `query_model_ensemble` checks every candidate before
sending anything, so images are never dropped silently and a text-only candidate never "fails" quietly on an image
prompt. Unknown models are assumed to take images (as before). Images reach the coding prompts only through
`use_visual_feedback` (off in all cap-bg configs) and the RoboMME conditioning video.

### Conditioning video (`condition_video_mode`)

- `images`: as before, `condition_frames` (32) PNG frames with captions at the end of the first prompt.
- `vdm_text`: after reset, the VDM describes the conditioning video **part by part** (default): every
  `condition_video_stride`-th frame (default 1 = every frame), `condition_video_segment_frames` frames per VDM call
  (default 20), each part starting with the last `condition_video_overlap` frames of the previous one (default 4), at
  most `condition_video_max_segments` calls (default 12; the stride grows until the video fits). A last part with fewer
  than half a step of new frames is merged into the previous part. Each call gets the frames with their times, the
  task goal (for context only) and the object state the VDM reported at the end of the previous part. It replies with
  `EVENTS:` (new events in this part, with times) and `STATE:` (each object's identity and position, which container
  covers what, placements so far and their order, highlights and buttons). The VDM is told to track every object
  individually, frame by frame. Containers can trade places without the robot touching them, and the first smoke
  missed a swap because the VDM described the set as a whole drifting left. It is told not to state the answer, the
  target or a plan. The first prompt gets every part's events in order, followed by the last part's state as
  "Final state (end of the video)". An empty or cut-off part is asked once more; an empty reply twice fails the trial.
  A part cut off at `max_tokens` on both attempts is used with
  `[truncated: the description of this part was cut off by the output token limit]` after its events and listed in
  `truncated_parts`. Its STATE is not passed on: the next part gets the last complete STATE, and if the last part
  was cut off, the final state heading says it is unknown and gives the last complete part's state. The VDM's
  `max_tokens` is the config's `max_tokens` (32768). The first smoke's single call used 4,590 completion tokens,
  3,826 of them reasoning. The single-call version marks a reply cut off twice the same way.
  `condition_video_segment_frames: 0` gives the first version: one call with the 32 sampled frames.
  The CLI flags are `--condition-video-stride`, `--condition-video-segment-frames`, `--condition-video-overlap` and
  `--condition-video-max-segments`. The RoboMME ensemble configs set the four keys explicitly.
- `condition_video_model` (`vdm_text` only): the model that writes the description. Unset, the VDM writes it and
  every request is byte for byte as before. Set, only the conditioning-video requests go to that model; prompts,
  server, `max_tokens` and temperature are the VDM's, and the turn VDM calls are unchanged
  (`tests/test_ensemble_nebius.py` compares the requests). It must take images: a model in `TEXT_ONLY_MODELS` or not
  in `VLM_MODELS` is an error at config load.
- `auto`: `vdm_text` if any coding-side model (every candidate and the synthesis model, or the single coding model
  without the ensemble) is text-only, else `images`.

Logged in the initial entry of `all_responses.json` as `condition_video`: `mode`, `config_mode`, `n_frames`,
`total_frames`, `frame_indices`, and for `vdm_text` the inserted `description`, `vdm_model`, `segmentation`
(stride used, settings, `n_segments`, `[first, last, count]` per part), `vdm_calls` (part, attempt, frame range,
latency, finish_reason, usage) and `parts` (each part's verbatim `reply`, parsed `events` / `state`, `reasoning`). Its usage is also in `vdm_usage`,
except with a `condition_video_model` other than the VDM: then `vdm_model` is that model, `usage_in_vdm_usage` is
false, and the usage is only in `vdm_calls` (so `vdm_usage` stays the VDM model's). `initial_prompt` in the same
entry is the prompt as sent (no images with `vdm_text`). Decision prompts never carry the video in either mode.

#### First prompt, `images` (VideoUnmaskSwap, stub env; API docs shortened)

```text
=== role: system ===
You are a helpful assistant that generates Python code to directly solve the task.
=== role: user ===
You are controlling a Franka Emika robot with API described below.
Goal: pick up the red cube twice, then press the button

APIs:
[... API documentation omitted ...]
Before you take control, this task showed a video from the front camera (the 'agentview' camera) of 100 frames (5.0 s at 20 frames per second). Below are 32 frames sampled uniformly from it in time order, including the first and last frame. The video is shown only here, once; the current scene is what get_observation() returns.
Video frame 1/32 (t = 0.00 s, frame 1 of 100):
[image: PNG, data URL]
Video frame 2/32 (t = 0.15 s, frame 4 of 100):
[image: PNG, data URL]
[... frames 3-31 ...]
Video frame 32/32 (t = 4.95 s, frame 100 of 100):
[image: PNG, data URL]
```

#### VDM requests, `vdm_text` (part 2 of a 102-frame video; images and middle frames shortened)

```text
=== role: system ===
You describe a robot task's demonstration video for another model that cannot see it. The video is shown to you in consecutive parts. That model will control the robot afterwards and will only have your text, so it must learn from you everything the video shows that could matter for the task. Report only what is visible. Do not say what the answer is, which object is the target, what the robot should do next or what the subgoals are. Do *NOT* write any code.
=== role: user ===
The robot's task (for context only; do not solve it): <task goal>
Track every object individually through the frames: its identity (color and kind, e.g. 'red cube', 'blue cup', 'left button') and its position (relative to the other objects, and left / right / front / back as seen from this camera) in each frame. Objects may move without the robot touching them (containers can slide past each other and trade places), so compare each object's position frame by frame instead of describing the scene as a whole. Report events with their times (t = ... s): an object covered or uncovered, containers swapped (say which container went from where to where), an object lifted, carried, placed (where, and in which order relative to earlier placements), a highlight or light appearing or disappearing, a button pressed, a path traced by the gripper or a stick (its shape and waypoints in order). Count repeated actions. If something cannot be seen (occluded, too fast, between frames), say so instead of guessing. Do not infer the goal's answer, the target object or a plan.
The video is from the front camera (the 'agentview' camera) and has 102 frames (5.1 s at 20 frames per second). This is part 2 of 6: frames 17-36 (t = 0.80-1.75 s), every frame, in time order. Its first 4 frames are the last frames of part 1; events in them are already described.
State at the end of the previous part, as you described it. Continue from it and keep each object's identity consistent:
<state_so_far>
<STATE section of the part-1 reply>
</state_so_far>
Frame 17 of 102 (t = 0.80 s):
[image: PNG, data URL]
[... frames 18-35 ...]
Frame 36 of 102 (t = 1.75 s):
[image: PNG, data URL]
<the tracking instructions again>
Reply in exactly two sections:
EVENTS:
a numbered list of the events in this part, in time order, each with its time (t = ... s); write 'none' if nothing happens. Do not repeat events from the frames this part shares with the previous part.
STATE:
the state at the last frame of this part: each object with its identity and position; which container is over which object (or 'unknown'); objects placed so far and their order; highlights or buttons so far.
```

Part 1 has no `<state_so_far>` and no overlap sentence. With `condition_video_segment_frames: 0` the single request
is the first version: the 32 sampled frames with the instructions "Describe the video in time order, as a numbered
list of events ..." before and after them.

#### First prompt, `vdm_text`

```text
=== role: system ===
You are a helpful assistant that generates Python code to directly solve the task.
=== role: user ===
You are controlling a Franka Emika robot with API described below.
Goal: pick up the red cube twice, then press the button

APIs:
[... API documentation omitted ...]
Before you take control, this task showed a video from the front camera (the 'agentview' camera) of 102 frames (5.1 s at 20 frames per second). You cannot see it. A vision model watched 102 of its frames (every frame, in 6 overlapping parts, in time order) and described it as below. The video is described only here, once; the current scene is what get_observation() returns.
<conditioning_video_description>
Part 1/6 (t = 0.00-0.95 s):
<EVENTS of part 1, verbatim>

Part 2/6 (t = 0.80-1.75 s):
<EVENTS of part 2>
[... parts 3-5 ...]
Part 6/6 (t = 4.00-5.05 s):
<EVENTS of part 6>

Final state (end of the video):
<STATE of part 6, verbatim>
</conditioning_video_description>
```

(With `use_img_differencing`, the VDM's initial-scene description is appended to the first text item, as before.)

## Logs: `ensemble_calls`

With the ensemble on, every entry of `all_responses.json` that made coding queries (the initial entry and every
decision) has `ensemble_calls`, one record per ensemble query, including cut-off / empty retries and S5 re-asks:

```json
{"kind": "multimodel", "is_multiturn": true, "n_candidates": 9, "n_failed": 1,
 "candidates": [{"model": "openai/gpt-oss-120b", "temp": 0.1, "ok": true, "latency_s": 21.4, "finish_reason": "stop",
                 "usage": {...}, "tokens": {"prompt": 11234, "completion": 2101, "reasoning": 1650}},
                {"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "temp": 0.9, "ok": false, "latency_s": 600.0, "usage": null,
                 "tokens": {...}, "error": "..."}, ...],
 "synthesis": {"model": "deepseek-ai/DeepSeek-V4-Pro", "temp": 0.2, "ok": true, "latency_s": 35.0,
               "finish_reason": "stop", "usage": {...}, "tokens": {...}, "format_issue": null},
 "candidates_s": 41.2, "wall_s": 76.3}
```

If the synthesis call raises (HTTP error, timeout), the exception (type unchanged) carries `ensemble_log` with the
finished candidates and `synthesis.ok: false`; the trial keeps it, and if the trial ends there (wall timeout) the
timeout summary adds an `{"decision": "aborted", "ensemble_calls": [...], "vdm_usage": [...]}` entry so the calls
are still costed. If all candidates fail, `synthesis` is null. Mean VDM latency in the cost table counts the timed
calls only (the conditioning-video calls; turn VDM calls log usage without latency).

`format_issue` checks the synthesis reply against the synthesis prompt's OUTPUT FORMAT: initial code needs exactly
one closed fenced block (`no_fence`, `multiple_fences`, `unclosed_fence`, `empty`); decisions need a `REGENERATE`
line with one closed block or a `FINISH` line (`no_keyword`, `both_keywords`, `finish_with_code`, plus the code
issues). The harness's own decision parser (`decision_source`, `decision_flags`) is logged as before. Reply texts
are in `ensemble_candidates*.txt` / `ensemble_synthesis*.txt` in the trial directory, as upstream. With the ensemble
off there is no `ensemble_calls` key.

## Cost

Prices are in `env_configs/pricing/nebius.yaml` (USD per 1M input / output tokens, Nebius catalogue 2026-11-01):
gpt-oss-120b 0.15/0.6, Qwen3-235B-2507 0.2/0.6, DeepSeek-V4-Flash-0731 0.14/0.28, DeepSeek-V4-Pro 1.75/3.5,
Qwen3.8-27B 0.45/3, Kimi-K3 3/15, and the alternatives Kimi-K2.7-Code 0.95/4, GLM-5.3 1.4/4.4,
`nvidia/Nemotron-3-Ultra-550b-a55b` 1/3, Qwen3.5-397B-A17B 0.6/3.6 (text2text on Nebius; all of them are in
`TEXT_ONLY_MODELS`; the GLM ID `zai-org/GLM-5.3` is unconfirmed). `scripts/model_costs.py` sums each trial's calls per model (ensemble: every
candidate and synthesis call in `ensemble_calls`; no ensemble: `usage`, `truncated_first_usage`, `s5_first_reply`
under the coding model; `vdm_usage` under the VDM model; with `condition_video_model`, the conditioning-video calls
in `condition_video.vdm_calls` under that model, e.g. Kimi-K3 at 3/15) and prices them as prompt × input + completion × output
(reasoning tokens are completion tokens). Molmo (local) and cached-input discounts are not included; the Nebius
console is the authority.

- RoboMME: `scripts/analyze_robomme_pilot.py <run>` — `results.tsv` gains `cost_usd`, `ens_queries`,
  `ens_failed_candidates`, `synth_format_errors`; `SUMMARY.md` gains "Cost by model" (calls, failed calls, tokens,
  mean latency, cost, $/trial) and "Ensemble" (queries per trial, mean wall time, failed candidates, cut-off
  candidates, synthesis format error rate by issue). Coding tokens now include all ensemble calls.
- LIBERO-Mem: `scripts/analyze_libero_mem_pilot.py <run>` (the same columns plus `init_source`; success by task
  and by init source). The older `analyze_pilot.py` on the cluster volume is not in the repo; it can import
  `scripts/model_costs.py` (`trial_usage`, `cost_by_model`, `load_prices`) for the same numbers.
- `--prices`, `--coding-model`, `--vdm-model` override the pricing file and the model names of non-ensemble calls
  (default: `pilot_manifest.jsonl`).

## Configs and runner

`scripts/gen_capbg_ensemble_configs.py` writes `env_configs/capbg_ensemble/{libero_mem,robomme}/` (18 files) from
the baseline configs: LIBERO-Mem tasks 0–9 (`env_configs/capbg_baseline/libero_mem/`) and RoboMME
PickXtimes, StopCube, VideoUnmaskSwap, ButtonUnmask, PickHighlight, VideoPlaceOrder, MoveCube, PatternLock
(`env_configs/robomme/`). Only the header, `use_parallel_ensemble` / `use_multimodel` (true), `ensemble`,
`condition_video_mode: auto` (RoboMME) and `output_dir` differ (`tests/test_ensemble_nebius.py` checks this);
`memory_mode` stays `none`. `--check` exits 1 when a file is stale.

LIBERO-Mem covers all 10 tasks since 2026-11-02 (the pilot 20261101_ens2_mem5x5 ran tasks 0, 4, 6, 7, 8; those five
configs are unchanged), so `run_ensemble_pilot.py --bench libero_mem` without `--tasks` now runs tasks 0–9; pass
`--tasks 1 2 3 5 9` for only the rest (e.g. run 20261102_ens2_mem10x5_rest).

Candidates and the synthesis model are arguments; the choice is kept in `env_configs/capbg_ensemble/models.json`, so
later runs and `--check` reuse it:

```bash
python scripts/gen_capbg_ensemble_configs.py --replace deepseek-ai/DeepSeek-V4-Flash-0731=<new/model>  # keeps its temperatures
python scripts/gen_capbg_ensemble_configs.py --candidates "a/m1@0.1,0.5,0.9;b/m2@0.1,0.5,0.9" --synthesis-model c/m3
# the current conditioning-video choice (models.json "condition_video"):
python scripts/gen_capbg_ensemble_configs.py --condition-video-model moonshotai/Kimi-K3 \
  --condition-video-tasks VideoUnmaskSwap,VideoPlaceOrder,MoveCube,PatternLock \
  --condition-video-segment-frames 0 --condition-frames 32
python scripts/gen_capbg_ensemble_configs.py --condition-video-model none  # back to the VDM, part by part
```

The four video configs get `condition_video_model: moonshotai/Kimi-K3` and `condition_video_segment_frames: 0`
(`condition_frames` stays 32, the baseline's); the other nine are unchanged.

The configs' headers say which coding-side models are text-only (`TEXT_ONLY_MODELS`); add a new text-only model
there before using it. A pilot started on older configs keeps their settings in its `pilot_manifest.jsonl`.

`scripts/run_ensemble_pilot.py --bench libero_mem|robomme` runs one `launch.py` per task (one sequential worker,
`--workers` tasks at a time), resumes at the first trial without `trial_end.json` when rerun, starts the SAM3 /
GraspNet / pyroki servers once, and passes `--trial-timeout-s 3600 --max-trial-retries 1` (as the baseline pilots).
LIBERO-Mem trial k resets with seed k (same init states as the baseline pilots, trials 1–5); RoboMME trial k is test
episode k-1 (episodes 0–4). `--model` only names the output directory (default: the synthesis model). Outputs:
`outputs/capbg_ensemble/<run>/deepseek-ai_DeepSeek-V4-Pro/<config stem>/trial_*`, logs in
`logs/capbg_ensemble/<run>/`, settings and commands in `<run>/pilot_manifest.jsonl`.

## Running on kubectl

Prerequisites in the GPU pod (as for the baselines, `docs/capbg-baseline.md`, `docs/memory-benchmarks.md`):
`.venv-libero-mem` / `.venv-robomme` synced, NVIDIA Vulkan ICD for RoboMME, `NEBIUS_SERVER_URL` and
`NEBIUS_API_KEY` set, and the Molmo2-8B server running on :8122 from `.venv-molmo`:

```bash
kubectl exec -it <GPU_POD> -- bash -lc 'cd /workspace/cap-bg && git fetch origin && git checkout exp/ensemble-nebius &&
  nohup .venv-molmo/bin/python capx/serving/vllm_server.py --model allenai/Molmo2-8B --port 8122 \
    --max-model-len 16384 --max-num-batched-tokens 16384 > logs/molmo_8122.log 2>&1 &'
```

(0) Conditioning-video descriptions only (VDM, no simulation steps, no coding model; episode 0 of the four video
tasks, also the old single-call description for comparison):

```bash
kubectl exec -it <GPU_POD> -- bash -lc 'cd /workspace/cap-bg && git fetch origin && git checkout exp/ensemble-nebius &&
  git pull --ff-only origin exp/ensemble-nebius &&
  .venv-robomme/bin/python scripts/describe_condition_video.py --run-name 20261102_cvdesc_seg \
    --tasks VideoUnmaskSwap VideoPlaceOrder MoveCube PatternLock --episode 0 --also-single'
```

Read `outputs/condition_video_desc/20261102_cvdesc_seg/<Task>_ep0.md` (description, every part's raw reply, the old
single-call description) and `SUMMARY.txt` (calls, seconds, tokens, $ and cut-off parts per task). The script reports
each method separately: the part-by-part description and the single-call comparison. With `--segment-frames 0`
the single call is the description itself, as in the trial. The PNGs in `<Task>_ep0_frames/` are
the frames the VDM saw. Settings come from the configs; override them with `--stride`, `--segment-frames`,
`--overlap`, `--max-segments`.

(1) Smoke — LIBERO-Mem task_00 × 1 and RoboMME VideoUnmaskSwap × 1 (ensemble and `vdm_text` paths):

```bash
kubectl exec -it <GPU_POD> -- bash -lc 'cd /workspace/cap-bg &&
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-libero-mem/bin/python scripts/run_ensemble_pilot.py \
    --bench libero_mem --run-name 20261102_ens_smoke_mem --tasks 0 --trials 1 --workers 1 &&
  .venv-libero-mem/bin/python scripts/analyze_libero_mem_pilot.py outputs/capbg_ensemble/20261102_ens_smoke_mem &&
  .venv-robomme/bin/python scripts/run_ensemble_pilot.py \
    --bench robomme --run-name 20261102_ens_smoke_robomme --tasks VideoUnmaskSwap --trials 1 --workers 1 &&
  .venv-robomme/bin/python scripts/analyze_robomme_pilot.py outputs/capbg_ensemble/20261102_ens_smoke_robomme'
```

Check: `SUMMARY.md` "Ensemble" (calls per query = 9 + 1, failed candidates, format errors) and "Cost by model"
(calls / tokens / latency per model); in the RoboMME trial's `all_responses.json`, `condition_video.mode ==
"vdm_text"`, the `description`, and no `image_url` in `initial_prompt`.

(2) Pilot — LIBERO-Mem 5 tasks × 5, RoboMME 8 tasks × 5 (rerun the same commands to resume):

```bash
kubectl exec -it <GPU_POD> -- bash -lc 'cd /workspace/cap-bg &&
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-libero-mem/bin/python scripts/run_ensemble_pilot.py \
    --bench libero_mem --run-name 20261102_ens_mem5x5 --trials 5 --workers 2 > logs/20261102_ens_mem5x5.out 2>&1;
  .venv-robomme/bin/python scripts/run_ensemble_pilot.py \
    --bench robomme --run-name 20261102_ens_robomme8x5 --trials 5 --workers 2 > logs/20261102_ens_robomme8x5.out 2>&1'
```

(2b) The four video tasks with the Kimi-K3 description, into the same run as a separate process (finished trials
in the run are skipped; it reuses SAM3 / GraspNet / pyroki servers already running, but if it started them they stop
when it ends):

```bash
kubectl exec -it <GPU_POD> -- bash -lc 'cd /workspace/cap-bg && git fetch origin && git checkout exp/ensemble-nebius &&
  git pull --ff-only origin exp/ensemble-nebius &&
  nohup .venv-robomme/bin/python scripts/run_ensemble_pilot.py --bench robomme --run-name 20261101_ens2_robomme8x5 \
    --tasks VideoUnmaskSwap VideoPlaceOrder MoveCube PatternLock --trials 5 --workers 2 \
    > logs/20261101_ens2_robomme8x5_video.out 2>&1 &'
```

(3) Aggregate:

```bash
kubectl exec -it <GPU_POD> -- bash -lc 'cd /workspace/cap-bg &&
  .venv-libero-mem/bin/python scripts/analyze_libero_mem_pilot.py outputs/capbg_ensemble/20261102_ens_mem5x5 &&
  .venv-robomme/bin/python scripts/analyze_robomme_pilot.py outputs/capbg_ensemble/20261102_ens_robomme8x5'
```

### Expected calls

One ensemble query = 9 candidate calls + 1 synthesis call. A trial makes one query for the first code and one per
decision turn (at most `MULTITURN_LIMIT + 1` = 11 turns), plus one more per cut-off / empty retry. The VDM makes one
initial-scene call and one call per turn. For the RoboMME video tasks (VideoUnmaskSwap, VideoPlaceOrder, MoveCube,
PatternLock) it also makes the conditioning-video calls: one per part, at most 12 per trial, e.g. 6 for a 102-frame
video. The LIBERO-Mem baseline averaged about 6 coding replies per trial.

| Stage | Trials | Coding calls (≈6 queries/trial; max 12) | VDM calls |
| --- | --- | --- | --- |
| (0) descriptions only | – | 0 | ≤ 4 × 12 (+4 with `--also-single`) |
| (1) smoke | 2 | ≈120 (20–240) | ≈18 |
| (2) pilot LIBERO-Mem | 25 | ≈1,500 (250–3,000) | ≈150 |
| (2) pilot RoboMME | 40 | ≈2,400 (400–4,800) | ≈240 + ≤ 240 conditioning (20 trials × ≤ 12) |
| (3) aggregate | – | 0 | 0 |

Rough cost from the baseline's per-reply sizes (≈11k prompt tokens): ≈$0.09 per ensemble query (the DeepSeek-V4-Pro
synthesis is about two thirds of it), ≈$0.55 per trial, ≈$35 for the pilot (up to ≈$70 at the turn limit), plus a
few dollars of VDM (each conditioning-video part sends 20 images). The aggregators report the measured numbers.

## Known limits

- `TEXT_ONLY_MODELS` is a fixed list; a new text-only model must be added there (unknown models are assumed to take
  images).
- The VDM description replaces the frames. The coding models cannot recover anything the VDM misses or gets wrong.
  The part-by-part version is untested against the real VDM; check it with (0) before the pilot. Each part sees only
  its own frames plus the previous state, so an event that spans a boundary relies on the overlap frames.
- The Kimi-K3 description is one request with 32 images (`condition_frames`); a long video is seen only through
  those 32 sampled frames.
- 20 images per VDM request in the part-by-part mode (the single call sends 32). Lower `condition_video_segment_frames` if the endpoint limits
  images, and report different settings as different conditions.
- Candidates cut off at `max_tokens` (or with empty content) still count as successful and go into the synthesis
  (upstream semantics); they are counted in the summary. A cut-off synthesis re-runs the whole ensemble once.
- 9 parallel requests per query × `--workers` tasks hit one Nebius endpoint; rate limits are untested. Failed
  candidates are logged and counted, and 404/5xx are retried with a long backoff (`_post_chat`), which adds latency
  inside the 3600 s trial budget.
- `reasoning_effort` is not sent to the Nebius models (only to `GPT_MODELS`, as upstream); gpt-oss-120b uses the
  server default.
- Costs are catalogue estimates from reported usage (reasoning tokens only if the API reports
  `completion_tokens_details`).
