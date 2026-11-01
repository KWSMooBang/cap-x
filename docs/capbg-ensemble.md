# cap-bg Nebius ensemble (CaP-Agent0-style, text-only models)

The baseline CaP-X loop with the CaP-Agent0 **multimodel parallel ensemble** on, using Nebius models:

| Role | Model | Temperatures |
| --- | --- | --- |
| Candidates (9 replies per query) | `openai/gpt-oss-120b`, `Qwen/Qwen3-235B-A22B-Instruct-2507`, `moonshotai/Kimi-K2.7-Code` | 0.1, 0.5, 0.9 each |
| Synthesis | `deepseek-ai/DeepSeek-V4-Pro` | 0.2 (upstream) |
| VDM | `Qwen/Qwen3.8-27B` | as the baseline |

All four coding-side models are **text-only**. RoboMME's conditioning video therefore goes into the first prompt as
a VDM description (`condition_video_mode: auto` → `vdm_text`, below).

## Configuration

| Key (YAML) | CLI | Default (unset) |
| --- | --- | --- |
| `ensemble.candidates` (list of `{model, temperatures}`) | `--ensemble-candidates "m1@0.1,0.5,0.9;m2@0.5"` | upstream `ENSEMBLE_CONFIGS` (`openai/gpt-5.4` × 0.1/0.5/0.9) |
| `ensemble.synthesis_model` | `--ensemble-synthesis-model M` | upstream `openai/gpt-5.4` |
| `condition_video_mode`: `images` \| `vdm_text` \| `auto` | `--condition-video-mode` | `images` |

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
- `vdm_text`: after reset, the VDM gets the 32 frames once (one request) and describes them in time order: what moved
  where, which container covers what and every swap, counts and order, path shapes, what appears or disappears, the
  last frame, and what is unclear. It is told not to state the answer, the target or a plan. The description replaces
  the images in the first prompt. An empty or cut-off description is asked once more; empty twice fails the trial.
- `auto`: `vdm_text` if any coding-side model (every candidate and the synthesis model, or the single coding model
  without the ensemble) is text-only, else `images`.

Logged in the initial entry of `all_responses.json` as `condition_video`: `mode`, `config_mode`, `n_frames`,
`total_frames`, `frame_indices`, and for `vdm_text` the verbatim `description`, `vdm_model`, `vdm_calls`
(latency, finish_reason, usage) and `vdm_reasoning`. Its usage is also in `vdm_usage`. `initial_prompt` in the same
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

#### VDM request, `vdm_text`

```text
=== role: system ===
You describe a robot task's demonstration video for another model that cannot see it. That model will control the robot afterwards and will only have your text, so it must learn from you everything the video shows that could matter for the task. Report only what is visible. Do not say what the answer is, which object is the target, what the robot should do next or what the subgoals are. Do *NOT* write any code.
=== role: user ===
The robot's task (for context only; do not solve it): <task goal>
Describe the video in time order, as a numbered list of events with their times (t = ... s). For each event say which object moved (color, shape, size, any marking, and which of several similar objects it is), from where to where (relative to the other objects, and left / right / front / back as seen from this camera), and what touched or moved it. Track hidden objects: say which container or cover is over which object, every time one is lifted, put down or swapped, and where each container is at the end. Count repeated actions and give their order. For movements that trace a path (a stick, a button press sequence, a route), describe the path's shape and its waypoints in order. Mention highlights, lights or anything that appears or disappears, and when. Then describe the last frame: where every object is. If something is unclear (occluded, between sampled frames, too small), say so instead of guessing. Do not infer the goal's answer, the target object or a plan; describe only what happened.
The video is from the front camera (the 'agentview' camera) and has 100 frames (5.0 s at 20 frames per second). Below are 32 frames sampled uniformly from it in time order, including the first and last frame.
Video frame 1/32 (t = 0.00 s, frame 1 of 100):
[image: PNG, data URL]
[... frames 2-31 ...]
Video frame 32/32 (t = 4.95 s, frame 100 of 100):
[image: PNG, data URL]
<the same instructions again, after the frames>
```

#### First prompt, `vdm_text`

```text
=== role: system ===
You are a helpful assistant that generates Python code to directly solve the task.
=== role: user ===
You are controlling a Franka Emika robot with API described below.
Goal: pick up the red cube twice, then press the button

APIs:
[... API documentation omitted ...]
Before you take control, this task showed a video from the front camera (the 'agentview' camera) of 100 frames (5.0 s at 20 frames per second). You cannot see it. A vision model watched 32 frames sampled uniformly from it in time order (including the first and last frame) and described it as below. The video is described only here, once; the current scene is what get_observation() returns.
<conditioning_video_description>
<VDM description, verbatim>
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
                {"model": "moonshotai/Kimi-K2.7-Code", "temp": 0.9, "ok": false, "latency_s": 600.0, "usage": null,
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
gpt-oss-120b 0.15/0.6, Qwen3-235B-2507 0.2/0.6, Kimi-K2.7-Code 0.95/4, DeepSeek-V4-Pro 1.75/3.5,
Qwen3.8-27B 0.45/3, Kimi-K3 3/15. `scripts/model_costs.py` sums each trial's calls per model (ensemble: every
candidate and synthesis call in `ensemble_calls`; no ensemble: `usage`, `truncated_first_usage`, `s5_first_reply`
under the coding model; `vdm_usage` under the VDM model) and prices them as prompt × input + completion × output
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

`scripts/gen_capbg_ensemble_configs.py` writes `env_configs/capbg_ensemble/{libero_mem,robomme}/` (13 files) from
the baseline configs: LIBERO-Mem tasks 0, 4, 6, 7, 8 (`env_configs/capbg_baseline/libero_mem/`) and RoboMME
PickXtimes, StopCube, VideoUnmaskSwap, ButtonUnmask, PickHighlight, VideoPlaceOrder, MoveCube, PatternLock
(`env_configs/robomme/`). Only the header, `use_parallel_ensemble` / `use_multimodel` (true), `ensemble`,
`condition_video_mode: auto` (RoboMME) and `output_dir` differ (`tests/test_ensemble_nebius.py` checks this);
`memory_mode` stays `none`. `--check` exits 1 when a file is stale.

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

(3) Aggregate:

```bash
kubectl exec -it <GPU_POD> -- bash -lc 'cd /workspace/cap-bg &&
  .venv-libero-mem/bin/python scripts/analyze_libero_mem_pilot.py outputs/capbg_ensemble/20261102_ens_mem5x5 &&
  .venv-robomme/bin/python scripts/analyze_robomme_pilot.py outputs/capbg_ensemble/20261102_ens_robomme8x5'
```

### Expected calls

One ensemble query = 9 candidate calls + 1 synthesis call. A trial makes one query for the first code and one per
decision turn (at most `MULTITURN_LIMIT + 1` = 11 turns), plus one more per cut-off / empty retry. The VDM makes one
initial-scene call, one call per turn, and for the RoboMME video tasks (VideoUnmaskSwap, VideoPlaceOrder, MoveCube,
PatternLock) one conditioning-video call. The LIBERO-Mem baseline averaged about 6 coding replies per trial.

| Stage | Trials | Coding calls (≈6 queries/trial; max 12) | VDM calls |
| --- | --- | --- | --- |
| (1) smoke | 2 | ≈120 (20–240) | ≈13 |
| (2) pilot LIBERO-Mem | 25 | ≈1,500 (250–3,000) | ≈150 |
| (2) pilot RoboMME | 40 | ≈2,400 (400–4,800) | ≈260 (20 conditioning) |
| (3) aggregate | – | 0 | 0 |

Rough cost from the baseline's per-reply sizes (≈11k prompt tokens): ≈$0.15 per ensemble query (Kimi-K2.7-Code
candidates and the DeepSeek synthesis are most of it), ≈$1 per trial, ≈$60 for the pilot (up to ≈$120 at the turn
limit). The aggregators report the measured numbers.

## Known limits

- `TEXT_ONLY_MODELS` is a fixed list; a new text-only model must be added there (unknown models are assumed to take
  images).
- The VDM description replaces the 32 frames; what the VDM misses (or gets wrong) the coding models cannot recover.
  Its quality is not measured yet (no API on the dev server); check the descriptions in the smoke run.
- 32 images in one VDM request: if Qwen3.8-27B rejects that many, use `CAPX_ROBOMME_CONDITION_FRAMES=16` (and report
  it as a different condition).
- Candidates cut off at `max_tokens` (or with empty content) still count as successful and go into the synthesis
  (upstream semantics); they are counted in the summary. A cut-off synthesis re-runs the whole ensemble once.
- 9 parallel requests per query × `--workers` tasks hit one Nebius endpoint; rate limits are untested. Failed
  candidates are logged and counted, and 404/5xx are retried with a long backoff (`_post_chat`), which adds latency
  inside the 3600 s trial budget.
- `reasoning_effort` is not sent to the Nebius models (only to `GPT_MODELS`, as upstream); gpt-oss-120b uses the
  server default.
- Costs are catalogue estimates from reported usage (reasoning tokens only if the API reports
  `completion_tokens_details`).
