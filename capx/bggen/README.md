# capx.bggen — Agent as BG Generator (Belief Graph Ver3)

Benchmark-independent Belief Graph generator. Input: a demonstration's video + robot state + instruction.
Output: an event-level Belief Graph trace and Ver1-format BG-CoT labels.

- **Mode A (offline, labels):** forward pass + backward pass → `cot_rows.jsonl`.
- **Mode B (online):** the forward pass alone (past frames only) → `cot_rows_forward.jsonl`. This is the same logic as an explicit BG at inference time.

Design: `/outputs/cap-bg/tasks/agent-bg-design.md` (artifact 「Agent as BG Generator」, design and example tabs).

Branch `exp/agent-bg-gen` only: this code is not merged into bg or main. `capx/belief` (bg_est2) is imported, never modified.

## Modules

| module | role |
|---|---|
| `records.py` | `FrameRecord` / `CameraFrame` / `ArmState`; the on-disk episode format; `Episode` (causal image access; refuses `oracle` / `truth` / `task_info` attributes) |
| `adapters/libero_mem.py`, `adapters/b1k.py` | benchmark data → `<root>/episode/` + `<root>/oracle/` (download only the needed files, dump the structure, prepare) |
| `events.py` | per-arm gripper events, base start / stop (debounced), keyframes every `keyframe_s`. A close is `grasp` / `close_empty` when the robot's fully-closed width is known (B1K). Otherwise it is `close`: the generator holds the nearest track provisionally and confirms or retracts it by co-motion (LIBERO-Mem: a bowl-rim grasp closes to 4–6 mm, the same as an empty hand). |
| `perception.py` | SAM3 per category on any camera; depth path lifts masks to 3D (as bg_est2); 2D path keeps masks only. `FormatFakeDetector` (fixed boxes, smoke only) |
| `tracks.py` | association, aliases from the image left, held objects follow the EEF, names from events (release place → last held object; late merges recorded); identity before a new name: reacquire an unmatched known track (`reacquire_radius`), bind an expected object, closed world / `fixed_closed` / `max_instances` (a new instance needs more detections of the category in one frame than known tracks); large objects associate by box gap (`large_extent`); `counts` → report |
| `vocab.py` | edge keys / statement text (`(on a b)`, `(inhand_left x)`, `(handempty)`); categories = bg_est2 `VOCABULARY_V2` + B1K (hot dog, refrigerator, microwave, countertop) with prompt candidates |
| `geometry.py` | depth path: d_xy, support radius (median over frames), dz, container `inside`; 2D path: mask overlap / bottom contact + release-time EEF |
| `verifier.py` | the only writer of P (rules below) |
| `graph.py` | edges (P, source obs/robot/exp/mem/conflict), goal branches of newly-true steps, causal progress |
| `prompts.py` | init (JSON remember / goal_branches / observe), agent (tool calls), observe = bg_est2 M5 format with its ON_RULE ("on = resting flat") |
| `agent.py` | reply → plan; format failure → default plan (objects around the event + re-checks + next goal relations), reason logged |
| `models.py` | one OpenAI-compatible client, roles init / agent / vdm, URL / key / model from env only, usage counted (incl. reasoning tokens, cut / empty replies). Output limit 32768 per role (`GenConfig.max_tokens`, `BGGEN_<ROLE>_MAX_TOKENS`); a cut (`finish_reason` length) or empty reply is asked once more with twice the limit; every attempt's finish_reason / reasoning size goes to the trace |
| `generator.py` | forward pass: init, then per event robot state → detect/track → marked image → agent → tools → verifier → row + trace |
| `backward.py` | mode A refinements (allowed / forbidden list below) |
| `serialize.py` | Ver1 lines, event rows + 1 Hz expansion, B1K renames on/in → ontop/inside |
| `pipeline.py`, `cli.py` | run and write outputs; the command line |
| `oracle.py`, `evaluate.py` | evaluation only. The generator modules never import these, and never open `<root>/oracle/` (tested) |
| `fake_models.py` | FAKE OpenAI-compatible server (scripted replies) for smoke runs and tests |

## Verifier rules (fixed; the agent never writes a confidence)

| rule | what it does |
|---|---|
| R1 | Observe 1 / 0 → P 0.9 / 0.1 (obs) |
| R2 | ? → previous value kept (source mem) |
| R3 | robot state 0.97 / 0.03 (inhand, handempty, lifted) wins over Observe. A grasped object rests on nothing (its on / in edges → 0.03 robot while held). |
| R4 | agent `expect` → 0.7 / 0.3 (exp), only where nothing was observed now |
| R5 | geometric veto: Observe 1 for on / in with both objects detected now, depth path. Not flat (d_xy > r + 0.5 cm, or dz > the pair's flat dz + 1 cm; flat dz from bg_est2's measured table, else the first accepted pair) → 0.5 conflict + S4. For a container, a center outside its box → 0.5 conflict + S4. |
| R6 | a state predicate (cooked, toggled_on) of an object inside a closed container is not observable → treated as ? |
| R7 | **new:** relations change only when the robot touches them. A contradicting Observe whose objects were not touched since the key's last observation is held; it is re-asked at the next event / keyframe (forced into the plan) and applied only after 2 agreeing answers. `VerifierConfig(touch_rule=False)` / `--no-touch-rule` turns it off. |
| R8 | process model (B1K): in(X, C) and toggled_on(C) for `cook_s` → cooked(X) 0.7 mem (never obs) |

Marks are drawn only for tracks detected in the current image. A non-held object at the fingertip pixel is not marked (the bg_est2 rule).

## Backward pass (mode A)

| | rule |
|---|---|
| allowed | B1 track-ID unification (tracker merges) |
| allowed | B2 single-shot noise: a 0 / 1 that disagrees with both neighbouring 0 / 1 answers (which agree each other), with no robot contact in between. Hindsight answers count as neighbours. Rows in between where the key was answered ? keep their value. |
| allowed | B3 Effect = state difference from segment start to the row where the end state is confirmed; attached to that row only |
| allowed | B4 Subtask = the segment's intent from the approach on: pick / place / put_in / open / close / toggle_on / move_to; "done" only from the row where the goal was confirmed. A row's Subtask holds from its frame on: pick → place at the close event row, the next pick from the release row; the confirming row (Effect) never moves a boundary |
| forbidden (checked by `check_forbidden`, logged in `backward_log.json`) | filling ? from later frames |
| forbidden | changing Delta / progress |
| forbidden | writing a hidden process state as obs |

The forward rows are kept unchanged (`rows_forward.jsonl`, `cot_rows_forward.jsonl`) for comparison.

## Outputs (`<out>/`)

| file | content |
|---|---|
| `trace.jsonl` | per event: graph, goal, agent prompt (text) and raw replies, parsed plan / fallback reason, tool results (observe raw reply + parse, geometry, detect, look, hindsight), verifier decisions, tracks, pending re-checks |
| `rows_forward.jsonl`, `rows_backward.jsonl` | machine-readable rows |
| `cot_rows_forward.jsonl`, `cot_rows.jsonl` | Ver1 lines (`kind: event` rows and `kind: hz` 1 Hz rows) |
| `keyframes/` | marked images |
| `backward_log.json` | backward decisions |
| `summary.json`, `report.md` | events, model calls / tokens per role, fallbacks, verifier rule counts, goal, final graph, labels |
| `eval.json`, `eval.md` | written by `cli eval`: forward vs forward+backward. Oracle object ↔ track map over the whole episode (3D, `--map-radius`, default 0.15 m / B1K 0.6 m; single-instance categories map every track). Relation accuracy is scored with ONE one-to-one map per episode (Hungarian on matched frames, the last observation weighted to the episode end; a never-seen alias only by elimination), so a name swap is an error and a decided prediction on an alias without an object counts wrong (`unmappable_wrong`); the per-row re-matched accuracy is an auxiliary column, and every re-match is a name-consistency error. Event tolerance ±`--tol` frames (default 5 / B1K 15), accuracy per relation kind, articulated grasps counted apart (only the arm the row believes holds the door / switch, for its inhand of that object and its handempty, inside those segments), boundary error vs `intent` (segment starts) and `contact` (oracle inhand changes) |

## Data (what the adapters read)

The episode format and the oracle format are documented in `records.py` and `oracle.py`.

### LIBERO-Mem: `libero-mem/LIBERO-Mem-Raw`

Swap task file: `KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5` (48.8 MB).

- **File ↔ task (checked 2026-11-01):** HF files are numbered 1–10, and file k+1 = cap-x `task_id` k (`TASK_NAMES`). `prepare` aborts if the file's `bddl_file_name` is another task.
- **README:** says only "initial states, proprioception, control actions".
- **Layout (measured on kubectl 1a):** demos start at `demo_1` (120 demos). Each holds only `actions` (N, 7), `init_state` (90,) and `states` (N, 90); there is no `obs/`. `states` = robosuite [time, qpos …], so the fingers are `states[:, 8:10]` (q0 0.0021–0.040, q1 −0.040 to −0.0021). The default demo is the first one in the file.

Replay:

- With `states`: `set_init_state(states[t])` (exact).
- Without: init state + actions through OSC_POSE (may drift; the prep JSON reports the EEF drift).

Fields:

- **Camera:** cap-x `get_observation()` after each replayed state — agentview RGB, metric depth, K, camera pose in the robot base frame. `--wrist` adds eye-in-hand.
- **Proprio:**
  - EEF: HDF5 `obs/ee_pos` if present, else cap-x `robot_cartesian_pos` + bg_est2's TCP offset.
  - Fingers: `obs/gripper_states`, else the replayed sim qpos[7:9] (= `states[:, 8:10]`). Width = |q0| + |q1| = q0 − q1, 0.0042–0.080 m.
- **Success:** the BDDL goal's final relations evaluated on the last state (`goal_success`). The LIBERO-Mem subgoal tracker does not advance under state replay; its value is kept as `tracker_success`.
- **Oracle:** `OracleEstimator` keys evaluated by `LiberoMemSource` (on / in / inhand), positions, subgoal tracker, `task_success`, grasp intervals.

### B1K: `behavior-1k/2025-challenge-demos`, task-0045 episode_00450010

The source tree is LeRobot v2.1 at 30 fps, robot R1Pro, 6805 frames. Annotations are valid on frames 0–6645.

Confirmed from `meta/info.json`, the episode meta, and BEHAVIOR-1K v3.7.2 source:

- **Videos:**
  - rgb head 720², rgb wrists 480², hevc.
  - depth head / wrists: yuv420p16le, log-quantized 14-bit. `d = exp(q/16383·(ln 13.5 − ln 3.5) + ln 3.5) − 3.5` m.
- **Intrinsics:**
  - head: fx = fy = 306, c = 360.
  - wrists: fx = fy = 388.66, c = 240.
- **`observation.state[256]`:**
  - eef_left 186:189 / quat 189:193 (xyzw, base frame); gripper_left 193:195.
  - eef_right 225:228 / 228:232; gripper_right 232:234.
  - Fingers are 0–0.05 m each; width = sum.
  - robot_pos 140:143 and robot_2d_ori 149: global odometry → world ← base.
- **`observation.cam_rel_poses[21]`:** 3 × [pos, quat xyzw], camera → base. OpenGL camera rotated 180° about x → OpenCV.
- **Instruction:** `meta/tasks.jsonl` task 45: "Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked."
- **Oracle only:**
  - `observation.task_info` follows episode meta `task_obs_keys` (agent; microwave; hotdog ×2; electric_refrigerator; floor). Each object has real / pos / ori_cos / ori_sin / in_gripper_left / in_gripper_right.
  - `annotations/` `skill_annotation`: 13 skills — move to, open door, pick up from ×2, close door, move to, place on, open door, place in, pick up from, place in, close door, turn on switch.
  - Not read: seg_instance_id videos.

Still ASSUMPTIONS (`prepare` / `check-episode` print a check for each):

- Camera order in cam_rel_poses is left_wrist, right_wrist, head. Checks: head 1–2 m above the base, looking forward.
- task_info field sizes are 1 / 3 / 3 / 3 / 1 / 1. Checked against the vector length; positions and inhand are off on a mismatch.
- `MIN_DEPTH` is 0 (v3.7.2); later code uses 0.01.

Oracle relations:

- in / on / open / toggled_on come from the annotation timeline. They change at segment ends, so they are approximate.
- inhand: the task_info `in_gripper` flags are −1 throughout in the 2025 demos (kubectl 1b), so it comes from co-motion. An object within 12 cm of the EEF that moves with it holds; holding ends when they stop moving together (`comotion_inhand`). Doors are not covered.
- Annotation ↔ task_info names: matched by when each object starts to move, during its pick segment.
- Positions are world-frame from task_info. Annotation names (hotdog_207) are mapped to task_info names (hotdog.n.02_1) through in_gripper during each pick.

## Running

Local tests (CPU, synthetic fixtures, fake detector, fake HTTP models):

```bash
python -m pytest tests/test_bggen*.py
```

kubectl, over the episode list `scripts/bggen/episodes.yaml` (LIBERO-Mem tasks 0 / 2 / 6 / 7 / 8 x demo_1..3, B1K task 45 x
episodes 00450010 / 20 / 30 / 40 / 50):

```bash
bash scripts/bggen/kubectl_runbook.sh list|1a|1b|2|3|4
RUN_TAG=batch1 JOBS=3 bash scripts/bggen/kubectl_runbook.sh 3   # runs/batch1/<episode id>/
ONLY=t08 bash scripts/bggen/kubectl_runbook.sh 1a               # a subset (id / group substrings, comma list)
```

- `capx/bggen/batch.py` drives prepare / run / eval per episode as subprocesses. Prepared episodes and finished runs are
  skipped (`FORCE=1` redoes them). A failing episode is recorded (`episodes/prep_status.json`,
  `runs/<RUN_TAG>/batch_status.json`, `logs/batch/<step>_<id>.log`) and the next one starts. A B1K data check that
  fails marks the episode `check_failed`, and step 3 skips it (`ALLOW_CHECK_FAIL=1` runs it anyway).
- `scripts/bggen/aggregate.py runs/<RUN_TAG>` (step 4) writes `aggregate.{md,json}`. It groups by task, benchmark and all
  episodes. Pass-dependent metrics get a forward line and a forward+backward line. Values are the mean over episodes with
  [min, max]. Cost uses `--price-in / --price-out` $ per 1M tokens (default 0.45 / 3).

What each step checks is in the script header. Outputs go under `/workspace/cap-bg/outputs/agent-bg-gen/`.

LIBERO-Mem task ↔ file (checked 2026-11-01):

- `task_id` k is HF file `KITCHEN_SCENE1_<k+1>_..._demo.hdf5` (`adapters/libero_mem.TASK_NAMES`).
- `prepare` checks the file's `data@bddl_file_name` against the task. The episode instruction is the env's BDDL language.
- Task 2's file stores "pick up the bowl and place it back on the plate" in `problem_info` (no "3 times"). That string
  is kept as `instruction_file`.

Table regions (task 8 "... place that basket in the center"):

- `at(X, center)` is a goal / observe relation. `statement_key` accepts the region name, and `(on X center)` becomes
  `(at X center)`.
- The region comes from the task definition (oracle meta `regions` from the BDDL On targets + goal, else the goal's final relations in `final.json`), never from whether it was reached. Eval scores `at` against the oracle's `on X
  kitchen_table_the_center`.
- If the init goal has no `at` step while the instruction names the region, `report.md` and `eval.md` say
  REGION MISSING FROM THE GOAL.
- `at` is judged by the VDM only: there is no geometry check for regions.

Models:

- **Defaults:** init / agent `moonshotai/Kimi-K3`, vdm `Qwen/Qwen3.8-27B`.
- **Overrides:** `BGGEN_<ROLE>_URL` / `_MODEL` / `_API_KEY` / `_MAX_TOKENS`. Otherwise `NEBIUS_SERVER_URL` / `NEBIUS_API_KEY`.
- **SAM3:** `BGGEN_SAM3_URL` (default `http://127.0.0.1:8114`); the client is `perception.sam3_http_client` (no `capx.integrations` / `capx.envs` import).
- **vLLM:** `BGGEN_AGENT_URL=http://host:8000/v1`.

## Known limits

- **bggen-fix2 settings not yet measured on the real episodes:** LIBERO `reacquire_radius` 0.12 m + closed world + `fixed_closed`; B1K `reacquire_radius` 0.25 m, one countertop / fridge / microwave (`max_instances`, an ASSUMPTION for this kitchen), `large_extent` 0.4 m; eval map radius 0.15 / 0.6 m. A closed-world assignment of a SAM3 false positive moves a track to the wrong place (counted in `closed_assigned`).
- **Expected objects** (named by the task / init, not detected at t=0) have no position until bound; the first unexplained detection of the category is bound to them, in alphabetical order (two hot dogs first seen together may get swapped names; events name them afterwards).
- **Init failure** (no goal) is reported (summary `init_failed`, report banner, CLI stderr, runbook step 3 exit 3), but the run still completes with the default initial statements.

- **Thresholds still unmeasured:**
  - LIBERO-Mem fully-closed width: `closed_width=None`, so co-motion decides.
  - B1K SAM3 prompt candidates.
  - B1K flat dz (starts empty).
- **Measured on kubectl:**
  - B1K gripper events by width (right 3/3, left 2/2).
  - Base debounce: stop ≥ 1 s, start-to-stop distance ≥ 0.05 m; `check-episode` prints raw vs debounced counts.
- **Provisional grasps (`Generator._grasp_evidence`):** a `close` stays provisional until clear evidence arrives. Each event while the hand is closed compares two hypotheses with this image's detections of the category:
  - **Moved with the hand:** the expected position = the EEF + the offset at the close.
  - **Stayed:** where the object was at the close.
  - **Where:** in 3D, or in pixels for tracks without depth (2D path). The two places must be ≥ 2 cm / 20 px apart.
  - **2D path and camera pose:** pixels are compared only when the camera pose is unchanged since the close (≤ 1 cm, ≤ 1°). Otherwise the verdict is `camera_moved`; with an unknown pose on a mobile robot, `camera_unknown`. Both defer.
  - **Rules:**
    - Detected at the hand and not at the old place → confirmed.
    - At the old place and not at the hand → retracted. The object's relations from before the close are restored.
    - Both → ambiguous (two same-looking bowls), not confirmed.
    - A detection that another known track of the category explains is not evidence.
    - No detection of the category → cannot tell (occlusion / miss).
    - Category seen elsewhere, nothing at either place, hand lifted ≥ 3 cm and no longer above the old place → confirmed_lifted.
  - **Undecided at the release:** checked once more with the release image. Object at the release place → confirmed; at the old place → retracted, and the release lets go of nothing.
  - **Still undecided:** UNVERIFIED. The close and release rows are flagged; the backward pass labels the segment `maybe_pick` / `maybe_place` and writes no Effect. `report.md` lists every check (`grasp_checks`: result, evidence, where decided).
  - **Articulated targets** (fridge, microwave) are accepted without the test, because a door moves on an arc while the body stays.
- **B1K hot dogs disappear inside closed doors.** The memory value is kept (? → mem).
- **Detection on a moving head camera** uses 3D world association only (no appearance embedding yet). Objects out of view keep their last state.
- **robot_rel is always computed.** The agent's choice covers observe / geometry / detect / look / hindsight.
- **Close_empty segments:** a press (e.g. the microwave switch) becomes a segment only if a track is within `grasp_radius` at the close.
- **B1K relation oracle** is annotation-derived (transitions at segment ends). Comparison with Ver1 GT-CoT rows is not implemented here.
- **Mode A vs mode B forward rows:** the mode-A forward rows equal a mode-B run except that the agent may also call hindsight. Hindsight never changes the forward graph.
