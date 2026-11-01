#!/usr/bin/env bash
# Agent as BG Generator (capx.bggen) on kubectl, over the episode list scripts/bggen/episodes.yaml
# (LIBERO-Mem tasks 0 / 2 / 6 / 7 / 8 x demo_1..3, B1K task 45 x 5 episodes; ONLY=... narrows it).
#
#   bash scripts/bggen/kubectl_runbook.sh <step>     step = list 1a 1b 2 3 4 (in this order), or "all"
#
#   list  the episodes, whether each is prepared and whether RUN_TAG has a run of it.
#   1a  data, LIBERO-Mem (CPU ok; needs the LIBERO-Mem venv PY_LM and a renderer): per task download that task's HDF5
#       only, dump its structure (data/libero_mem/structure_tNN.json), then per demo replay + render (agentview + wrist
#       RGB-D, K, camera pose, proprio) and the simulator oracle, then check-episode. Prepared episodes are skipped
#       (FORCE=1 redoes them). prepare checks the HDF5's bddl_file_name against the task_id and uses the BDDL language.
#   1b  data, B1K (CPU): per episode download its parquet + head / wrist rgb / depth videos + meta + annotation only,
#       decode at 5 Hz, write the oracle, check-episode. A false data check (head_height_1_2m, head_looks_forward_down,
#       task_info_layout_ok) marks the episode check_failed: step 3 does not run it (ALLOW_CHECK_FAIL=1 to run anyway).
#       Per-episode results: episodes/prep_status.json; command output: logs/batch/{download,prep,check}_<id>.log.
#   2   smoke with FAKE models (scripted local server) + FAKE detector (fixed boxes) on SMOKE_EPS: output format only.
#       CHECKS: trace.jsonl / cot_rows.jsonl / report.md / summary.json exist, agent fallbacks > 0 (bad_every=5),
#       backward forbidden_violations == [].
#   3   real models + SAM3 (GPU 1): starts the SAM3 server, runs mode A on every prepared episode into
#       runs/${RUN_TAG}/<id>/ (JOBS episodes at once, default 1) and evaluates each. A run that already has
#       summary.json is skipped (FORCE=1 reruns). A failing episode is recorded (runs/${RUN_TAG}/batch_status.json,
#       logs/batch/run_${RUN_TAG}_<id>.log) and the next one starts. Init failures (no goal) are recorded; after the
#       whole batch the step exits 3 if any episode's init failed, unless ALLOW_INIT_FAIL=1.
#       Needs NEBIUS_SERVER_URL / NEBIUS_API_KEY (or BGGEN_<ROLE>_URL / _MODEL for a vLLM server).
#   4   re-evaluate every run of RUN_TAG, then aggregate: runs/${RUN_TAG}/aggregate.{md,json} (per task / benchmark /
#       all, forward and forward+backward lines, mean [min, max]).
#
# Exit codes: a failing command of the driver itself fails the step (set -Eeuo pipefail; the ERR trap names the step
# and line). Per-episode failures do not: they are in the status files and in aggregate.md. Step 3 exits 3 after the
# whole batch when an init failed (no goal), unless ALLOW_INIT_FAIL=1.
set -Eeuo pipefail
STEP=${1:-}
trap 'rc=$?; echo "[runbook] step ${STEP} FAILED (exit ${rc}) at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${OUT:-/workspace/cap-bg/outputs/agent-bg-gen}
PY=${PY:-$REPO/.venv/bin/python}                       # main venv (uv sync): huggingface_hub, pandas, pyarrow, SAM3 client
PY_LM=${PY_LM:-$REPO/.venv-libero-mem/bin/python}      # LIBERO-Mem venv (docs/memory-benchmarks.md)
EPISODES=${EPISODES:-$REPO/scripts/bggen/episodes.yaml}
RUN_TAG=${RUN_TAG:-batch1}
JOBS=${JOBS:-1}
ONLY=${ONLY:-}                                         # e.g. ONLY=t08 or ONLY=b1k or ONLY=libero_mem_t06_demo_2
SMOKE_EPS=${SMOKE_EPS:-"libero_mem_t06_demo_1 libero_mem_t08_demo_1 b1k_t45_e00450010"}
cd "$REPO"
mkdir -p "$OUT"/{data,episodes,runs,logs}

batch() {  # batch <step> [--only X] ...
  local step=$1; shift
  local args=(--episodes "$EPISODES" --out "$OUT" --tag "$RUN_TAG" --py "$PY" --py-lm "$PY_LM")
  [ "${FORCE:-0}" = 1 ] && args+=(--force)
  [ "${ALLOW_CHECK_FAIL:-0}" = 1 ] && args+=(--allow-check-fail)
  $PY -m capx.bggen.batch "$step" "${args[@]}" "$@"
}

only() {  # the --only value: ONLY if set, else the step's default ($1)
  echo "${ONLY:-$1}"
}

step_list() { batch list --only "$(only '')"; }

step_1a() {
  # GPU node: EGL; CPU: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1
  MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} batch prep --only "$(only libero_mem)" | tee "$OUT/logs/batch_prep_libero.log"
}

step_1b() {
  batch prep --only "$(only b1k)" | tee "$OUT/logs/batch_prep_b1k.log"
}

step_2() {
  for ep in $SMOKE_EPS; do
    local root="$OUT/episodes/$ep" bench=libero_mem
    case "$ep" in b1k_*) bench=b1k ;; esac
    if [ ! -f "$root/oracle/states.jsonl" ]; then echo "[runbook] smoke: $ep not prepared, skipped"; continue; fi
    $PY -m capx.bggen.cli run --root "$root" --out "$OUT/runs/smoke/$ep" --benchmark "$bench" --mode A \
      --fake-models --detector format_fake > "$OUT/logs/smoke_$ep.json"
    ls "$OUT/runs/smoke/$ep"/{trace.jsonl,cot_rows.jsonl,cot_rows_forward.jsonl,rows_backward.jsonl,report.md,summary.json} >/dev/null
    $PY -c "import json; s=json.load(open('$OUT/runs/smoke/$ep/summary.json')); \
print('$ep', 'fallbacks', s['agent_fallbacks'], 'violations', s['backward']['forbidden_violations'])"
  done
}

SAM3_PORT=${SAM3_PORT:-8114}
export BGGEN_SAM3_URL=${BGGEN_SAM3_URL:-http://127.0.0.1:$SAM3_PORT}

sam3_up() {
  if ! curl -s -m 2 "$BGGEN_SAM3_URL" >/dev/null 2>&1; then
    nohup $PY -m capx.serving.launch_sam3_server --device cuda --port "$SAM3_PORT" --host 127.0.0.1 > "$OUT/logs/sam3.log" 2>&1 &
    for _ in $(seq 60); do curl -s -m 2 "$BGGEN_SAM3_URL" >/dev/null 2>&1 && break; sleep 5; done
  fi
  # never run on with a dead detector (every detection would be empty)
  curl -s -m 2 "$BGGEN_SAM3_URL" >/dev/null 2>&1 || { echo "[runbook] SAM3 server not up at $BGGEN_SAM3_URL (log: $OUT/logs/sam3.log)" >&2; return 1; }
}

step_3() {
  : "${NEBIUS_SERVER_URL:?set NEBIUS_SERVER_URL (or BGGEN_<ROLE>_URL)}"
  sam3_up
  batch run --only "$(only '')" --jobs "$JOBS" | tee "$OUT/logs/batch_run_${RUN_TAG}.log"
  # every episode has run; an init failure (no goal) is loud here, as in the single-episode runbook
  local bad
  bad=$($PY - "$OUT/runs/$RUN_TAG/batch_status.json" <<'PY'
import json, sys
st = json.load(open(sys.argv[1]))
print(" ".join(k for k, v in st.items() if v.get("init_failed")))
PY
)
  if [ -n "$bad" ] && [ "${ALLOW_INIT_FAIL:-0}" != 1 ]; then
    echo "[runbook] init failed (no goal) in: $bad -- see their report.md; every other episode ran. ALLOW_INIT_FAIL=1 to go on" >&2
    exit 3
  fi
}

step_4() {
  batch eval --only "$(only '')" | tee "$OUT/logs/batch_eval_${RUN_TAG}.log"
  $PY scripts/bggen/aggregate.py "$OUT/runs/$RUN_TAG"
}

case "${1:-}" in
  list) step_list ;; 1a) step_1a ;; 1b) step_1b ;; 2) step_2 ;; 3) step_3 ;; 4) step_4 ;;
  all) step_1a; step_1b; step_2; step_3; step_4 ;;
  *) echo "usage: $0 list|1a|1b|2|3|4|all"; exit 2 ;;
esac
