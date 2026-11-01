#!/usr/bin/env bash
# Agent as BG Generator (capx.bggen) on kubectl: one LIBERO-Mem task-06 demo and B1K task-45 episode_00450010.
#
#   bash scripts/bggen/kubectl_runbook.sh <step>     step = 1a 1b 2 3 4 (in this order), or "all"
#
#   1a  data, LIBERO-Mem (CPU ok): download the swap HDF5 only, dump its structure, replay + render demo_0
#       (agentview RGB-D, K, camera pose, proprio from the HDF5) and the simulator oracle. Needs the LIBERO-Mem venv.
#       CHECKS: structure.json (datasets / attrs / gripper range), prep JSON (replay mode, eef drift HDF5 vs cap-x,
#       success), check-episode (gripper events = 3 grasp + 3 release, depth range, camera height).
#   1b  data, B1K (CPU): download the episode's parquet + head/wrist rgb/depth videos + meta + annotation only,
#       dump the structure, decode at 5 Hz, write the oracle (annotation timeline, task_info inhand / positions).
#       CHECKS: structure.json checks (task_info layout_ok, head_height_1_2m, head_looks_forward_down),
#       check-episode (depth p50 0.5-5 m, events: grasps per arm, base_start/stop).
#       If a check is false, the corresponding path is wrong: STOP and report (camera order / depth constants).
#   2   smoke with FAKE models (scripted local server) + FAKE detector (fixed boxes): output format only.
#       CHECKS: trace.jsonl / cot_rows.jsonl / report.md / keyframes exist, agent fallbacks > 0 (bad_every=5),
#       backward forbidden_violations == [].
#   3   real models + SAM3 (GPU 1): starts the SAM3 server, runs mode A on both episodes (rows_forward = mode B).
#       Needs NEBIUS_SERVER_URL / NEBIUS_API_KEY (or BGGEN_<ROLE>_URL / _MODEL for a vLLM server).
#   4   oracle evaluation: eval.md (forward vs forward+backward), name / goal consistency.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${OUT:-/workspace/cap-bg/outputs/agent-bg-gen}
PY=${PY:-$REPO/.venv/bin/python}                       # main venv (uv sync): huggingface_hub, pandas, pyarrow, SAM3 client
PY_LM=${PY_LM:-$REPO/.venv-libero-mem/bin/python}      # LIBERO-Mem venv (docs/memory-benchmarks.md)
LM_H5=KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5
LM_EP=$OUT/episodes/libero_mem_t06_demo0
B1K_EP=$OUT/episodes/b1k_t45_e00450010
DEMO=${DEMO:-demo_0}
cd "$REPO"
mkdir -p "$OUT"/{data,episodes,runs,logs}

step_1a() {
  $PY -m capx.bggen.cli download-libero --dest "$OUT/data/libero_mem"
  $PY -m capx.bggen.cli dump-libero --h5 "$OUT/data/libero_mem/$LM_H5" --out "$OUT/data/libero_mem/structure.json" | head -80
  # GPU node: EGL; CPU: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1
  MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} \
    $PY_LM -m capx.bggen.cli prep-libero --h5 "$OUT/data/libero_mem/$LM_H5" --demo "$DEMO" --root "$LM_EP" --task-id 6 --wrist \
    | tee "$OUT/logs/prep_libero.json"
  $PY -m capx.bggen.cli check-episode --root "$LM_EP" | tee "$OUT/logs/check_libero.json"
}

step_1b() {
  $PY -m capx.bggen.cli download-b1k --dest "$OUT/data/b1k" --wrist
  $PY -m capx.bggen.cli dump-b1k --src "$OUT/data/b1k" --out "$OUT/data/b1k/structure.json" | tail -40
  $PY -m capx.bggen.cli prep-b1k --src "$OUT/data/b1k" --root "$B1K_EP" --wrist --image-stride 6 | tee "$OUT/logs/prep_b1k.json"
  $PY -m capx.bggen.cli check-episode --root "$B1K_EP" | tee "$OUT/logs/check_b1k.json"
}

step_2() {
  $PY -m capx.bggen.cli run --root "$LM_EP" --out "$OUT/runs/smoke_libero" --benchmark libero_mem --mode A \
    --fake-models --detector format_fake | tee "$OUT/logs/smoke_libero.json"
  $PY -m capx.bggen.cli run --root "$B1K_EP" --out "$OUT/runs/smoke_b1k" --benchmark b1k --mode A \
    --fake-models --detector format_fake | tee "$OUT/logs/smoke_b1k.json"
  for r in smoke_libero smoke_b1k; do
    ls "$OUT/runs/$r"/{trace.jsonl,cot_rows.jsonl,cot_rows_forward.jsonl,rows_backward.jsonl,report.md,summary.json} >/dev/null
    $PY -c "import json,sys; s=json.load(open('$OUT/runs/$r/summary.json')); \
print('$r', 'fallbacks', s['agent_fallbacks'], 'violations', s['backward']['forbidden_violations'])"
  done
}

sam3_up() {
  if ! curl -s -m 2 http://127.0.0.1:8114 >/dev/null 2>&1; then
    nohup $PY -m capx.serving.launch_sam3_server --device cuda --port 8114 --host 127.0.0.1 > "$OUT/logs/sam3.log" 2>&1 &
    for _ in $(seq 60); do curl -s -m 2 http://127.0.0.1:8114 >/dev/null 2>&1 && break; sleep 5; done
  fi
}

step_3() {
  : "${NEBIUS_SERVER_URL:?set NEBIUS_SERVER_URL (or BGGEN_<ROLE>_URL)}"
  sam3_up
  $PY -m capx.bggen.cli run --root "$LM_EP" --out "$OUT/runs/real_libero_A" --benchmark libero_mem --mode A \
    | tee "$OUT/logs/real_libero.json"
  $PY -m capx.bggen.cli run --root "$B1K_EP" --out "$OUT/runs/real_b1k_A" --benchmark b1k --mode A \
    | tee "$OUT/logs/real_b1k.json"
}

step_4() {
  $PY -m capx.bggen.cli eval --root "$LM_EP" --out "$OUT/runs/real_libero_A"
  $PY -m capx.bggen.cli eval --root "$B1K_EP" --out "$OUT/runs/real_b1k_A"
}

case "${1:-}" in
  1a) step_1a ;; 1b) step_1b ;; 2) step_2 ;; 3) step_3 ;; 4) step_4 ;;
  all) step_1a; step_1b; step_2; step_3; step_4 ;;
  *) echo "usage: $0 1a|1b|2|3|4|all"; exit 2 ;;
esac
