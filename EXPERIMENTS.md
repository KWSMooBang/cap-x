# cap-bg EXPERIMENTS

## libero-pro-pilot-run10 (2026-10-29) — discard (참고용)
- 실행: kubectl 세션 baccus_cap-bg, pod cap-bg-gpu (A100 1장), MUJOCO_GL=egl, 10 trial · 5 workers, 3794s
- 설정: env_configs/libero/franka_libero_cap_agent0.yaml (libero_object_swap task 0)
  - coding: moonshotai/Kimi-K3 (Nebius), 단일모델 9-temp ensemble + synthesis
  - VDM: Qwen/Qwen3.8-27B (Nebius), 이미지 차이
  - Molmo2-8B 는 .venv-molmo 에서 별도 vLLM 기동 (libero extra 와 conflict)
- 결과 (baccus 보고 원문): code-gen 0.900, avg reward 0.300, task completed 3/10, avg regen 1.6.
  trial 8 은 1000s timeout 3회 초과로 실패 처리. Molmo 42 OK / 0 fail, Nebius 재시도 0.
- 판단: discard — FrankaLiberoEnv.reset init state 버그(최대 0.037 차이, 확정) 때문에 장면이 무작위 배치라
  논문 수치와 비교 불가. 파이프라인이 Nebius 모델로 끝까지 도는지 확인한 용도로만 keep.
- next: init state 버그 수정 → 같은 설정으로 재실행. LIBERO-Mem 기준선 실패 분류(Phase 1).
- 산출물(kubectl PVC): /workspace/cap-bg/outputs/libero_pro_pilot/RESULTS.md, .../moonshotai_Kimi-K3/run10,
  로그 /workspace/cap-bg/logs/libero_pro_run10.log. 재시도된 trial 은 디렉터리가 2개일 수 있음 → summary 기준 집계.
