# cap-bg EXPERIMENTS

## 20261030_mem10x5_capx (2026-10-30) — keep (첫 유효 기준선 파일럿, 단 floor effect)
- 실행: kubectl baccus_cap-bg, 06:57–08:48 UTC. pod 3 × (A100, CPU 18, 40Gi), worker 3 (worker 5 는 영상 프레임 누적으로 OOM → 재시작,
  중단분 _aborted_w5/). GPU 약 6 GPU-h (재시작 전 포함).
- 코드: bg 808bc8e (하네스 수정 전부 반영: init state, timeout, model retry, seg off, no privileged obs, decision parse, step cap, 생성 init).
- 설정: env_configs/capbg_baseline/libero_mem task_00..09 × 5. Kimi-K3 + Qwen3.8-27B VDM (Nebius), 앙상블 끔, VDM 켬, 영상 on,
  step 8000 / wall 3600 / retries 1, max_tokens 20480.
- 결과 (baccus 보고 원문): 성공 2/50 (task_00 1/5, task_08 1/5) · 2/50 (API timeout 0) · 2/49 (timeout 1건 제외).
  init_source 별: upstream(task 0·6·7·8) 2/20, 평균 하위 목표 진행률 0.10 / capbg_generated(task 1–5·9) 0/30, 진행률 0.32 / 전체 0.23.
  종료 사유: step_cap 34, finish_but_fail 6, turn_limit 5, success 2, misparsed_finish 1, empty_response 1, timeout 1, api_timeout 0.
  decision_source (coding 응답 293): keyword 228, finish_keyword 8, code_block 5, tool_call 1, fallback 1, empty_retry 1.
  원래 파서였다면 판정이 달랐을 턴 5. unclosed_fence 17. finish_reason=length 17/293 (5.8%). 오염 검사 0/50.
  토큰 (coding): prompt 3.24M, completion 2.57M (reasoning 2.08M). 호출 599회 (coding+VDM). 카탈로그 단가 기준 coding 약 48달러 (정확한 금액은 Nebius 콘솔).
- 판단: keep — 하네스 문제 없이 얻은 첫 기준선. 다만
  (1) 성공률 4% 로 floor effect → BG 와의 차이를 성공률로는 보기 어려움. 하위 목표 진행률·실패 유형이 주 지표가 되어야 함.
  (2) step_cap 이 34/50 으로 주 종료 사유 → 8000 step 이 반복·교체 과제에 충분한지 먼저 확인 필요 (하위 목표당 사용 step).
  (3) 잘린 응답(length) 의 미종결 코드가 실행됨: unclosed_fence 17건, 그중 task_02 tr1 은 무한 루프로 3600s timeout (영상 없음).
  (4) task_03 1건은 펜스 없는 raw 코드가 fallback → finish 처리.
- next: 실패 유형 분석 (턴별 sim_steps·진행 상태·overshot + 영상) → step 예산 판단 → 잘림 처리·max_tokens 결정 후 BG 비교 run.
- 산출물(kubectl PVC): /workspace/cap-bg/outputs/capbg_baseline/20261030_mem10x5_capx/{RESULTS.md,ANALYSIS.md},
  로그 /workspace/cap-bg/logs/20261030_mem10x5_capx/, 영상 .../moonshotai_Kimi-K3/<task>/trial_*/video_combined.mp4 (49/50)

## 20261029_mem5x5_step8000 (2026-10-30) — discard (하네스 버그 3건), 관찰은 keep
- 실행: kubectl baccus_cap-bg. 코드 52c119f (bg 2e59b74 + exp/libero-mem-egl + exp/trial-timeout).
  camera seg on (느린 경로, 약 2 step/s/worker). OOM 2회·vLLM 중복 1회로 task_06/07 은 재실행분만 집계 (부분 산출물 _oom_partial/).
- 설정: env_configs/capbg_baseline/libero_mem task_00/02/06/07/08 × 5. Kimi-K3 + Qwen3.8-27B VDM (Nebius), 앙상블 끔, VDM 켬.
  예산: step 8000(주) / wall 3600s / retries 1. 영상: task 0·2·7 on, 6·8 off.
- 결과 (baccus 보고 원문): 성공 1/25 (전부 집계) · 1/23 (API timeout 제외) · 1/10 (모든 timeout 제외). 성공은 task_00 trial5 하나.
  종료 사유: timeout 13, misparsed_finish 5, api_timeout 2, finish_but_fail 2, empty_response 1, step_cap 1.
  task_00: 성공 1 / finish_but_fail 1 / timeout 3 · task_02: finish_but_fail 1 / misparsed 1 / timeout 3 ·
  task_06: misparsed 3 / timeout 2 · task_07: misparsed 1 / timeout 3 / api 1 · task_08: empty 1 / step_cap 1 / timeout 2 / api 1
- 판단: discard. 결과를 가린 하네스 문제:
  (1) seg 관측 때문에 sim 이 느려 step cap 대신 wall timeout 이 걸림 → exp/libero-mem-noseg 로 해결(bg 0452124)
  (2) _parse_multi_turn_decision 이 'REGENERATE' 가 없으면 무조건 FINISH → FINISH 종료 9개 중 6개가 실제로는 코드(5)·빈 응답(1). 미수정
  (3) 모델 HTTP 200s timeout → exp/model-request-retry 로 해결(bg 0452124)
  (4) 생성 코드가 robosuite 'terminated episode' ValueError 를 삼키면 step cap 이 강제되지 않음 (task_08 trial2: 8000 step 도달 후 3600s timeout). 미수정
  (5) init state 가 없는 LIBERO-Mem task(1·2·3·4·5·9)는 raw BDDL reset 이라 시작 장면이 물리적으로 깨짐
      (plate region 3 이 1·2 를 포함 → 접시 겹침 → 병·그릇 넘어짐). task_02 trial 1·2·4·5 는 깨진 장면에서 시작 → 무효.
      ferrari-dev 실측: task 2 raw reset 10회 중 8회에서 20° 넘게 기운 물체. upstream 은 contact 0 + 사람 검수로 만든 init 을 task 0·6·7·8 에만 제공.
- 특권 정보 오염 점검 (2026-10-30, baccus 읽기 전용 검사): LIBERO-Mem 전 run(pilot_ensoff, pilot_t1800, mem5x5, _oom_partial)과
  LIBERO-PRO run10 의 모든 생성 코드에서 obs["memory"]·task_completed·_check_success·env.·sim.data·obs 키 나열 사용 0건 → 결과 오염 없음.
  단 노출 경로(get_observation() 의 obs["memory"], sandbox 전역 env/obs)는 열려 있음.
- next: (2)(4)(5) 수정 후 bg 기준으로 LIBERO-Mem task 0–9 기준선 재실행.
- 산출물(kubectl PVC): /workspace/cap-bg/outputs/capbg_baseline/20261029_mem5x5_step8000/{RESULTS.md,ANALYSIS.md},
  로그 /workspace/cap-bg/logs/20261029_mem5x5_step8000/, 뷰어 데이터 /workspace/cap-bg/viewer/20261029_mem5x5_step8000/,
  뷰어 https://claude.ai/artifact/3DhmnW2RMeki3S4QnzQJLg (사람 계정 private)

## 20261029_mem_pilot_ensoff_t1800 (2026-10-30) — discard (예산 문제로 판정 불가), 관찰은 keep
- 실행: kubectl baccus_cap-bg, A100 1장 · CPU 16, EGL, worker 5. 비용 약 8달러.
- 코드: 17b71ae (= bg 2e59b74 + exp/libero-mem-egl 0b3fc22 + exp/trial-timeout 17b71ae)
- 설정: env_configs/capbg_baseline/libero_mem task_00 / task_02 / task_06 × 5 trial.
  Kimi-K3 coding + Qwen3.8-27B VDM (Nebius), 앙상블 끔, VDM 켬, multi-turn 켬.
  budget: 실제로는 reset 후 1000s · 재시도 없음 (--trial-timeout-s 1800 은 trial.py 하드코딩 alarm(1000) 때문에 무시됨 → 8b3edea 에서 수정).
- 결과 (baccus 보고 원문): 0/15 성공 (timeout 13 / FINISH-but-fail 2).
  task_00: 0/5 (timeout 3, finish_but_fail 2) → timeout 제외 0/2 · task_02: 0/5 (timeout 5) · task_06: 0/5 (timeout 5)
- 관찰:
  - timeout 원인은 API 지연이 아니라 sim 코드 실행 시간. 모델 호출 69회, 중앙값 6–8s, Nebius 재시도 0, Molmo 실패 0. 1000s 안에 trial 당 0–4턴.
  - 중단 run(20261029_mem_pilot_ensoff, timeout 미강제, task_00): trial 5 가 약 16분에 과제 완료 상태에 도달했지만 인지하지 못하고
    계속 행동해 실패(9턴, 1149s). trial 3 도 약 13분에 완료 상태 도달. → CaP-BG 신호 S5(완료 검증)가 노리는 실패가 실제로 나타남.
  - GPU EGL 패치(exp/libero-mem-egl) 실검증 완료 — trial 중 EGL 에러 없음.
- 판단: discard. 1000s 예산이 해결 가능한 trial 도 자르므로 기준선 숫자로 쓸 수 없음.
- next: 턴별 시간(099eabe 부터 all_responses.json 에 exec_s·decision_s)으로 sim 실행이 왜 느린지 확인 → 예산 재설정 후 기준선 재실행.
- 한계: 5 trial/task, CPU 경합(worker 5 + 서버들) 미측정.
- 산출물(kubectl PVC): /workspace/cap-bg/outputs/capbg_baseline/20261029_mem_pilot_ensoff_t1800/{RESULTS.md,ANALYSIS.md},
  로그 /workspace/cap-bg/logs/20261029_mem_pilot_ensoff_t1800/, 집계 /workspace/cap-bg/analyze_pilot.py

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
