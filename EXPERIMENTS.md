# cap-bg EXPERIMENTS

## 20261101_bgest2_vocab_smoke (2026-11-01) — 통과, 단 t06 Observe 회귀 → 10×5 중단·진단
- 실행: kubectl baccus_cap-bg, bg d16ede4 (SAM3 프롬프트 후보 + 재검출), 20:41–21:24 UTC, t06·t08·t09 × 2, 6 trial 모두 과제 실패.
- 결과 (baccus 보고 원문): 크래시 0 · 누출 0 · Observe 파싱 실패 0 · belief_trace 6/6.
  t08·t09: turn 0 검출 3/3 (cream cheese "small blue box", 재검출 0), cheese 표식 위치가 프레임의 파란 상자와 일치, goal_match(·normalized) True,
  목표 술어 20/20·12/12·24/24·20/20, 진행 오차 0, Observe 답한 것 전부 정답. turn 0 의 (at cream_cheese center) 1건 불일치.
  t06: 검출 5/5 (plate "saucer", bowl "bowl"), goal_match True 2/2. 그러나 Observe on(진실 1) 정답 3/10·8/10 (앞 smoke 10/11·17/18),
  목표 술어 17/24·14/16, 진행 오차 7·6, S5b 오발동 2·1.
- 판단: 기준상 통과였지만 t06 회귀(접시 프롬프트 변경 의심)가 S5b 오발동으로 이어져, 21:27 UTC 에 시작한 20261101_mem10x5_bgest2 를 바로 중단하고
  표식 위치 진단 → 필요 시 plate 1순위 복귀 → t06 smoke → 10×5 재시작.
- 진단 (baccus, 기존 산출물): 실제로 시작된 10×5 task 는 0 (pod 준비 중 중단). 원인은 프롬프트·표식 위치가 아님 — turn 0 표식 픽셀은 두 run 동일.
  on 오답 10/40 중 9개가 VDM 1·정답 0 이고 모두 그릇이 접시 테두리에 걸쳐 기운 상태(d_xy 6.4–7.3 cm > r 5.4–6.3, dz 3.0–3.6 cm > 평평 1.8):
  VDM 0.9 가 기하를 덮어써 단계 확정·S5b 오발동. 추가로 팔에 가려 미검출된 plate_C 의 표식이 팔 위에 그려져 VDM 이 "gripper holding plate_C" 로 답함.
  → 수정 L2 exp/bg-est2-rim: 기하 거부권(d_xy > r 또는 dz 초과 시 VDM 1 거부, P 0.5·S4), 미검출 트랙 표식 생략, M2/M5 에 "on = resting flat" 문장.
  진단 이미지: kubectl outputs/diag_t06_saucer/, 스크립트 /workspace/cap-bg/t06_*.py

## sam3_prompt_probe (2026-11-01) — keep: SAM3 범주 프롬프트 선택
- 실행: kubectl baccus_cap-bg, GPU pod 1개, agentview RGB 800×512, LIBERO-Mem task 0–9 × trial 1·2 = 20장, 정답 = problem.objects_dict, 판정 = Sam3PerceptionSource 규칙(점수 ≥0.4, IoU >0.5 중복 제거).
- 결과 (baccus 보고 원문, '정답 개수와 정확히 맞은 장면 / 범주가 있는 장면', 평균 점수):
  cream cheese: "small blue box" 4/4 0.86 · "blue box" 4/4 0.73 · "blue package" 3/4 · "cream cheese"·"cream cheese box"(v1)·"white box"·"carton" 0/4 (렌더가 작은 파란 상자). 없는 장면 오검출 0.
  plate: "saucer" 16/16 0.84 · "plate"(v1) 15/16 0.65 (t07 접시가 모두 덮인 장면 0개). bowl: "bowl" 16/16 0.91 · "black bowl"(v1) 14/16 (t07 과검출).
  bottle: "bottle" 12/12 0.96 · "wine bottle" 12/12 0.94. basket: "basket" 4/4 0.81 · "wicker basket" 3/4.
- 판단: bg_est2 범주별 프롬프트 목록을 실측 1순위로 (L2 exp/bg-est2-vocab). 한계: 같은 장면 안 다른 물체를 잡았는지(위치 대조)는 미확인, "saucer" 의 접시 없는 장면 오검출 미확인.
- 산출물(kubectl PVC): outputs/sam3_prompt_probe/{table.md,results.json,round2/,frames/}

## 20261101_bgest2_smoke (2026-11-01) — 미통과 (task_08 cream cheese 검출 실패), 그 외 통과
- 실행: kubectl baccus_cap-bg, bg d72231a, 18:55–19:36 UTC. task_06·task_02·task_08 × 2, memory_mode=bg_est2, 실제 SAM3. 6 trial 모두 과제 실패.
- 기준 (baccus 보고 원문): 크래시 0 ✓ · turn 0 검출 수 t06 2/2 ✓, t02 2/2 ✓, t08 0/2 ✗ (cream cheese 를 "cream cheese box" 로 한 번도 못 찾음) ·
  goal_match_normalized t06 2/2 ✓ (v1 누적 표기 문제 해결), t02 2/2 ✓, t08 ✗ (검출 목록에 cheese 가 없어 목표 파싱이 unknown object 로 거부) ·
  누출 0 ✓ · Observe 파싱 100% ✓ · belief_trace 6/6 ✓.
- 보고 항목: 선언 비율 t06 5/5·10/10, t02 6/6·6/6, t08 0/6·1/7. Effect 정답 t06 1/5·14/16, t02 6/8·2/4. 하위 작업 판정 정답 t06 3/4·4/10, t02 2/3·0/3.
  Observe 정확도(그림자) 약 97%, ? 1회. 목표 술어 정확도 t06 20/20·39/40, t02 6/6·5/6. 진행 오차 t06 0·1, t02 3·2. 이름 뒤바뀜 0.
  신호 [발동, 정답]: S1 [2,2][7,3][3,1][3,2][0,0][3,3], S2 전부 정답, S4 [1,0][0,0][2,2], S5b [1,0] (t06 tr2 오발동), S5a [1,1]×2 (t02).
  추가 호출: Observe 턴당 1 (Qwen 출력 3k–36k 토큰/trial), SAM3 14–24회/trial, goal parse 1–2회.
- 판단: v1 문제(누적 표기·진행 오차)는 해결. 10×5 를 막는 것은 지각 어휘 하나 → SAM3 프롬프트 탐색 후 범주별 프롬프트 목록 + 재검출 로직 추가, t08·t09 재-smoke.
  참고로 볼 것: Effect 정답률·하위 작업 판정 정확도가 trial 에 따라 낮음(t06 tr1 1/5, tr2 4/10), S5b 오발동 1.
- 산출물(kubectl PVC): outputs/capbg_bg_est2_smoke/, logs/20261101_bgest2_smoke/

## eval_observe (2026-11-01) — keep: VDM Observe 오프라인 정확도
- 실행: kubectl baccus_cap-bg, CPU OSMesa, 코드 exp/bg-est2 fa342b4 (scripts/eval_observe.py). LIBERO-Mem t2·6·7·8·9 × 8 = 장면 40, 진술 440.
  조건: init 10, arm 10, normal 6, tilted 3, table 5, in_basket 4, basket_center 2 (생성 보고 기준).
- 결과 (baccus 보고 원문):
  | VDM | 정확도(답한 진술) | ? 비율 | 1→0 오답 | 0→1 오답 | 비용 |
  | Qwen/Qwen3.8-27B | 0.968 (425/439) | 1/440 | 6/141 (4.3%) | 8/298 (2.7%) | 0.37달러 |
  | moonshotai/Kimi-K3 | 0.960 (413/430), 읽기 실패 응답 1 | 0 | 6.5% | 2.7% | 4.35달러 |
  Qwen 조건별: init 1.00 · arm 1.00 · table 0.98 · in_basket 0.97 · normal 0.95 · tilted 0.92 · basket_center 0.72 (13/18).
  predicate별(Qwen): on(진실0) 0.98 · on(진실1) 0.96 · in 0.93/1.00 · at(center) 진실1 0.50 (2개) · inhand 0.97 · handempty 0.97.
  오답 사례: 기운 그릇을 '쥐고 있음'으로 봄(t02 tilted), 표식 위치 때문에 plate_B/C 를 바꿔 읽음(t06 normal, Notes 에 스스로 언급),
  테이블 위 그릇을 접시 위로 봄(t07 table), basket_center 의 at·in 약함.
- 판단: keep. Qwen3.8-27B 를 bg_est2 VDM 으로 유지 (정확도 같고 비용 약 1/12). 10×5 진행 조건(≥80%) 통과.
  한계: arm 장면이 실제로는 물체를 거의 가리지 않아 '가리면 ? 를 쓰는지'는 미확인. 영역 목표(center) 판정이 약함. 표식 위치 오류가 오답 원인.
- 산출물(kubectl PVC): outputs/eval_observe/{scenes,qwen3.8,kimi-k3}/{table.md,score.json,results.jsonl}

## 20261030_mem10x5_{none_v2,vdmhist,bgoracle} (2026-10-30 ~ 10-31) — keep (첫 메모리 조건 비교), 해석은 제한적
- 실행: kubectl baccus_cap-bg, 큐 13:33 UTC(10/30) – 11:50 UTC(10/31), 인프라 재실행 12:10. 코드 bg 11008c5.
  pod 1–2개(worker 1–2)로 운영: pod kill 1회(15:37), 사람 작업(ferrari-vla)에 GPU 양보 1회, OOM 1회(00:23, 3.8h 미감지) → 중단 trial 은 resume·재실행.
- 설정: env_configs/capbg_baseline/libero_mem task_00..09 × 5, 조건별 --memory-mode. Kimi-K3 + Qwen3.8-27B VDM (Nebius), 앙상블 끔, VDM 켬,
  seg 끔, 영상 on, step 8000 / wall 3600 / retries 1, max_tokens 32768. 세 조건을 task 단위로 섞은 큐, 같은 seed(trial 1–5) 짝.
- 결과 (baccus 보고 원문, '인프라 재실행 반영' 기준; 재실행 2건(bg t1 tr1 HTTP 529, none t1 tr5 ReadTimeout)은 둘 다 실패):
  | 조건 | 성공 | 진행률 | upstream init(t0,6,7,8) | 생성 init(t1-5,9) | 종료 사유 |
  | none | 0/50 | 0.17 | 0/20 (0.09) | 0/30 (0.22) | step_cap 44, finish_but_fail 6 |
  | vdm_history | 1/50 | 0.25 | 1/20 (0.21) | 0/30 (0.28) | step_cap 42, finish_but_fail 6, turn_limit 1 |
  | bg_oracle | 7/50 | 0.42 | 3/20 (0.34) | 4/30 (0.47) | step_cap 39, finish_but_fail 4 |
  wall timeout 0, 오염 0, finish_reason=length: none 1 / vdm 0 / bg 0.
  짝 비교: bg vs none 성공 bg만 7 / none만 0 (부호검정 p≈0.016), 진행률 높음 23 / 같음 16 / 낮음 11 (p≈0.058).
           vdm vs none 성공 1 / 0, 진행률 높음 16 / 같음 25 / 낮음 9 (p≈0.23).
  task별 성공·진행률 (none | vdm | bg): t0 0/5 .20 | 1/5 .60 | 3/5 .60 · t1 0 .20 | 0 .20 | 2/5 1.00 · t2 0 .40 | 0 .60 | 0 .40 ·
    t3 0 .00 | 0 .07 | 2/5 .80 · t4 .48 | .48 | .48 · t5 .14 | .14 | .14 · t6 .07 | .13 | .20 · t7 .00 | .00 | .55 · t8 .10 | .10 | .00 · t9 .10 | .20 | .00
  bg 신호: S2 151회(41 trial), 다음 턴 진전 20/121. S5b 8회 중 FINISH 5. S5a 1회(두 번째 답 regenerate, 실패). overshot 5.
- 토큰 (coding Kimi-K3 만; VDM usage 미기록): 입력 약 7.2M, 출력 약 8.4M (none 2.35/2.49, vdm 2.69/3.06, bg 2.07/2.72 M).
  비용(카탈로그 $3/$15 per M, baccus 계산): none 약 44달러, vdm 약 54, bg 약 47, 재실행 약 2 → coding 약 147달러. VDM(Qwen, 약 780회)은 usage 미기록, 수 달러 추정.
- 판단: keep. bg_oracle 이 none 보다 뚜렷이 높음. 그러나 효과가 기억 과제(반복 t2·t4·t5, 교환 t6)가 아니라
  단순 목표(t0·t1·t3, 특히 bottle 과제)와 t7(지시문 방향 모호)에 몰림 → 정답 물체 좌표(지각 대체)와 정답 목표 순서(BDDL) 효과일 가능성이 큼.
  vdm_history 는 none 대비 유의한 차이 없음.
- next: (1) bg_est (정답 없음) smoke → 10×5 로 같은 seed 비교. (2) bg_oracle ablation: 좌표 제거(관계만), 목표를 LLM 파싱으로 교체, 신호 끄기 —
  정답 지각·정답 목표·기억 구조 효과 분리. (3) 영상 프레임 디스크 스트리밍(1i) 으로 worker 수 회복.
- 산출물(kubectl PVC): outputs/capbg_baseline/20261030_memcond_COMPARE_{orig,reflected}.md, logs/20261030_memcond_final/{orig,reflected}/,
  outputs/capbg_baseline/20261030_mem10x5_{none_v2,vdmhist,bgoracle}{,_infra_rerun}/, 중단분 _aborted_podkill, _aborted_podkill_vla, _aborted_oom.

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
- 실패 분석 (FAILURES.md, baccus, 기존 산출물만):
  - 실패 48건: 조작 31 · 물체 혼동 5 · 잊음·순서 3 · 하네스 3 · 완료 후 계속 행동 2 · 예산 부족 2(task 5) · 지시 해석 반대 2(task 7).
    기억 관련(혼동+잊음·순서+완료 후 계속) 10건 = 21%. upstream init 실패 18건 중 2, 생성 init 실패 30건 중 8 — 횟수 세기 과제(t2–t4)에 집중.
  - task 7 은 지시문("rotate ... from left to right")과 BDDL 순서가 거울상으로 모호 → tr3 은 거울상 3단계를 해내고 0점. 기억 실패로 세지 않음.
  - step 예산: 하위 목표당 step 중앙값 t0 455 · t1 569 · t2 1590 · t3 710 · t4 757 · t5 1778 · t8 1343 · t9 623 (t6·t7 달성 0).
    하위 목표 수 × 중앙값이 8000 을 넘는 것은 task 5(7회, 약 12,448)뿐. step_cap 34건 중 23건은 하위 목표 0, 마지막 1000 step 안 진전 4건.
    → 8000 은 task 5 외에는 병목 아님.
  - unclosed_fence 17건: SyntaxError 후 다음 턴 12 · 잘린 코드가 오류 없이 실행 3 · 무한 루프 timeout 1 · 그 블록에서 turn_limit 1.
  - 뷰어: https://claude.ai/artifact/6KRnzgXKgwfBZHsw8kiZVd (사람 계정 private)
- next: 잘림 처리 수정 → task 5 step 예산 → 조작 실패가 지배적인 상황에서 BG 효과를 볼 평가 설계 결정 → Oracle BG 구현.
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
