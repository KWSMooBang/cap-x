# cap-bg — CaP-X + Belief Graph

## 목표
cap-x 에 Belief Graph 모듈을 추가해 memory 관리 능력을 높이고 long-horizon task 성능을 올린다.
- 방법: VDM 관측 → Belief Graph 업데이트 → 그래프가 바뀌면 code agent 가 code refinement.
- 배경: Behavior-1k challenge 의 VLA + Belief Graph 방식 ("Belief Graph 구현 대조", "Belief Graph 쉬운 해설" 아티팩트,
  Notion: https://app.notion.com/p/Skeleton-short-40716fdc38d28054a330c8f59df8451a)
- 타겟 benchmark: LIBERO-Mem (https://libero-mem.github.io, arXiv 2511.11478), RoboMME (https://robomme.github.io, arXiv 2603.04639)

## 결정
- 2026-10-29: 연구 브랜치는 `bg` (main 에서 분기). L2 브랜치는 리뷰 후 bg 로 merge.
- 2026-10-29: 두 benchmark 는 `capx/third_party/` 아래 submodule 로 추가 (기존 LIBERO-PRO 등과 같은 방식).
  cap-x 연결 코드(adapter·simulator·task·env_configs)는 submodule 밖에 둔다. 의존성 충돌 시 별도 uv extra + conflicts.
- 2026-10-29: 병렬 앙상블(use_parallel_ensemble·use_multimodel)은 기본으로 끈다. code agent 는 단일 호출.
  코드 기본값은 이미 False(capx/utils/launch_utils.py:143,152)이고 켜는 곳은 YAML 뿐 — upstream
  env_configs/libero/franka_libero_cap_agent0.yaml(논문 CaP-Agent0 설정)과 hillclimb ensemble_*/multimodel_* 변형.
  upstream YAML 은 논문 설정 기록으로 두고, 그 YAML 로 돌릴 때는 --use-parallel-ensemble False --use-multimodel False 를 준다.
  앙상블은 필요할 때 ablation 으로만 켠다.
- 2026-10-30: 기준선 trial 예산 (사람이 kubectl 세션에 승인): --trial-timeout-s 1800 --max-trial-retries 1, timeout 은 실패로 집계.
  배경: 원래 runner 의 1000s SIGALRM timeout 이 sandbox(except BaseException)에 삼켜져 사실상 동작하지 않았음 → exp/trial-timeout 에서 수정.
  재시도 3회를 그대로 두면 timeout 난 trial 을 다시 굴려 성공률이 위로 편향됨. 종료 사유(success / FINISH 후 실패 / 턴 한도 / timeout)는 따로 집계한다.
  주의: 논문 수치도 이 버그가 있는 harness 에서 나왔을 가능성 → LIBERO-PRO 를 논문과 비교할 때 timeout 설정 차이를 적는다.
- 2026-10-30: LIBERO-Mem 은 camera segmentation 관측을 끈다 (exp/libero-mem-noseg). 켜 두면 robosuite 의 픽셀별 ID 매핑 때문에
  step 당 0.470s (LIBERO-PRO 0.023s) 였고, privileged=false 인데 GT instance mask 가 get_observation() 으로 노출됐음.
  끈 뒤 0.025s/step. Oracle BG 가 정답이 필요하면 seg 가 아니라 sim state 를 직접 읽는다.
  그 전 LIBERO-Mem run(20261029_mem_pilot_*, 5x5 run)은 'seg on (느린 경로)' 조건으로 기록.
- 2026-10-30: 비교 run 평가 설계 — CaP-Agent0 API(FrankaLiberoApiReducedSkillLibrary) 그대로 유지 (사람 결정 '그대로').
  정답 위치 API 진단 실험은 하지 않음. 주 지표는 하위 목표 진행률·기억 관련 실패 비율·완료 후 계속 행동 비율.
- 2026-10-30: 판단 프롬프트에 목표 문장이 채워지지 않는 원본 버그('Goal: {libero_environment_goal}')를 모든 조건에 고친다 (사람 승인).
- 2026-10-30: 모든 run 은 영상을 항상 녹화한다 (사람 결정). 실패 유형 분류에 필요. 메모리 부족이면 worker 수를 줄이고 영상은 유지.

## 백로그
1. [완료·bg 병합 b9c0046] bench-setup: LIBERO-Mem·RoboMME submodule + uv extra + cap-x wrapper (RoboMME 카메라 렌더링은 kubectl 쪽에서 진행)
1a. [완료·bg 병합 b51b346] libero-init-fix: reset 직후 sim state = init_states[(seed-1)%N] (차이 0.0). LIBERO-Mem 우회 제거
1b. [완료·bg 병합 b98b4ac] baseline-configs: env_configs/capbg_baseline/ (LIBERO-PRO swap 0, LIBERO-Mem task_00..09),
    앙상블 끔·VDM 켬·multi-turn 켬, 생성기 scripts/gen_capbg_baseline_configs.py, docs/capbg-baseline.md
1c. RoboMME control API (지금 apis: [] 라 cap-x 로 못 돌림)
1d. LIBERO 설정 경로를 레포별로 분리 (이 서버의 ~/.libero/config.yaml 이 다른 프로젝트 RACA 를 가리킴)
1e. [완료] kubectl 파일럿 3회 (EXPERIMENTS.md, 모두 discard — 하네스 문제)
1f. [완료·bg 병합 f9b3e13] 하네스 수정 4건 (2026-10-30 사람 승인, baccus_cap-bg 가 순서대로 쌓아 구현):
    exp/no-privileged-obs (생성 코드에서 obs["memory"]·env 제거, info 로만 기록, 오염 검사) →
    exp/decision-parse (REGENERATE 누락 코드 블록·tool-call·빈 응답 처리, decision_source 기록) →
    exp/step-cap (max_steps 도달 시 삼킬 수 없는 예외로 종료) →
    exp/libero-mem-init-states (task 1·2·3·4·5·9 init state 50개씩 생성, upstream 4개와 분리 보고)
    원칙: 두 조건(기준선·BG)에 같은 하네스. BG 가 풀 문제(완료 판단·이력 기억)는 하네스에 넣지 않는다.
    init 생성: task 1–5 통과율 33.6% (탈락 대부분 plate contact), task 9 100%. task 1–5 는 물체·영역 정의가 같아
    같은 seed 로 같은 50개 배치가 나옴 → 그대로 사용 (같은 장면에서 지시만 다른 통제된 비교; task 0 은 upstream 세트).
1i. 영상 프레임을 메모리에 쌓지 않고 디스크로 바로 쓰기 (800x512 프레임이 4 step 마다 trial 끝까지 RAM 에 쌓여 worker 당 수 GB →
    40Gi pod 에서 worker 5 가 OOM, 2026-10-30 파일럿은 worker 3 으로 대응). 에이전트가 보는 VDM 이미지와는 별개 경로라 결과엔 영향 없음.
1j. [완료·bg 병합 660d1a9] exp/truncation (2026-10-30 사람 결정, baccus_cap-bg): finish_reason=length 응답은 실행 안 함(재질의 1회, 또 잘리면 피드백만),
    max_tokens 32768 (YAML 지원, coding·VDM 공통, 원본 기본값 20480 과 다름), timeout trial 도 영상 저장.
    이번엔 안 함: task 5 step 상한, 영상 프레임 디스크 스트리밍(1i), raw 코드 파서 보완.
2a. [완료·bg 병합 cae13af] oracle-bg (L2 exp_cap-bg_oracle-bg): memory_mode none / vdm_history / bg_oracle, 정답 그래프 직렬화 + S2(정체)·S5(완료 검증).
    Phase A(capx/belief 독립 모듈) → exp/truncation 병합 후 Phase B(하네스 통합). 평가 설계(실험 규모·API 수준)는 비교 run 전 결정.
1g. 결정 대기: 비교 예산 기준 (제안: sim step 상한 8000 이 주 예산, wall 은 넉넉한 안전장치)
1h. 2026-10-30 사람 사전 승인: 하네스 수정 4건 리뷰 통과 시 bg 병합·push → LIBERO-Mem 파일럿 재실행
    (task_00..09 × 5, step 8000 / wall 3600 / retries 1, max_tokens 20480, 앙상블 끔·VDM 켬·seg 끔).
2. 기존 cap-x baseline 으로 두 benchmark 평가 (Belief Graph 없이)
3. Belief Graph 모듈 설계·구현 (VDM 연동, code refinement 트리거)

## GPU 예산
- 미정 (main-control [BUDGET] 대기)
- 2026-10-29: 사람 지시로 현재 GPU 사용 금지 (bench-setup 은 CPU 만). 해제 지시 전까지 GPU run 을 제출하지 않는다.
- 2026-10-29: 이 서버는 쓸 GPU 가 없음. 남은 환경 설정(RoboMME 카메라 렌더링 등 GPU 필요한 부분)은
  kubectl 서버 전용 세션 baccus_cap-bg 가 진행 (RoboMME: lavapipe 로는 렌더 불가 → NVIDIA Vulkan ICD 필요).
