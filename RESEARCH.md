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

## 백로그
1. [완료·bg 병합 b9c0046] bench-setup: LIBERO-Mem·RoboMME submodule + uv extra + cap-x wrapper (RoboMME 카메라 렌더링은 kubectl 쪽에서 진행)
1a. [완료·bg 병합 b51b346] libero-init-fix: reset 직후 sim state = init_states[(seed-1)%N] (차이 0.0). LIBERO-Mem 우회 제거
1b. [완료·bg 병합 b98b4ac] baseline-configs: env_configs/capbg_baseline/ (LIBERO-PRO swap 0, LIBERO-Mem task_00..09),
    앙상블 끔·VDM 켬·multi-turn 켬, 생성기 scripts/gen_capbg_baseline_configs.py, docs/capbg-baseline.md
1c. RoboMME control API (지금 apis: [] 라 cap-x 로 못 돌림)
1d. LIBERO 설정 경로를 레포별로 분리 (이 서버의 ~/.libero/config.yaml 이 다른 프로젝트 RACA 를 가리킴)
1e. [진행] kubectl 기준선 파일럿 — 사람이 baccus_cap-bg 에 직접 지시
2. 기존 cap-x baseline 으로 두 benchmark 평가 (Belief Graph 없이)
3. Belief Graph 모듈 설계·구현 (VDM 연동, code refinement 트리거)

## GPU 예산
- 미정 (main-control [BUDGET] 대기)
- 2026-10-29: 사람 지시로 현재 GPU 사용 금지 (bench-setup 은 CPU 만). 해제 지시 전까지 GPU run 을 제출하지 않는다.
- 2026-10-29: 이 서버는 쓸 GPU 가 없음. 남은 환경 설정(RoboMME 카메라 렌더링 등 GPU 필요한 부분)은
  kubectl 서버 전용 세션 baccus_cap-bg 가 진행 (RoboMME: lavapipe 로는 렌더 불가 → NVIDIA Vulkan ICD 필요).
