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

## 백로그
1. [진행] bench-setup: LIBERO-Mem·RoboMME submodule + uv extra + cap-x env wrapper + smoke rollout
2. 기존 cap-x baseline 으로 두 benchmark 평가 (Belief Graph 없이)
3. Belief Graph 모듈 설계·구현 (VDM 연동, code refinement 트리거)

## GPU 예산
- 미정 (main-control [BUDGET] 대기)
