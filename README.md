# OGC 2026 — 조선소 블록 공간·시간 스케줄링 솔버

The Grand Shipyard Puzzle (LG CNS Optimization Grand Challenge 2026) 참가 솔버.
각 블록을 **어느 bay에·어디에·어떤 방향으로·언제 넣고(ENTRY)·언제 빼는가(EXIT)** 를
동시에 결정해 `w1·Z1(지연) + w2·Z2(워크로드 불균형) + w3·Z3(선호 페널티)` 를 최소화한다.
제약은 배정·시간·공간(레이어 겹침 금지)·**크레인(수직 이동 오버행 규칙)** 네 가지이며,
채점은 `utils.check_feasibility` 로 검증한 뒤 목적값으로 팀 간 순위를 매긴다.

## 핵심 접근

베이스라인 프로파일링에서 실패의 **97%가 크레인 통로 접근**(공간 충돌은 0%)임을 확인하고,
"면적이 아니라 **크레인 코엑시스턴스**가 병목"이라는 진단 위에 **feasibility를 절대 보장하는
레이어 구조**를 쌓았다. 각 레이어는 이전 해를 **개선-또는-불변**으로만 갱신하고, 최종 반환 해는
반드시 공식 채점기를 통과시킨다(−1을 절대 받지 않는다).

- **크레인 규칙의 양방향 검사** — 베이스라인은 "기존 블록이 새 블록을 막는가"만 보고
  "**새 블록이 기존 블록의 진입·반출을 막는가**"를 누락했다. 이 누락이 stage-2/3 실패의
  구조적 원인이었고, 배치 시점 pairwise 양방향 검사로 제거했다.
- **정확 목적값 추적** — LNS 내부는 `utils` 와 동일 산술로 목적값을 정확히 추적해
  루프마다 채점기를 부르지 않고, 최종 1회만 공식 검증한다.
- **모든 시간 로직은 timelimit의 비율** — 서버 시간제한(수 분~30분)이 인스턴스마다 다르고
  사후 공개되므로 절대상수를 쓰지 않고 안전마진(90%)만 둔다.

## 알고리즘 구조 (레이어)

| 레이어 | 구현 | 역할 |
| --- | --- | --- |
| **Layer 0** | `Solver.build_floor` | serial-per-bay 보장 feasible 생성기(`N(t,j)≤1`)로 크레인·충돌을 구성적으로 회피. 40/40 feasible 하한 확보. |
| **Layer 1** | `Solver.build_coexist` | 크레인-인지 coexistence. **양방향 크레인 검사**로 베이스라인 실패 모드 제거, bbox-conflict 오름차순 후보 정렬. |
| **Layer 2** | `Solver.lns_improve` | 국면 적응 LNS. 결정론적 tardiness 스윕 + 지배 항 룰렛 destroy-repair + **SA 온도 수락** + **timewin destroy**(시간창 통째 재시퀀싱). |
| **Layer 3a** | `Solver.milp_reassign` | bay 배정 마스터 MILP(Gurobi). `Z2`(epigraph)·`Z3`(선형) 배정 수준 최적을 정확히 풀고, rebuild/스텝별 실현 + 투기 분기 헤지로 반영. |
| **Layer 3b** | `Solver.milp_refine_times` | 배치 고정 시 `Z2/Z3`는 상수 → bay별 독립 타이밍 MILP(`_milp_bay_times`)로 `Z1`만 재최적화(**타이밍 최적성 증명**). |
| **하한 증명** | `Solver.prove_o23_bound` | 배정 완화 MILP의 dual bound로 `w2·Z2+w3·Z3` 전역 하한을 인증, 최적 근접 시 조기 종료. |

### 속도·병렬 스택

| 구성 | 구현 | 효과 |
| --- | --- | --- |
| **numba 정수-좌표 기하 커널** | `_geom_*` + `_triangulate` | 좌표 ×10⁴ 정수화, ear-clipping 삼각분할 + 삼각형쌍 SAT로 "면적>0 교차"를 정확 판정. 크레인 술어 미스당 ~62배. |
| **어댑티브 후보 캡** | `_cand_cap` (`_l1_secs > 0.5·TL` 게이트) | 구성이 창을 지배하는 시간막힘 인스턴스만 후보 위치 상한, 수렴형은 uncapped. 비율 기반이라 서버 긴 TL에 자동 완화. |
| **플래그 캐시** | `_flag` | 크레인 술어는 병진 불변 → `(shape, orient, dx, dy)` 키로 재사용(LNS 반복 히트율 ~73%). |
| **전체 해 포트폴리오** | `portfolio_search` | 유휴 코어(서버 400%=4코어)에 독립 LNS 브랜치, 부모가 기본 브랜치를 직접 실행해 best-of-k가 구조적으로 질 수 없음. |
| **bay별 병렬 refine** | `parallel_bay_refine` | 배정 고정 시 per-bay tardiness 완전 분리 → 워커별 최소해 병합(정확). |

## 목적함수

```text
minimize  w1·Z1 + w2·Z2 + w3·Z3

  Z1 = Σ max(0, EXIT_i − Due_i)                     # 총 tardiness
  Z2 = max_{j1≠j2} | u_j1·Σ L − u_j2·Σ L |          # bay 간 정규화 워크로드 불균형
  Z3 = Σ (S_max,i − S_ij)                            # 선호도 페널티
```

`w1, w2, w3` 은 인스턴스마다 크게 다르며, 타이트한 인스턴스는 `Z1`(w1 지배), 느슨한 인스턴스는
`Z3` 가 지배하는 이중 구조를 보인다.

## 설치

서버와 동일한 conda 환경(miniforge)에서 실행한다. 제출본은 `myalgorithm.py + utils.py` 만으로
동작하며, 추가 파이썬 패키지(numpy, numba, gurobipy 등)는 서버 환경에 설치되어 있다.

```bash
conda env create -f alg_tester/ogc2026_env.yml
conda activate ogc2026
```

- `numba` / `gurobipy` 가 없어도 자동으로 순수 shapely 폴백 / no-op 으로 동작한다(무결성 유지).
- Gurobi 는 서버가 주최측 라이선스를 제공한다.

## 실행

엔트리 포인트는 `baseline/myalgorithm.py` 의 `algorithm(prob_info, timelimit=60)` 이며,
서버가 이 시그니처를 호출한다.

- **GUI 테스터 / 시각화**

  ```bash
  conda activate ogc2026
  cd alg_tester
  python alg_tester_app.py     # 인스턴스 → 알고리즘 폴더(baseline/) → 시간제한 → Run
  ```

- **CLI 단일 실행**

  ```python
  import json, myalgorithm
  from utils import check_feasibility

  prob = json.load(open("연습문제 1/train/prob_1.json", encoding="utf-8"))
  sol  = myalgorithm.algorithm(prob, timelimit=30)
  res  = check_feasibility(prob, sol)   # res["feasible"], res["objective"], obj1/2/3
  ```

## 프로젝트 구조

```text
ogc2026/
├── baseline/
│   ├── myalgorithm.py        # 엔트리 + 전 로직(Solver 클래스 + Placement) — 단일 파일 제출본
│   ├── baseline_greedy.py    # 참조 구현 · 헬퍼 출처
│   └── utils.py              # 공식 채점기(check_feasibility) — 수정 금지, 임포트만
├── alg_tester/
│   ├── alg_tester_app.py     # PyQt6 GUI 테스터 · 시각화
│   └── ogc2026_env.yml       # conda 환경 파일
├── 연습문제 1/train/         # 학습 인스턴스 prob_1..20
├── 연습문제 2/train/         # 학습 인스턴스 prob_21..40
├── report/                   # 기술 리포트
├── 문제 정의.pdf             # 공식 문제 정의(v1.2)
├── CLAUDE.md                 # 작업 지침서 · 진행 로그
└── README.md
```

## 설계 원칙 · 개발 규율

- **feasibility 최우선** — 리더보드는 infeasible/시간초과/크래시를 모두 −1로 매긴다.
  항상 "보장된 feasible 해"를 들고 가고, 최종 반환 해는 반드시 공식 채점기로 통과 확인한다.
- **단일 파일 방침** — 우리 로직은 전부 `myalgorithm.py` 안에 둔다(`Solver` + `Placement`).
- **측정 우선(measurement discipline)** — 인스턴스별 run 분산이 커서(일부 3배 이상) 단일 실행으로
  ±% 변화를 판정하지 않는다. 성능 판정은 **격리 A/B 다회 반복**으로만, 40개 순차 벤치는
  feasibility·시간 규율 게이트로만 쓴다.
- **오버피팅 금지** — 파라미터는 연습 40개의 우연한 분포가 아니라 인스턴스 특성으로 스케일링한다.

## 서버 / 환경 제약

AMD Threadripper PRO, Ubuntu 24.04, **≤4 CPU 코어(400%)**, ≤16GB RAM, 인터넷 없음,
상위 디렉토리 접근 불가(상대경로만). `firejail`+`cpulimit` 로 격리하며 bay는 독립 크레인이라
**bay별 병렬화**가 가능하다. 사용 가능: python 3.12, numba, numpy, gurobipy, shapely 등(GPU 없음).
