# BioLiP and neural models: held-out ranking comparison

## 실행

이 비교는 **여기 CPU 서버에서만** 실행합니다. 기존 예측값을 사용하므로 GPU,
새 학습, 새 임베딩 생성, sklearn, torch가 필요 없습니다. Python 3.7 이상과
numpy/pandas만 필요합니다. 디렉터리 재귀 탐색은 하지 않습니다.

```bash
cd /disk9/13.Heesu_Allostery
BOOTSTRAP_WORKERS=8 bash analysis/biolip_neural_ranking_comparison/scripts/run_local.sh
```

데이터는 작아 8 workers면 충분하며 `BOOTSTRAP_WORKERS=1`도 가능합니다.
모든 worker는 BLAS 한 thread만 사용합니다. 완료 후 같은 명령은 해시를 확인하고
결과를 재사용합니다. 도중 중단 후 재실행하면 bootstrap 계산은 처음부터 하지만
학습·GPU 추론은 어떤 경우에도 하지 않습니다. 코드/입력이 바뀌면 중단하며,
기존 contract를 삭제해서 우회하지 마세요. 계산 중에는 같은 코드 파일을 수정하지 마세요.

## 고정한 비교 대상

- BioLiP 최신 consensus-60 재분석의 reference 1,322쌍에서 시작합니다.
- every-pair에서 학습한 8개 모델의 **기존 Pfam held-out** 예측과 exact UniProt /
  full InChIKey로 연결한 공통 **987쌍·278 단백질**을 사용합니다.
- NN은 기존 세 seed 평균입니다. 각 pair는 해당 family가 outer test인 예측입니다.
- BioLiP도 자신의 `family_held_out` OOF 점수를 사용합니다. 학습 전체에 refit한
  배포용 점수와 섞지 않습니다. 두 파이프라인의 fold 배정과 학습 자료는 동일하지
  않으므로 동일 학습량을 통제한 모델 우열 실험으로 해석하지 않습니다.
- 사전 고정한 네 BioLiP 방법: `RAW:RAW_D`, `KDE_LEGACY:D+AR`,
  `KDE_LEGACY:D+MW`, `QNB:D+AR`. 결과를 보고 46개 중 좋은 것만 고르지 않습니다.
- 후보 2,890쌍의 신규 신경망 추론은 **이번 패키지에 포함하지 않습니다**.
- 참조의 1,158쌍은 every-pair 전체 학습 자료에도 있습니다. 전체 학습 배포 모델을
  사용하지 않는 이유입니다. 987쌍은 그중 기존 OOF 예측이 있는 부분이며,
  나머지 171쌍은 training-only여서 해당 OOF가 없고 164쌍은 기존 OOF에 없는 pair입니다.
  `REFERENCE_COVERAGE.tsv`에 이 구분을 남깁니다. 이 분석은 공유 참조 사례를
  이용한 **방법 간 비교이지 독립적인 외부 검증은 아닙니다**.

## 통계 정의

1. **동일 단백질 안의 Spearman 상관**: 3쌍 이상이고 양쪽 점수가 모두 변하는
   단백질에서 평균동점순위 Pearson 상관을 구한 후 단백질별 동일 가중 평균.
   2쌍이면 상관이 사실상 ±1이 되기 때문에 주 상관 분석에 넣지 않습니다.
   상수 점수는 undefined/NA이며 0으로 채우지 않습니다. 유효/제외 단백질·행수를 보고합니다.
2. **상위 1개 일치**: 2쌍 이상 단백질에서 계산합니다. 두 모델의 top-score 동점
   집합 A, B에서 각각 균등하게 하나를 고르면 같을 확률 `|A∩B|/(|A||B|)`입니다.
   row ID로 임의 선택하거나 protein-only의 전부 동점을 100% 일치로 세지 않습니다.
   후보 수 n에서 무작위 일치 기대값은 1/n이며 원값과 원값−1/n을 모두 보고합니다.
3. **라벨 AUROC**: 같은 단백질에 두 라벨이 있는 **81 단백질·409쌍**에서
   단백질별 AUROC를 평균합니다. NN 8개와 BioLiP 네 방법 모두 같은 지지 집합입니다.
4. **대조군 대비 차이**: 각 pair 모델에서 ligand-only를 뺀 상관, top-1 일치,
   AUROC를 계산합니다. 상관 차이는 두 통계가 모두 정의되는 단백질의 교집합에서
   계산하며 서로 다른 유효 집합의 평균을 빼지 않습니다. 단백질을 고정한 분석이므로
   protein-only는 상관 NA, top-1 1/n, AUROC .5인 구현 대조군입니다.

95% percentile CI는 **10,000회 paired family-cluster bootstrap**입니다.
두 데이터셋의 family 그룹이 다를 수 있어, 어느 쪽에서든 같은 family로 연결된
단백질은 하나로 합친 union component를 재표집합니다. 이는 단백질을 서로
독립이라고 놓는 것보다 보수적인 단위이며, 원래 두 family 정의를 넘어 모든
생물학적 의존성을 제거한다는 주장은 아닙니다. family가 여러 번 뽑히면 그 안의
모든 단백질 기여를 그 횟수만큼 유지합니다. 점수는 고정되므로 재학습 변동이나
seed 변동까지 포함한 CI가 아닙니다.

모든 비교는 같은 추출 multiplicity를 사용합니다. 유효 복제본이 95% 미만이면
최종 PASS를 내지 않습니다. 지지가 전혀 없거나 family가 2개 미만이면 CI 대신
사유를 기록합니다. CI는 **개별 비교용이며 다중비교 보정 전**입니다. 양의 CI를
여러 비교에서 탐색한 뒤 확증적 유의성으로 표현하지 마세요.

`POOLED_DESCRIPTIVE.tsv`의 전체 AUROC/상관은 보조 기술통계이며 CI를 붙이지
않습니다. 서로 다른 표적과 서로 다른 fold 모델의 점수 스케일이 섞이기 때문입니다.
순위 상관은 공통 라벨 차이나 물성 특징으로도 생길 수 있습니다. BioLiP 거리와
참조 부위 정의의 의존성도 남으므로 높은 상관만으로 기능적 allostery나 모델의
상호작용 이해를 입증하지 않습니다.

## 산출물

- `validation/CPU_CONTRACT.json`: 입력·코드 해시, 방법·분모·bootstrap 설정.
- `data/MATCHED_SCORES.tsv.gz`: 모든 모델/방법을 같은 pair에 연결한 점수표.
- `data/REFERENCE_COVERAGE.tsv`, `data/PROTEIN_SUPPORT.tsv`: 제외 사유와 지지 집합.
- `cpu_output/WITHIN_PROTEIN_METRICS_AND_CI.tsv`: 주 통계와 paired 차이·95% CI.
- `cpu_output/PER_PROTEIN_AGREEMENT.tsv.gz`: 단백질별 상관/동점 처리/일치율.
- `cpu_output/PER_PROTEIN_LABEL_AUROC.tsv`, `POOLED_DESCRIPTIVE.tsv`: 라벨 비교와 보조 통계.
- `cpu_output/BOOTSTRAP_DRAWS.npz`: 모든 비교에 공통 사용한 family 추출 횟수.
- `cpu_output/REPORT.md`, `VALIDATION.json`: 해석 범위·결과 파일 해시·완료 상태.

이 코드는 원고, 기존 benchmark/BioLiP 결과, 웹 DB를 변경하지 않습니다.
