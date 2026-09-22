# Design of an Embryo Grading Model using ResNet-50-based Multi-Task Classification

단일 배아 이미지에서 `Stage`, `ICM`, `TE`를 동시에 예측하는 PyTorch 기반 멀티태스크 분류 프로젝트.

## Architecture

- ImageNet pretrained ResNet-50 backbone
- 2,048D feature to shared bottleneck
- Independent heads for Stage, ICM, and TE
- Optional task adapters and uncertainty-based task weighting
- Weighted sampling, class weights, focal loss, label smoothing, and augmentation options
- Freeze-to-fine-tune training with backbone learning-rate warmup

## Files

```text
dataset.py                    # image/JSON matching and transforms
model.py                      # multi-task model
train.py                      # training and fine-tuning
eval.py                       # checkpoint evaluation
run_sweep.py                  # hyperparameter sweep
eval_sweep_runs.py            # batch evaluation
report_sweep.py               # sweep summary generation
generate_performance_materials.py
performance_materials/        # aggregate tables and figures
models/best_finetune.pt      # trained final fine-tuning checkpoint
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data Description

| 항목 | 설명 |
|---|---|
| 입력 데이터 | 배아 현미경 이미지 |
| 라벨 | JSON 형식의 Stage, ICM, TE 등급 |
| 데이터 분할 | Training set, Validation set |
| 전처리 | 이미지-라벨 매칭, 224×224 크기 변환, ImageNet 평균·표준편차 정규화 |
| 예측 과제 | Stage, ICM, TE를 독립적인 출력 헤드로 예측하는 멀티태스크 분류 |

## Trained Model

`models/best_finetune.pt`는 최종 fine-tuning 단계에서 저장한 학습 모델. 파일 크기가 100MB를 초과하므로 Git LFS로 관리함.

## Result Snapshot

Validation example (1,005 samples):

- Stage accuracy: 99.20%
- ICM accuracy: 54.93%
- TE accuracy: 59.40%

Across 13 completed sweep runs, the best `min(ICM, TE)` score was 0.5542, and Stage classification was stable near 0.99. These results do not indicate clinical or educational efficacy.
