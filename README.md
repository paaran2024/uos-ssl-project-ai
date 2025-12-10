# CATANet 모델 프루닝 및 최적화 프로젝트

## 개요

이 프로젝트는 이미지 복원 작업을 위한 딥러닝 모델인 CATANet의 프루닝(가지치기)을 통해 모델을 경량화하고 성능을 최적화하는 과정을 담고 있습니다. 모델의 특정 구성 요소를 제거하여 연산 비용(MACs)과 파라미터 수를 줄이고, 파인튜닝(fine-tuning)을 통해 성능 손실을 최소화하는 것을 목표로 합니다.

## 프로젝트 구조

```
.
├── basicsr/         # 이미지 복원 작업을 위한 기본 프레임워크 (BasicSR)
│   ├── archs/       # 모델 아키텍처 (CATANet 등)
│   ├── data/        # 데이터셋 로더 및 전처리
│   ├── losses/      # 손실 함수
│   ├── models/      # 모델 정의 (CATANet 모델 포함)
│   └── utils/       # 유틸리티 함수
├── prune/           # 모델 프루닝 관련 핵심 로직
│   ├── head_pruning.py       # Head 프루닝 로직
│   ├── vision_pruning.py     # Vision 모듈 프루닝 로직
│   └── main_prune.py         # 프루닝 프로세스 실행
├── analysis/        # 프루닝된 모델의 성능 분석 스크립트
│   ├── calculate_performance.py # 성능 지표 계산
│   └── visualize_tradeoff.py    # 연산량-성능 트레이드오프 시각화
├── datasets/        # 학습 및 평가에 사용되는 데이터셋
├── weights/         # 원본 및 프루닝된 모델 가중치
├── config_catanet.yml        # CATANet 모델 및 프루닝 설정 파일
├── requirements.txt          # 프로젝트 의존성 라이브러리 목록
├── run_catanet_pruning.py    # 프루닝 프로세스 실행 스크립트
└── finetune_pruned_model.py  # 프루닝된 모델 파인튜닝 스크립트
```

### 주요 디렉토리 및 파일 설명

-   **`basicsr/`**: 이미지 초해상도(Super-Resolution)와 같은 복원 작업을 위한 오픈소스 프레임워크인 [BasicSR](https://github.com/XPixelGroup/BasicSR)을 기반으로 합니다. CATANet 모델의 아키텍처, 데이터 처리, 학습 로직 등이 포함되어 있습니다.
-   **`prune/`**: 모델의 Head와 Vision Transformer 블록을 프루닝하기 위한 핵심 코드가 위치합니다.
-   **`analysis/`**: 프루닝된 모델의 성능(PSNR, SSIM 등)을 계산하고, 연산량(MACs)과 성능 간의 관계를 시각화하는 스크립트를 포함합니다.
-   **`run_catanet_pruning.py`**: `config_catanet.yml` 설정에 따라 프루닝 중요도 점수를 계산하고 모델을 프루닝하는 메인 스크립트입니다.
-   **`finetune_pruned_model.py`**: 프루닝 후 성능 저하를 복구하기 위해 모델을 다시 학습(파인튜닝)하는 스크립트입니다.
-   **`config_catanet.yml`**: 모델 경로, 데이터셋, 프루닝 비율, 파인튜닝 하이퍼파라미터 등 프로젝트의 모든 설정을 관리하는 파일입니다.

## 실행 방법

1.  **의존성 설치**
    ```bash
    pip install -r requirements.txt
    ```

2.  **모델 프루닝 실행**
    `config_catanet.yml` 파일에서 프루닝 관련 설정(예: 프루닝 비율)을 조정한 후, 다음 명령어를 실행합니다.
    ```bash
    python run_catanet_pruning.py
    ```
    이 과정은 프루닝된 모델 파일(`catanet_pruned.pth`)과 마스크 파일(`catanet_pruning_masks.pth`)을 `weights/` 디렉토리에 생성합니다.

3.  **모델 파인튜닝**
    프루닝된 모델의 성능을 복구하기 위해 파인튜닝을 진행합니다.
    ```bash
    python finetune_pruned_model.py
    ```

4.  **성능 평가 및 분석**
    파인튜닝된 모델의 성능을 평가하고 결과를 분석합니다.
    ```bash
    python analysis/calculate_performance.py
    ```

## 설정

주요 설정은 `config_catanet.yml` 파일에서 변경할 수 있습니다. 이 파일을 통해 데이터셋 경로, 모델 가중치 경로, 프루닝 및 파인튜닝 파라미터를 제어할 수 있습니다.
