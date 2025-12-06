# CATANet-L 프루닝 프레임워크 아키텍처

## 1. 개요 (Overview)

이 문서는 커스텀 모델인 `CATANet-L`을 `OPTIN` 프레임워크의 프루닝 로직을 사용하여 최적화하기 위해 구축된 독립적인 실행 환경에 대해 설명합니다.

**핵심 목표:**
- 기존 참조 코드(`ai/references/`)를 전혀 수정하지 않고, `ai` 폴더 내에 독립적인 프레임워크를 구축합니다.
- `CATANet-L` 모델을 "교사(Teacher)" 모델로 사용하여 `OPTIN`의 프루닝 알고리즘을 적용합니다.
- 향후 다른 커스텀 모델에도 적용할 수 있는 확장 가능한 구조를 마련합니다.

## 2. 프레임워크 구성 파일 및 역할

프레임워크는 `ai` 폴더 내의 새로운 파일들로 구성되어 있으며, 각 파일의 역할은 다음과 같습니다.

### 📄 `ai/run_catanet_pruning.py`
- **역할**: **메인 실행 스크립트 (Entry Point)**
- **설명**: 전체 프루닝 과정을 시작하고 조율하는 핵심 파일입니다. 설정 파일을 읽고, 모델과 데이터를 로드한 후, `OPTIN`의 프루닝 및 평가 함수를 순서대로 호출합니다.

### 📄 `ai/config_catanet.yml`
- **역할**: **실행 설정 파일 (Configuration)**
- **설명**: 프루닝 과정에 필요한 모든 파라미터를 정의하는 YAML 파일입니다. 모델의 가중치 경로, 데이터셋 경로, 프루닝 강도(`mac_constraint`) 등 사용자가 수정해야 할 값들이 모두 여기에 포함됩니다.

### 📄 `ai/scripts/load_catanet.py`
- **역할**: **커스텀 모델 로더 (Model Loader)**
- **설명**: `CATANet-L` 모델을 불러오는 역할을 전담합니다. 모델의 아키텍처를 생성하고, `config_catanet.yml`에 명시된 경로의 사전 훈련된 가중치(`.pth` 파일)를 로드합니다.

### 📄 `ai/models/catanet_arch.py`
- **역할**: **모델 아키텍처 정의 (Model Architecture)**
- **설명**: `CATANet-L` 모델의 신경망 구조(PyTorch `nn.Module`)를 정의하는 파일입니다. 원본 `CATANet-main` 프로젝트에서 코드를 가져와 `BasicSR` 프레임워크에 대한 의존성을 제거하여 독립적으로 만들었습니다.

### 📁 `ai/weights/`
- **역할**: **가중치 파일 저장소 (Weights Storage)**
- **설명**: `CATANet-L` 모델을 훈련시킨 후 생성되는 사전 훈련된 가중치 파일(`.pth`)을 저장하는 폴더입니다.

---

## 3. 작동 순서 (Execution Flow)

사용자가 `python ai/run_catanet_pruning.py --config ai/config_catanet.yml` 명령을 실행했을 때의 내부 작동 순서는 다음과 같습니다.

1.  **`[run_catanet_pruning.py]`**: 스크립트가 시작되고, `--config` 인자로 전달된 `ai/config_catanet.yml` 파일의 경로를 인식합니다.

2.  **`[run_catanet_pruning.py]`**: YAML 파일을 파싱하여 모든 설정을 로드합니다. (e.g., `weights_path`, `mac_constraint`, `dataset` 정보 등)

3.  **`[run_catanet_pruning.py]`** ➡️ **`[load_catanet.py]`**: 설정된 `weights_path`와 `scale` 값을 인자로 `get_catanet_teacher_model` 함수를 호출합니다.

4.  **`[load_catanet.py]`** ➡️ **`[catanet_arch.py]`**: `CATANet` 클래스를 임포트하여 모델의 뼈대(인스턴스)를 생성합니다.

5.  **`[load_catanet.py]`**: `weights_path` 경로에 있는 가중치 파일(`.pth`)을 `torch.load`로 불러와 생성된 모델 인스턴스에 주입합니다.

6.  **`[load_catanet.py]`** ➡️ **`[run_catanet_pruning.py]`**: 사전 훈련된 `CATANet-L` 교사 모델 객체를 반환합니다.

7.  **`[run_catanet_pruning.py]`**: `OPTIN`의 `pruneModel` 함수가 요구하는 형식에 맞게 `model_config`, `prunedProps` 등의 인자들을 준비합니다.

8.  **`[run_catanet_pruning.py]`** ➡️ **`[ai/references/OPTIN/prune/main_prune.py]`**: 준비된 인자들과 교사 모델을 `pruneModel` 함수에 전달하여 프루닝을 실행합니다. 이 함수는 최적의 프루닝 마스크(`pruningParams`)를 계산하여 반환합니다.

9.  **`[run_catanet_pruning.py]`** ➡️ **`[ai/references/OPTIN/evals/gen_eval.py]`**: `pruningParams`를 `evalModel` 함수에 전달하여 원본 모델과 프루닝된 모델의 성능을 각각 평가하고 비교합니다.

10. **`[run_catanet_pruning.py]`**: 최종 성능 및 FLOPs 감소율 등의 결과를 터미널에 출력합니다.

---

## 4. 사용 방법 (How to Use)

1.  **데이터셋 준비**: `ai/config_catanet.yml` 파일의 `datasets` 섹션에 명시된 경로(예: `datasets/DIV2K/`)에 맞게 학습 및 검증 데이터셋을 위치시킵니다.

2.  **가중치 파일 경로 업데이트**: `CATANet-L` 모델 훈련이 완료되면, 생성된 가중치 파일(`.pth`)을 `ai/weights/` 폴더에 저장합니다. 그 후, `ai/config_catanet.yml` 파일의 `weights_path` 값을 실제 파일명(예: `ai/weights/catanet_l_final.pth`)으로 수정합니다.

3.  **프루닝 실행**: 터미널에서 다음 명령어를 실행합니다.
    ```bash
    python ai/run_catanet_pruning.py --config ai/config_catanet.yml
    ```
---

## 5. 고급 워크플로우 및 분석 스크립트

단순 프루닝 성능 평가를 넘어, 실제로 압축된 모델을 만들고 종합적으로 분석하기 위한 심화 워크플로우와 스크립트입니다.

### 5.1. 최종 모델 생성 및 분석 워크플로우

1.  **모델 프루닝 (`run_catanet_pruning.py`)**
    -   `config_catanet.yml`에 정의된 제약 조건에 따라 모델을 프루닝할 계획(마스크)을 세우고, 가중치가 0으로 처리된 모델(`catanet_pruned.pth`)과 프루닝 마스크(`catanet_pruning_masks.pth`)를 생성합니다.
    -   `mac_constraint` 값을 조절하여 프루닝 강도를 변경할 수 있습니다.

2.  **프루닝된 모델 파인튜닝 (`finetune_pruned_model.py`)**
    -   가중치가 0으로 설정된 `catanet_pruned.pth` 모델을 불러와 추가 학습(파인튜닝)을 진행하여, 프루닝으로 인한 성능 저하를 복구합니다.
    -   결과물로 파인튜닝이 완료된 희소 모델(`catanet_finetuned_feature_kd.pth`)이 생성됩니다.

3.  **압축 모델 재구성 (`analysis/rebuild_pruned_model.py`)**
    -   위 단계들에서 생성된 마스크와 파인튜닝된 가중치를 사용하여, **물리적으로 더 작은 새로운 모델**을 재구성합니다.
    -   이 과정을 통해 이론적인 FLOPs 감소가 아닌, 실제 FLOPs와 파라미터 수가 줄어든 최종 모델(`catanet_rebuilt.pth`)이 생성됩니다.

4.  **최종 성능 비교 (`analysis/calculate_performance.py`)**
    -   원본 모델과 최종 압축 모델을 가져와 PSNR, SSIM, FLOPs, 총 파라미터 수 등을 종합적으로 비교하여 보여줍니다.

---

### 5.2. 분석 스크립트 상세 설명

`ai/analysis/` 폴더에 위치한 스크립트들은 모델 분석 및 재구성을 담당합니다.

#### 📄 `analysis/rebuild_pruned_model.py`
-   **역할**: **압축 모델 재구성**
-   **설명**: 프루닝 마스크에 따라 더 작은 `nn.Linear`와 `Attention` 레이어를 가진 새로운 `CATANet` 아키텍처를 동적으로 생성하고, 파인튜닝된 가중치에서 0이 아닌 값들만 새로운 모델로 이식합니다.
-   **입력**: `config_catanet.yml`, `weights/catanet_pruning_masks.pth`, `weights/catanet_finetuned_feature_kd.pth`
-   **출력**: `weights/catanet_rebuilt.pth` (최종 압축 모델)

#### 📄 `analysis/calculate_performance.py`
-   **역할**: **종합 성능 비교**
-   **설명**: 두 모델(주로 원본과 최종 압축 모델)의 성능을 다각도로 비교합니다. `thop` 라이브러리를 사용하여 실제 FLOPs를 측정하고, 테스트 데이터셋으로 PSNR/SSIM을 계산합니다.
-   **입력**: `weights/CATANet-L_x2.pth`, `weights/catanet_rebuilt.pth` 등
-   **출력**: 터미널에 비교 테이블 출력

#### 📄 `analysis/inspect_weights.py`
-   **역할**: **가중치 희소성 검사**
-   **설명**: `.pth` 가중치 파일을 직접 열어, 전체 파라미터 중 실제로 0인 값의 비율(희소성, Sparsity)이 얼마나 되는지 확인합니다. 프루닝이 제대로 적용되었는지 검증하는 데 사용됩니다.
-   **입력**: 분석하고 싶은 `.pth` 파일 경로(들)
-   **출력**: 터미널에 희소성(%) 출력