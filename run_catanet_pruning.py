# -*- coding: utf-8 -*-
import argparse
import yaml
import torch
import os

# --- 커스텀 모듈 및 OPTIN 라이브러리 임포트 ---
from scripts.load_catanet import get_catanet_teacher_model
from prune.main_prune import pruneModel
from evals.gen_eval import evalModel, _apply_pruning_in_memory # MODIFIED: Import the correct pruning function
from data.scripts.gen_dataset import generateDataset
from utils.utility import calculateComplexity

# 이 스크립트는 CATANet-L 모델에 대한 프루닝 전체 과정을 실행하는 메인 스크립트입니다.

def apply_and_save_pruned_model(model, pruning_params, save_path):
    """
    MODIFIED: Applies all pruning masks (head and neuron) and saves the resulting pruned state_dict.
    This now uses the same logic as the evaluation step to ensure consistency.
    """
    print(f"--- 프루닝된 모델 가중치 저장 시작 ---")
    print(f"저장 경로: {save_path}")

    # Use the tested function from eval to get the correctly pruned state_dict
    pruned_state_dict = _apply_pruning_in_memory(model, pruning_params)

    # Save the pruned state_dict.
    # It's good practice to save it in a dictionary, similar to the original model checkpoints.
    # The finetune script will look for the 'params' key.
    torch.save({'params': pruned_state_dict}, save_path)
    
    print(f"--- 프루닝된 모델 가중치 저장 완료 ---")


def main():
    # --- 1. 설정 파일 로드 ---
    parser = argparse.ArgumentParser(description="CATANet-L 모델을 위한 커스텀 프루닝 실행 스크립트")
    parser.add_argument("--config", required=True, help="프루닝 설정을 담은 YAML 파일 경로")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    
    # YAML 파일 내용을 args 객체에 병합합니다.
    for key, value in yaml_config.items():
        setattr(args, key, value)

    print("--- 설정 파일 로드 완료 ---")
    print(f"모델: {args.model_name}, 데이터셋: {args.dataset}")
    print(f"프루닝 제약 조건 (MAC): {args.mac_constraint}")
    print("--------------------------\n")

    # --- 2. 교사 모델(CATANet-L) 로드 ---
    teacher_model = get_catanet_teacher_model(weights_path=args.weights_path, upscale=args.scale)
    
    model_config = {
        "num_attention_heads": teacher_model.heads,
        "intermediate_size": teacher_model.mlp_dim,
        "hidden_size": teacher_model.dim,
        "num_hidden_layers": teacher_model.block_num,
    }
    args.model_config = model_config
    
    print(f"--- 교사 모델 '{args.model_name}' 로드 완료 ---")
    num_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"모델 파라미터 수: {num_params / 1e6:.2f}M")
    print("-----------------------------------\n")

    # --- 3. 데이터셋 로드 ---
    train_dataset, val_dataset, args = generateDataset(args)
    print("--- 데이터셋 로드 완료 ---")

    # --- 4. 프루닝 실행 ---
    prunedProps = {
        "num_att_head": model_config["num_attention_heads"],
        "inter_size": model_config["intermediate_size"],
        "hidden_size": model_config["hidden_size"],
        "num_layers": model_config["num_hidden_layers"],
    }
    
    print("--- 모델 프루닝 시작 ---")
    pruningParams, baselineComplexity, prunedComplexity = pruneModel(args, teacher_model, train_dataset, model_config)
    print("--- 모델 프루닝 완료---")

    # --- 5. 프루닝된 모델 및 마스크 저장 ---
    # MODIFIED: Save the pruning parameters (masks) as well
    pruned_model_save_path = os.path.join('weights', 'catanet_pruned.pth')
    pruning_masks_save_path = os.path.join('weights', 'catanet_pruning_masks.pth')
    
    print(f"프루닝 마스크를 '{pruning_masks_save_path}'에 저장합니다...")
    torch.save(pruningParams, pruning_masks_save_path)
    print("마스크 저장 완료.")

    apply_and_save_pruned_model(teacher_model, pruningParams, pruned_model_save_path)
    print("\n")

    # --- 6. 결과 평가 ---
    print("--- 프루닝된 모델 성능 평가 시작 ---")
    if isinstance(baselineComplexity, dict): baselineComplexity = baselineComplexity.get('MAC', 1)
    if isinstance(prunedComplexity, dict): prunedComplexity = prunedComplexity.get('MAC', 1)
    flop_reduction_amount = 100 - (prunedComplexity / baselineComplexity * 100.0)
    print(f"FLOPs 감소율: {flop_reduction_amount:.2f}%")
    
    baselinePerformance, finalPerformance = evalModel(args, teacher_model, train_dataset, val_dataset, pruningParams, prunedProps)
    print("--- 성능 평가 완료---")

    # --- 7. 최종 결과 출력 ---
    print("--- 최종 결과 ---")
    print(f"원본 모델 성능: {baselinePerformance}")
    print(f"프루닝된 모델 성능: {finalPerformance}")
    print(f"FLOPs 감소율: {flop_reduction_amount:.2f}%")
    print(f"FLOPs 비율: {prunedComplexity / baselineComplexity * 100.0:.2f}%")
    print("------------------")

if __name__ == "__main__":
    main()
