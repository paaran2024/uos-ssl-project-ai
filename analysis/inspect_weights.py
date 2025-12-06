import sys
import os

# Add the parent directory (ai) to the Python path for consistency
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse

def inspect_model_weights(file_path):
    """
    Loads a .pth file and inspects its state_dict to calculate weight sparsity.
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일 '{file_path}'를 찾을 수 없습니다. 현재 작업 디렉토리: {os.getcwd()}")
        return

    print(f"\n--- 가중치 파일 분석 중: {os.path.basename(file_path)} ---")
    
    state_dict = torch.load(file_path, map_location='cpu')

    # The weights might be nested under a 'params' key
    if 'params' in state_dict:
        print("'params' 키 아래에 있는 가중치를 사용합니다.")
        state_dict = state_dict['params']
    elif 'params_g' in state_dict:
        print("'params_g' 키 아래에 있는 가중치를 사용합니다.")
        state_dict = state_dict['params_g']

    total_params = 0
    nonzero_params = 0

    for param_name, param_tensor in state_dict.items():
        if param_tensor.is_floating_point(): # Only count floating point parameters
            total_params += param_tensor.numel()
            nonzero_params += param_tensor.count_nonzero().item()

    if total_params == 0:
        print("분석할 파라미터를 찾을 수 없습니다.")
        return

    sparsity = (1 - (nonzero_params / total_params)) * 100
    
    print(f"총 파라미터 수         : {total_params:,}")
    print(f"0이 아닌 파라미터 수    : {nonzero_params:,}")
    print(f"희소성 (Sparsity)        : {sparsity:.2f}% (값이 0인 파라미터의 비율)")
    print("-" * (40 + len(os.path.basename(file_path))))


def main():
    parser = argparse.ArgumentParser(description="PyTorch 가중치 파일(.pth)의 희소성을 분석합니다.")
    parser.add_argument(
        '--files', 
        nargs='+', 
        required=True, 
        help="분석할 .pth 파일의 경로 목록. 예: --files weights/original.pth weights/pruned.pth"
    )
    
    # --- 실행 위치 보정 ---
    # 스크립트의 CWD를 상위 디렉토리(ai)로 변경하여 상대 경로가 올바르게 작동하도록 합니다.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(base_dir)
    print(f"Working directory changed to: {os.getcwd()}")
    
    args = parser.parse_args()

    for file_path in args.files:
        inspect_model_weights(file_path)

if __name__ == '__main__':
    main()