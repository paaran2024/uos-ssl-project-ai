import torch
import torch.nn as nn
from models.catanet_arch import CATANet
import os

# 이 스크립트는 CATANet-L 모델을 로드하는 커스텀 로더 역할을 합니다.
# OPTIN 프레임워크의 gen_vision_model.py를 직접 수정하는 대신,
# ai 폴더 내에서 독립적으로 모델 로딩 로직을 관리합니다.

def get_catanet_teacher_model(weights_path: str = None, upscale: int = 4):
    """
    CATANet-L 교사 모델을 생성하고 (선택적으로) 사전 훈련된 가중치를 로드합니다.

    Args:
        weights_path (str, optional): 사전 훈련된 가중치 파일(.pth)의 경로.
                                      None이면 무작위로 초기화된 모델을 반환합니다.
        upscale (int): 모델의 업스케일링 비율 (예: 2, 3, 4). CATANet 아키텍처에 전달됩니다.

    Returns:
        torch.nn.Module: CATANet-L 모델 인스턴스.
    """
    print(f"CATANet-L 모델을 생성합니다. Upscale 비율: {upscale}")

    # 1. CATANet 모델 아키텍처 인스턴스 생성
    model = CATANet(in_chans=3, upscale=upscale)

    # 2. 사전 훈련된 가중치 로드 (선택 사항)
    if weights_path:
        if os.path.exists(weights_path):
            print(f"사전 훈련된 가중치를 로드합니다: {weights_path}")
            loaded_state = torch.load(weights_path, map_location='cpu')
            
            # MODIFIED: Check if weights are nested under a 'params' key
            if 'params' in loaded_state:
                print("'{params}' 키 아래에 있는 가중치를 추출합니다.")
                state_dict = loaded_state['params']
            else:
                state_dict = loaded_state
                
            model.load_state_dict(state_dict)
            print("가중치 로드 성공.")
        else:
            print(f"경고: 지정된 가중치 파일 '{weights_path}'를 찾을 수 없습니다. "
                  "무작위로 초기화된 모델을 사용합니다.")
    else:
        print("가중치 경로가 제공되지 않았습니다. 무작위로 초기화된 모델을 사용합니다.")

    # 모델을 평가 모드로 설정합니다. (프루닝 시에는 중요)
    model.eval()

    return model