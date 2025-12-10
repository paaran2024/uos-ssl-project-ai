"""
이 스크립트는 SR을 위한 PyTorch 데이터 로더(DataLoader)를 생성합니다.
"""

import torch
from os import path as osp
from copy import deepcopy

# `basicsr` 라이브러리는 이제 `ai/basicsr`에 로컬 디렉토리로 존재하므로,
# 직접 임포트할 수 있습니다.
from basicsr.data import build_dataloader, build_dataset


def generateDataset(args):
    """
    제공된 설정에 따라 훈련 및 검증 데이터 로더를 생성합니다.

    이 함수는 BasicSR 데이터 로딩 파이프라인의 래퍼(wrapper) 역할을 합니다.
    YAML 파일로부터 채워진 `args` 객체에서 데이터셋 설정을 읽어와,
    적절한 데이터셋 및 데이터 로더 인스턴스를 생성하고 반환합니다.

    인자(Args):
        args (Namespace): YAML 파일로부터 로드된 설정을 담고 있는 객체입니다.
                          반드시 'datasets' 속성(훈련 및/또는 검증 설정 포함)과
                          'scale' 속성을 포함해야 합니다.

    반환(Returns):
        tuple: 다음을 포함하는 튜플입니다:
            - train_loader (DataLoader): 훈련 세트를 위한 데이터 로더.
            - val_loader (DataLoader): 검증 세트를 위한 데이터 로더.
            - args (Namespace): 원본 `args` 객체 (수정되지 않음).
    """
    train_loader = None
    val_loader = None

    # --- 훈련 데이터 로더 생성 ---
    if hasattr(args, 'datasets') and 'train' in args.datasets:
        train_opt = deepcopy(args.datasets['train'])
        train_opt['phase'] = 'train'
        train_opt['scale'] = args.scale # 모델의 스케일 팩터 전달
        
        if not torch.cuda.is_available():
            train_opt['pin_memory'] = False
            train_opt['prefetch_mode'] = 'cpu'
        
        train_set = build_dataset(train_opt)
        train_loader = build_dataloader(
            train_set,
            train_opt,
            num_gpu=1,  # 프루닝 스크립트용 단일 GPU 가정
            dist=False, # 분산 훈련 아님을 가정
            seed=getattr(args, 'seed', None) # 사용 가능한 경우 설정에서 시드 사용
        )
        print(f"'{train_opt['name']}' 데이터셋을 위한 훈련 데이터 로더를 성공적으로 생성했습니다. 이미지 수: {len(train_set)}개.")

    # --- 검증 데이터 로더 생성 ---
    if hasattr(args, 'datasets') and 'val' in args.datasets:
        val_opt = deepcopy(args.datasets['val'])
        val_opt['phase'] = 'val'
        val_opt['scale'] = args.scale # 모델의 스케일 팩터 전달
        
        if not torch.cuda.is_available():
            val_opt['pin_memory'] = False
            val_opt['prefetch_mode'] = 'cpu'

        val_set = build_dataset(val_opt)
        val_loader = build_dataloader(
            val_set,
            val_opt,
            num_gpu=1,
            dist=False,
            seed=getattr(args, 'seed', None)
        )
        print(f"'{val_opt['name']}' 데이터셋을 위한 검증 데이터 로더를 성공적으로 생성했습니다. 이미지 수: {len(val_set)}개.")
    
    if train_loader is None and val_loader is None:
        raise ValueError("Could not create any dataloader. Check your 'datasets' configuration in the YAML file.")
    
    return train_loader, val_loader, args