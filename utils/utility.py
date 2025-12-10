"""
[리팩토링 노트]
이 스크립트는 모델의 MACs(Multiply-Accumulate Operations) 계산을 위한 래퍼(wrapper)
함수를 제공합니다. `OPTIN` 프레임워크에서 `CATANet` 프로젝트로 옮겨오면서
불필요한 코드와 깨진 임포트(import)를 정리했습니다.

[원본과의 주요 차이점]
1.  불필요한 임포트 및 정의 제거: 언어 관련 Vit 관련 작업 제거
"""
import torch
from utils.mac_complexity import compute_base_mac, compute_pruned_mac

def calculateComplexity(args, model, train_dataset, prunedProps, pruningParams={}):
    """
    기준 모델과 가지치기된 모델의 MACs(Multiply-Accumulate Operations)를
    계산하는 래퍼 함수입니다.
    """
    
    original_complexity = {
        "MAC": 1,
        "Latency": 1
    }
    
    pruned_complexity = {
        "MAC": 1,
        "Latency": 1
    }
    
    
    original_mac = compute_base_mac(args, prunedProps, skipConv=False)
    pruned_mac = compute_pruned_mac(args, prunedProps, pruningParams, skipConv=False)
    
   
    original_complexity["MAC"] = original_mac
    pruned_complexity["MAC"] = pruned_mac
    
    
    print(f"Original MACs: {original_complexity['MAC'] / 1e9:.2f} G-MACs")
    print(f"Pruned MACs: {pruned_complexity['MAC'] / 1e9:.2f} G-MACs")
    return original_complexity, pruned_complexity
