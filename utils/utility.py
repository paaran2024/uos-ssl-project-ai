"""
[리팩토링 노트]
이 스크립트는 모델의 MACs(Multiply-Accumulate Operations) 계산을 위한 래퍼(wrapper)
함수를 제공합니다. `OPTIN` 프레임워크에서 `CATANet` 프로젝트로 옮겨오면서
불필요한 코드와 깨진 임포트(import)를 정리했습니다.

[원본과의 주요 차이점]
1.  불필요한 임포트 및 정의 제거:
    -   이전: 언어 작업(GLUE)과 관련된 `from data.scripts.glue import avg_seq_length`
      임포트와 `GLUE_TASKS` 리스트 정의가 포함되어 있었습니다.
    -   현재: `glue.py` 파일이 삭제되었고, 이 프로젝트에서는 언어 작업을 수행하지
      않으므로 관련 임포트와 정의를 제거했습니다. 이는 코드의 명확성을 높이고
      오류 가능성을 줄입니다.
"""
import torch
from utils.mac_complexity import compute_base_mac, compute_pruned_mac

# MODIFIED: 언어 작업과 관련이 없고 이 프로젝트에서 사용되지 않으므로
# GLUE_TASKS 리스트와 avg_seq_length 임포트를 제거했습니다.
# 이들이 의존하는 `glue.py` 파일도 삭제되었습니다.

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
