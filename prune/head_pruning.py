"""
[리팩토링 노트]
이 스크립트는 `CATANet` 모델 및 `BasicSR` 데이터 로딩 파이프라인과
호환되도록 리팩토링되었습니다. 변경 사항은 `vision_pruning.py`와 유사합니다.

[원본과의 주요 차이점]
1.  Hooking 클래스:
    -   이전: `_hooks.py`의 `ModelHooking`을 임포트했습니다.
    -   현재: `catanet_hooks.py`의 `CATANetModelHooking`을 임포트합니다.

2.  데이터 처리:
    -   이전: 데이터 로더가 `(이미지, 레이블)` 튜플을 반환한다고 가정했습니다.
    -   현재: `BasicSR` 데이터 로더의 `{'lq': ..., 'gt': ...}` 딕셔너리 형식을
      올바르게 처리하고 `lq_tensor`를 추출합니다.

3.  모델 및 데이터 전달:
    -   이전: `forwardPass`에 딕셔너리를, hooking 클래스에 `CATANetModel` 객체를
      전달했습니다.
    -   현재: hooking 클래스에 실제 네트워크인 `model`을, `forwardPass`에
      `lq_tensor`를 직접 전달합니다.
      
4.  헤드 프루닝 비호환성:
    -   이 스크립트가 헤드 중요도 점수를 *계산*하지만, `CATANet` 아키텍처는
      `head_mask`를 모델에 전달하여 동적으로 프루닝하는 메커니즘을 지원하지
      않는다는 점을 강조하는 주석이 추가되었습니다. 가중치를 0으로 만드는
      실제 헤드 프루닝 적용 또한 후속 평가 로직에 구현되어 있지 않으므로,
      이 계산은 순수 분석용으로만 기능합니다.
"""
import torch
from prune.loss_components import KLDiv, manifold_Distillation
from queue import PriorityQueue
from torch.nn import functional as F
from tqdm import tqdm
import time
from utils.catanet_hooks import CATANetModelHooking # MODIFIED: Use the new CATANet-specific hook
import numpy as np
import os
import pickle


@torch.no_grad()
def Prune(args, prunedProps, FullHeadMasking, lq_tensor, model, base_layer_wise_output, base_logit_output, base_grad_output=None):
    
    # 이 함수는 각 어텐션 헤드의 중요도를 계산합니다.
    # 핵심 로직은 OPTIN 것을 유지하되, 모델과의 상호작용은 리팩토링되었습니다.
    
    PerHeadMasking = -1*(torch.eye(prunedProps["num_att_head"]) - 1) 
    globalHeadRanking = PriorityQueue() 
    
    for layer in range(prunedProps["num_layers"]):
        print(f"Layer Sample: {layer} / {prunedProps['num_layers']}")

        for head in tqdm(range(prunedProps["num_att_head"]), desc=f"Analyzing Heads in Layer {layer}"):
            then = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            FullHeadMasking[layer] = PerHeadMasking[head]
            
            # 중요 참고: 원본 OPTIN 프레임워크는 `head_mask` 인자를 모델의
            # forward pass에 직접 전달하여 헤드 프루닝을 적용했습니다.
            # CATANet 아키텍처는 이러한 동적 마스킹을 지원하지 않습니다.
            # 따라서 이 코드가 헤드 중요도 점수를 *계산*하지만, 이 forward pass 동안
            # 마스크가 실제로 적용되지는 않습니다. '마스킹'은 기준 실행과 단일 헤드의
            # 기여가 가상으로 제거되었을 경우의 실행을 비교함으로써 시뮬레이션됩니다.
            # 가중치를 0으로 만드는 최종 프루닝 적용은 별도로 구현해야 합니다.
            # 원본 `run_catanet_pruning.py` 역시 이 단계를 생략하고 뉴런 프루닝만
            # 적용했습니다.
            maskingProps = {
                "state":"head",
                "layer": layer,
                "mask": FullHeadMasking
            }
            
            modelObject = CATANetModelHooking(args=args, model=model, maskProps=maskingProps, disable_grad=True)
            with torch.no_grad():
                # MODIFIED: Pass the lq_tensor directly
                current_logit_output, current_layer_wise_output = modelObject.forwardPass(lq_tensor)
            modelObject.purge_hooks()
            
            MMDLayerResults = 0
            KLErr = 0
            
            # --- 아래의 손실 계산 로직은 OPTIN에서 가져온 것입니다 ---
            if args.loss_type == "MMD" or args.loss_type == "MMD+KL":
                for idx in range(len(base_layer_wise_output)):
                    if idx > layer or ((layer == prunedProps["num_layers"]-1) and layer == idx and args.head_include_fin_layer_mmd):
                        with torch.no_grad():
                            err = manifold_Distillation(args, base_layer_wise_output[idx], current_layer_wise_output[idx])
                            MMDLayerResults += err
            
            if args.loss_type == "KL" or args.loss_type == "MMD+KL":    
                # MODIFIED: Removed temp=args.temp as it's no longer used
                KLErr = KLDiv(base_logit_output, current_logit_output)
            
            MMDResults = MMDLayerResults + KLErr
            # --- OPTIN 손실 로직 끝 ---
                
            globalHeadRanking.put((MMDResults.detach().cpu(), layer, head, "head"))

    return globalHeadRanking, None # 사용하지 않는 두 번째 변수에 대해 None 반환

            
def pruneHead(model, train_dataset, args, prunedProps):

    # 이 함수는 이제 BasicSR 데이터 로더와 CATANet 모델을 올바르게 처리합니다.
    
    storage_path_body = f"./storage/{args.task_name}/{args.dataset}/{args.model_name}/head_ranking_body.pkl"
    os.makedirs(os.path.dirname(storage_path_body), exist_ok=True)
    
    prunedProps["lambda"] = args.lambda_contribution
    
    if not os.path.isfile(storage_path_body):
    
        FullHeadMasking = torch.ones((prunedProps["num_layers"],prunedProps["num_att_head"]))
        torch.backends.cudnn.benchmark = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # MODIFIED: 단일 배치를 가져와 'lq' 텐서를 추출합니다
        print("분석을 위해 데이터 로더에서 배치를 가져오는 중...")
        batch = next(iter(train_dataset))
        lq_tensor = batch['lq'].to(device, non_blocking=True)
            
        # MODIFIED: 이 함수에 전달된 `model`이 실제 네트워크이므로 직접 사용합니다.
        model.eval()
        
        print("가지치기 전 기준 출력 계산 중...")
        # 참고: 기준선 계산을 위해 maskProps 없이 전달합니다.
        modelObject = CATANetModelHooking(args=args, model=model, maskProps=None, disable_grad=True)
        
        # MODIFIED: lq_tensor를 직접 전달합니다
        base_logit_output, base_layer_wise_output = modelObject.forwardPass(lq_tensor)
        modelObject.purge_hooks()
        
       
        print("헤드 중요도 계산 시작...")
        # MODIFIED: lq_tensor와 실제 네트워크를 전달합니다
        globalHeadRanking, _ = Prune(args, prunedProps, FullHeadMasking, lq_tensor, model, base_layer_wise_output, base_logit_output)

        exportglobalHeadRanking = []
        while not globalHeadRanking.empty():
            exportglobalHeadRanking.append(globalHeadRanking.get())
        
        with open(storage_path_body, 'wb') as f:
            pickle.dump(exportglobalHeadRanking, f)
            
    else:
        print(f"{storage_path_body}에서 기존 헤드 랭킹을 로드하는 중")
        with open(storage_path_body, 'rb') as f:
            exportglobalHeadRanking = pickle.load(f)
    
    return {"final_head_ranking": exportglobalHeadRanking}