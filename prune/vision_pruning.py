"""
[리팩토링 노트]
이 스크립트는 `CATANet` 모델 및 `BasicSR` 데이터 로딩 파이프라인과
호환되도록 리팩토링되었습니다.

[원본과의 주요 차이점]
1.  Hooking 클래스:
    -   이전: HuggingFace Vision Transformer 모델용으로 설계된 `_hooks.py`의
      `ModelHooking`을 임포트했습니다.
    -   현재: `CATANet` 아키텍처에 특별히 맞춰진 `catanet_hooks.py`의
      `CATANetModelHooking`을 임포트합니다.

2.  데이터 처리:
    -   이전: 데이터 로더가 `(이미지, 레이블)` 튜플을 반환하고, 이를
      `{'pixel_values': ..., 'labels': ...}` 딕셔너리로 재포장한다고 가정했습니다.
    -   현재: `BasicSR` 데이터 로더가 반환하는 `{'lq': ..., 'gt': ...}`
      딕셔너리 형식을 올바르게 처리하고 `lq_tensor`를 추출합니다.

3.  모델 및 데이터 전달:
    -   이전: `forwardPass`에 딕셔너리를, hooking 클래스에 `CATANetModel` 객체를
      전달했습니다.
    -   현재: hooking 클래스에 실제 네트워크인 `model`을, `forwardPass`에
      `lq_tensor`를 직접 전달하여 `CATANetModelHooking` 및 `CATANet` 모델의
      기대치에 부합하도록 수정했습니다.
"""
import torch
from prune.loss_components import KLDiv, manifold_Distillation, patch_based_manifold_Distillation
from queue import PriorityQueue
from torch.nn import functional as F
from tqdm import tqdm
import time
from utils.catanet_hooks import CATANetModelHooking # MODIFIED: Use the new CATANet-specific hook
import numpy as np
import os
import pickle

    
@torch.no_grad()
def Prune(args, prunedProps, lq_tensor, model, base_layer_wise_output, base_logit_output, base_grad_output=None):

    # 이 함수의 핵심 로직(KLDiv, MMD 기반 중요도 점수 계산)은 원본 OPTIN 구현을
    # 따릅니다. 호환성 변경은 주로 모델 호출 및 데이터 처리 방식에 있습니다.
    
    PerNeuronIntermediateMasking = -1*(torch.eye(prunedProps["inter_size"]) - 1) 
    
    globalNeuronRanking = PriorityQueue() 
    averageScaling = []
    
    for layer in range(prunedProps["num_layers"]):
        
        print(f"Layer Sample: {layer} / {prunedProps['num_layers']}")
        if layer==0:
            neuronRanking = PriorityQueue() 
        
        # 참고: 원본 코드는 Vision Transformer 아키텍처와 관련된 "패치"도
        # 프루닝했습니다. CATANet의 경우, FFN 뉴런에만 관심이 있으므로 명확성을
        # 위해 패치 로직은 제거되었습니다.
        TotalNumNeurons = prunedProps["inter_size"]
        
        for neuron in tqdm(range(TotalNumNeurons), desc=f"Analyzing Neurons in Layer {layer}"):
            then = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            maskingProps = {
                "state":"neuron",
                "layer": layer,
                "mask": PerNeuronIntermediateMasking[neuron]
            }
                
            modelObject = CATANetModelHooking(args=args, model=model, maskProps=maskingProps, disable_grad=True) # MODIFIED
            with torch.no_grad():
                # MODIFIED: Pass the lq_tensor directly
                current_logit_output, current_layer_wise_output = modelObject.forwardPass(lq_tensor)
            modelObject.purge_hooks()
            
            MMDLayerResults = 0
            KLErr = 0
            
            # --- 아래의 손실 계산 로직은 OPTIN에서 가져온 것입니다 ---
            # 단일 뉴런 제거에 대한 모델의 민감도를 계산합니다.
            # `manifold_Distillation`은 중간 피처 맵을 비교합니다.
            # `KLDiv`는 최종 출력을 비교합니다. SR의 경우, 이는 출력 이미지를 비교합니다.
            if args.loss_type == "MMD" or args.loss_type == "MMD+KL":
                for idx in range(len(base_layer_wise_output)):
                    if idx > layer or ((layer == prunedProps["num_layers"]-1) and layer == idx):
                        with torch.no_grad():
                            err = manifold_Distillation(args, base_layer_wise_output[idx], current_layer_wise_output[idx])
                            MMDLayerResults += err
            
            if args.loss_type == "KL" or args.loss_type == "MMD+KL":    
                # MODIFIED: Removed temp=args.temp as it's no longer used
                KLErr = KLDiv(base_logit_output, current_logit_output)
            
            MMDResults = MMDLayerResults + KLErr
            
            # 원본 OPTIN 코드의 손실 스케일링 휴리스틱
            if MMDLayerResults < KLErr:
                try:
                    ratio = np.log10(-1*MMDLayerResults.detach().cpu().item()) - np.log10(-1*KLErr.detach().cpu().item())
                    targetRatio = np.log10(prunedProps["lambda"])
                    scaling = 10**int(ratio - targetRatio) 
                    averageScaling.append(scaling)   
                    KLErr *= scaling
                except:
                    scaling = np.mean(averageScaling)
                    if np.isnan(scaling): scaling = 1
                    KLErr *= scaling
            
            MMDResults = MMDLayerResults + KLErr
            # --- OPTIN 손실 로직 끝 ---
                
            globalNeuronRanking.put((MMDResults.detach().cpu(), layer, neuron, "neuron"))
            
    return globalNeuronRanking, None # 사용하지 않는 두 번째 변수에 대해 None 반환
            
def pruneVisionNeurons(model, train_dataset, args, prunedProps):
    
    # 이 함수는 이제 BasicSR 데이터 로더와 CATANet 모델을 올바르게 처리합니다.
    
    storage_path_body = f"./storage/{args.task_name}/{args.dataset}/{args.model_name}/neuron_ranking_body.pkl"
    os.makedirs(os.path.dirname(storage_path_body), exist_ok=True)
    
    prunedProps["lambda"] = args.lambda_contribution
    
    if not os.path.isfile(storage_path_body):
        
        torch.backends.cudnn.benchmark = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # MODIFIED: 단일 배치를 가져와 'lq' 텐서를 추출합니다
        print("분석을 위해 데이터 로더에서 배치를 가져오는 중...")
        batch = next(iter(train_dataset))
        lq_tensor = batch['lq'].to(device, non_blocking=True)
        
        # MODIFIED: 이 함수에 전달된 `model`이 실제 네트워크이므로 직접 사용합니다.
        model.eval()
        
        print("가지치기 전 기준 출력 계산 중...")
        modelObject = CATANetModelHooking(args=args, model=model, disable_grad=True)
        
        # MODIFIED: lq_tensor를 직접 전달합니다
        base_logit_output, base_layer_wise_output = modelObject.forwardPass(lq_tensor)
        modelObject.purge_hooks()

        print("뉴런 중요도 계산 시작...")
        # MODIFIED: lq_tensor와 실제 네트워크를 전달합니다
        globalNeuronRanking, _ = Prune(args, prunedProps, lq_tensor, model, base_layer_wise_output, base_logit_output)

        exportglobalNeuronRanking = []
        while not globalNeuronRanking.empty():
            exportglobalNeuronRanking.append(globalNeuronRanking.get())
        
        with open(storage_path_body, 'wb') as f:
            pickle.dump(exportglobalNeuronRanking, f)
            
    else:
        print(f"{storage_path_body}에서 기존 뉴런 랭킹을 로드하는 중")
        with open(storage_path_body, 'rb') as f:
            exportglobalNeuronRanking = pickle.load(f)
    
    return {
        "final_neuron_ranking": exportglobalNeuronRanking,
    }
