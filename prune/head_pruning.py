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
    # 핵심 로직은 OPTIN 것을 유지하되, CATANet 모델에 맞게 refactoring 되었습니다.
    
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
            # 따라서 마스킹만 진행될 뿐 가중치를 0으로 만드는 최종 프루닝 적용은 별도로 구현해야 합니다.
            # 원본 `run_catanet_pruning.py` 역시 이 단계를 생략하고 뉴런 프루닝만 적용했습니다.
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
                KLErr = KLDiv(base_logit_output, current_logit_output)
            
            MMDResults = MMDLayerResults + KLErr
            # --- OPTIN 손실 로직 끝 ---
                
            globalHeadRanking.put((MMDResults.detach().cpu(), layer, head, "head"))

    return globalHeadRanking, None # 사용하지 않는 두 번째 변수에 대해 None 반환

            
def pruneHead(model, train_dataset, args, prunedProps):

    storage_path_body = f"./storage/{args.task_name}/{args.dataset}/{args.model_name}/head_ranking_body.pkl"
    os.makedirs(os.path.dirname(storage_path_body), exist_ok=True)
    
    prunedProps["lambda"] = args.lambda_contribution
    
    if not os.path.isfile(storage_path_body):
    
        FullHeadMasking = torch.ones((prunedProps["num_layers"],prunedProps["num_att_head"]))
        torch.backends.cudnn.benchmark = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("분석을 위해 데이터 로더에서 배치를 가져오는 중...")
        batch = next(iter(train_dataset))
        lq_tensor = batch['lq'].to(device, non_blocking=True)
            
        model.eval()
        
        print("가지치기 전 기준 출력 계산 중...")
        # 참고: 기준선 계산을 위해 maskProps 없이 전달합니다.
        modelObject = CATANetModelHooking(args=args, model=model, maskProps=None, disable_grad=True)
        
        base_logit_output, base_layer_wise_output = modelObject.forwardPass(lq_tensor)
        modelObject.purge_hooks()
        
       
        print("헤드 중요도 계산 시작...")
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