import torch
# MODIFIED: Removed unused language_pruning import
# from prune.language_pruning import pruneLanguageNeurons
from prune.head_pruning import pruneHead
from prune.vision_pruning import pruneVisionNeurons
import numpy as np
from utils.mac_complexity import get_mac_details, compute_pruned_mac
from utils.utility import calculateComplexity

global Head_mean
global Neuron_mean

def pruneModel(args, model, train_dataset, model_config):
    
    prunedProps = {
        "num_att_head": model_config["num_attention_heads"],
       "inter_size": model_config["intermediate_size"],
       "hidden_size": model_config["hidden_size"],
       "num_layers":model_config["num_hidden_layers"],
       "patch_size": args.seq_len+1 
    }
    
    if args.task_name == "vision":
        head_mask_results = pruneHead(model, train_dataset, args, prunedProps)
        intermediate_neuron_results = pruneVisionNeurons(model, train_dataset, args, prunedProps)
        
        visionProps = {
            "head_results": head_mask_results,
            "intermediate_results": intermediate_neuron_results,
            "mac_details": get_mac_details(args, prunedProps)
        }
        
        masks, prunedComplexity = globalRankingVision(args, model, prunedProps, visionProps)
        
        pruningParams = {
            "head_mask":  masks["head_mask"],
            "neuron_mask": masks["intermediate_mask"],
            "patch_mask": masks["patch_mask"]
        }
        
        baselineComplexity = visionProps["mac_details"]["base_mac"]
        
    else:
        raise ValueError(f"Task name '{args.task_name}' is not supported for pruning.")

    return pruningParams,baselineComplexity,prunedComplexity 

def globalRankingVision(args, model, prunedProps, visionProps):
    
    head_mask = visionProps["head_results"]["final_head_ranking"]
    head_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in head_mask]
    head_rank = np.array(head_rank)
    
    neuron_mask = visionProps["intermediate_results"]["final_neuron_ranking"]
    neuron_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in neuron_mask]
    neuron_rank = np.array(neuron_rank)
    
    head_mac = visionProps["mac_details"]["head_mac"]
    neuron_mac = visionProps["mac_details"]["neuron_mac"]
    patch_mac = visionProps["mac_details"]["patch_mac"]
    baseline_mac = visionProps["mac_details"]["base_mac"]
    capacity_mac = args.mac_constraint * baseline_mac
    ammount_to_be_reduced = (1-args.mac_constraint)*baseline_mac
    
    if args.beta_config_only:
        controlled_ammount_head_neuron = ammount_to_be_reduced
        controlled_ammount_patches = 0
    else:
        controlled_ammount_head_neuron = ammount_to_be_reduced*np.random.uniform(0.15, 0.30)
        controlled_ammount_patches = ammount_to_be_reduced - controlled_ammount_head_neuron
    
    head_neuron_based_capacity = baseline_mac - controlled_ammount_head_neuron
    
    head_rank = head_rank[::-1]
    neuron_rank = neuron_rank[::-1]
    
    max_importance = -float('inf')
    best_neuron_indicies = None
    
    for num_heads in (range(1, prunedProps["num_att_head"]*prunedProps["num_layers"] + 1)):
        current_importance = 0
        
        for i in range(num_heads):
            score, _, _, _ = head_rank[i]
            current_importance += -1*float(score)
        
        count_head_mac = head_mac * (num_heads)
        remaining_mac = head_neuron_based_capacity - count_head_mac
        
        idx = 0
        num_neurons=0
        neuron_indicies =[]
        while remaining_mac > 0 and num_neurons < prunedProps["inter_size"]*prunedProps["num_layers"]:
            score, neuron_layer, neuron_index, name = neuron_rank[idx]
            idx += 1
            if int(neuron_index) >= prunedProps["inter_size"]:
                continue
            current_importance += -1*float(score)
            num_neurons +=1 
            remaining_mac -= neuron_mac
            neuron_indicies.append(idx-1)

        if current_importance > max_importance:
            max_importance = current_importance
            head_indicies = num_heads
            best_neuron_indicies = neuron_indicies
        
    final_head_mask = torch.zeros((prunedProps["num_layers"],prunedProps["num_att_head"]))
    final_neuron_mask = torch.zeros((prunedProps["num_layers"],prunedProps["inter_size"]))
    
    for i in range(head_indicies):
        score, head_layer, head_index, name = head_rank[i]
        final_head_mask[int(head_layer)][int(head_index)] = 1
        
    if best_neuron_indicies:
        for i in best_neuron_indicies:
            score, neuron_layer, neuron_index, name = neuron_rank[i]
            neuron_layer = int(neuron_layer)
            neuron_index = int(neuron_index)
            final_neuron_mask[neuron_layer][neuron_index] = 1
            
    # --- [핵심 수정] ---
    # 각 레이어에 최소 1개의 헤드와 뉴런이 유지되도록 강제합니다.
    # 이를 통해 0개의 헤드/뉴런을 가진 유효하지 않은 아키텍처가 생성되는 것을 방지합니다.
    
    # 1. 헤드 강제 유지
    layer_head_sums = final_head_mask.sum(dim=1)
    for i in range(prunedProps["num_layers"]):
        if layer_head_sums[i] == 0:
            # 이 레이어에서 가장 중요한 헤드를 찾습니다.
            # head_rank는 [score, layer, head_idx, name] 형태입니다.
            most_important_head_for_layer = None
            for _, h_layer, h_idx, _ in head_rank:
                if int(h_layer) == i:
                    most_important_head_for_layer = int(h_idx)
                    break
            if most_important_head_for_layer is not None:
                print(f"[CONSTRAINT] Layer {i} has 0 heads. Forcing to keep the most important head: {most_important_head_for_layer}")
                final_head_mask[i][most_important_head_for_layer] = 1

    # 2. 뉴런 강제 유지 (일반적으로는 뉴런이 모두 제거되는 경우가 적지만, 안전장치로 추가)
    layer_neuron_sums = final_neuron_mask.sum(dim=1)
    for i in range(prunedProps["num_layers"]):
        if layer_neuron_sums[i] == 0:
            most_important_neuron_for_layer = None
            for _, n_layer, n_idx, _ in neuron_rank:
                if int(n_layer) == i:
                    most_important_neuron_for_layer = int(n_idx)
                    break
            if most_important_neuron_for_layer is not None:
                print(f"[CONSTRAINT] Layer {i} has 0 neurons. Forcing to keep the most important neuron: {most_important_neuron_for_layer}")
                final_neuron_mask[i][most_important_neuron_for_layer] = 1

    print(f"!!! DEBUG: Final masks created. Heads kept: {final_head_mask.sum().item()}. Neurons kept: {final_neuron_mask.sum().item()}.")
    
    final_patch_mask = torch.ones((prunedProps["num_layers"],prunedProps["patch_size"]))
    
    pruningParams = {
        "head_mask": final_head_mask, 
        "neuron_mask": final_neuron_mask,
        "patch_mask":final_patch_mask
    }
    
    # Recalculate pruned MACs with the constrained masks
    curr_mac = compute_pruned_mac(args, prunedProps, pruningParams, skipConv=False)
    
    flop_red = (baseline_mac - curr_mac) / baseline_mac
    
    print("-----------------------------------------------")
    print("||")
    print(f"|| Planned Total Flop Reduction is {1 - args.mac_constraint:.2f}") 
    print(f"|| Actual Total Flop Reduction is {flop_red:.4f}") 
    print("||")
    print("-----------------------------------------------")
    
    masks = {
        "head_mask": final_head_mask,
        "intermediate_mask": final_neuron_mask,
        "patch_mask": process_patch_mask(final_patch_mask)
    }
    
    return masks, curr_mac

def process_patch_mask(patch_mask):
    newPatchMask = []
    prev_sel_ind = None
    for idx in range(len(patch_mask)):
        if idx == 0:
            newPatchMask.append(patch_mask[idx])
            prev_sel_ind = np.where(patch_mask[idx] == 1)[0]
        else:
            if patch_mask[idx].sum() >= newPatchMask[-1].sum() or patch_mask[idx].sum()/len(patch_mask[idx]) <= 0.1:
                newPatchMask.append(torch.ones(int(newPatchMask[-1].sum().item())))
            else:
                curr_indices = np.where(patch_mask[idx] == 1)[0]
                mask = torch.zeros((len(prev_sel_ind)))
                new_curr_indices = []
                for ind in curr_indices:
                    if ind in prev_sel_ind:
                        new_curr_indices.append(ind)
                        mask[np.where(prev_sel_ind == ind)[0]] = 1
                newPatchMask.append(mask)
                prev_sel_ind = np.array(new_curr_indices)
    return newPatchMask