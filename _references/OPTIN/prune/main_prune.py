import torch
from prune.language_pruning import pruneLanguageNeurons
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
    
    
    if args.task_name == "language":
        
        head_mask_results = pruneHead(model, train_dataset, args, prunedProps)
        
        # Only Prune Heads and Intermediate Neurons
        intermediate_neuron_results = pruneLanguageNeurons(model, train_dataset, args, prunedProps)
        
        
        languageProps = {
            "head_results": head_mask_results,
            "intermediate_results": intermediate_neuron_results,
            "mac_details": get_mac_details(args, prunedProps)
            
        }
        
        masks = globalRankingLanguage(args, prunedProps, languageProps)
        
        
        pruningParams = {
        "head_mask":  masks["head_mask"], 
        "neuron_mask": masks["intermediate_mask"],
        }
        
        baselineComplexity, prunedComplexity = calculateComplexity(args, model, train_dataset, prunedProps, pruningParams)
        baselineComplexity = baselineComplexity["MAC"]
        prunedComplexity = prunedComplexity["MAC"]
        
    elif args.task_name == "vision":
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
        
    
    return pruningParams,baselineComplexity,prunedComplexity 





def globalRankingLanguage(args, prunedProps, languageProps):
    head_mask = languageProps["head_results"]["final_head_ranking"]
    head_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in head_mask]
    head_rank = np.array(head_rank)
    
    neuron_mask = languageProps["intermediate_results"]["final_neuron_ranking"]
    neuron_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in neuron_mask]
    neuron_rank = np.array(neuron_rank)
    
    head_mac = languageProps["mac_details"]["head_mac"]
    neuron_mac = languageProps["mac_details"]["neuron_mac"]
    baseline_mac = languageProps["mac_details"]["base_mac"]
    
    capacity_mac = args.mac_constraint * baseline_mac
    
    
    max_importance = 0
    for num_heads in (range(1, prunedProps["num_att_head"]*prunedProps["num_layers"] + 1)):
        current_importance = 0
        
        for i in range(num_heads):
            score, _, _, _ = head_rank[i]
            current_importance += -1*float(score)
        
        count_head_mac = head_mac * (num_heads)
        remaining_mac = capacity_mac - count_head_mac
        
        num_neurons=0
        while remaining_mac >= neuron_mac and num_neurons < len(neuron_rank):
            score, neuron_layer, neuron_index, name = neuron_rank[num_neurons]
            current_importance += -1*float(score)
            num_neurons +=1 
            remaining_mac -= neuron_mac
        
        if current_importance > max_importance:
            max_importance = current_importance
            head_indicies = num_heads
            neuron_indicies = num_neurons
    
    final_head_mask = torch.zeros((prunedProps["num_layers"],prunedProps["num_att_head"]))
    final_neuron_mask = torch.zeros((prunedProps["num_layers"],prunedProps["inter_size"]))
    
    for i in range(head_indicies):
        score, head_layer, head_index, name = head_rank[i]
        final_head_mask[int(head_layer)][int(head_index)] = 1
        
    for i in range(neuron_indicies):
        score, neuron_layer, neuron_index, name = neuron_rank[i]
        final_neuron_mask[int(neuron_layer)][int(neuron_index)] = 1
    
    
    print(final_head_mask.sum(-1),final_neuron_mask.sum(-1))
    
    masks = {
        "head_mask": final_head_mask,
        "intermediate_mask": final_neuron_mask
    }
    
    return masks


def globalRankingVision(args, model, prunedProps, visionProps):
    
    head_mask = visionProps["head_results"]["final_head_ranking"]
    head_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in head_mask]
    head_rank = np.array(head_rank)
    
    neuron_mask = visionProps["intermediate_results"]["final_neuron_ranking"]
    neuron_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in neuron_mask]
    neuron_rank = np.array(neuron_rank)
    
    #* Patch neurons are at idx > inter_size
    
    head_mac = visionProps["mac_details"]["head_mac"]
    neuron_mac = visionProps["mac_details"]["neuron_mac"]
    patch_mac = visionProps["mac_details"]["patch_mac"]
    
    baseline_mac = visionProps["mac_details"]["base_mac"]
    
    capacity_mac = args.mac_constraint * baseline_mac
    
    ammount_to_be_reduced = (1-args.mac_constraint)*baseline_mac
    
    
    if args.beta_config_only: #* Beta Configuration
        controlled_ammount_head_neuron = ammount_to_be_reduced
        controlled_ammount_patches = 0
    else: #* Tau Configuration
        controlled_ammount_head_neuron = ammount_to_be_reduced*np.random.uniform(0.15, 0.30)
        controlled_ammount_patches = ammount_to_be_reduced - controlled_ammount_head_neuron
    
    head_neuron_based_capacity = baseline_mac - controlled_ammount_head_neuron
    
    patch_based_capacity = head_neuron_based_capacity - controlled_ammount_patches
    
    print(baseline_mac, capacity_mac, head_neuron_based_capacity, patch_based_capacity)
    
    max_importance = 0
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
        while remaining_mac > 0 and num_neurons < prunedProps["inter_size"]*prunedProps["num_layers"]: # Subtract Patch Len!
            score, neuron_layer, neuron_index, name = neuron_rank[idx]
            idx += 1
            
            #* Skipping Patches in this search
            if int(neuron_index) >= prunedProps["inter_size"]: 
                continue
            
            current_importance += -1*float(score)
            num_neurons +=1 
            
            remaining_mac -= neuron_mac
            
            neuron_indicies.append(idx-1)
            
        print(max_importance, current_importance, remaining_mac)
        if current_importance > max_importance:
            max_importance = current_importance
            head_indicies = num_heads
            best_neuron_indicies = neuron_indicies
        
    final_head_mask = torch.zeros((prunedProps["num_layers"],prunedProps["num_att_head"]))
    final_neuron_mask = torch.zeros((prunedProps["num_layers"],prunedProps["inter_size"]))
    
    #* Populate Head and Neuron only Masks
    for i in range(head_indicies):
        score, head_layer, head_index, name = head_rank[i]
        final_head_mask[int(head_layer)][int(head_index)] = 1
        
    for i in best_neuron_indicies:
        score, neuron_layer, neuron_index, name = neuron_rank[i]
        
        neuron_layer = int(neuron_layer)
        neuron_index = int(neuron_index)
        final_neuron_mask[neuron_layer][neuron_index] = 1
        
    
    ### We have now achieved our first search (i.e completed Head and Neuron Level Searches)
    final_patch_mask = torch.ones((prunedProps["num_layers"],prunedProps["patch_size"]))

    
    reversed_neuron_rank = neuron_rank[::-1] # We are going in order of removal
    
    remaining_mac = controlled_ammount_patches
    
    
    pruningParams = {
        "head_mask":  final_head_mask, 
        "neuron_mask": final_neuron_mask,
        "patch_mask":final_patch_mask # Assumes all Patches Present !
        }
    

    new_baseline_mac = compute_pruned_mac(args, prunedProps, pruningParams, skipConv=True)
    
    curr_mac = new_baseline_mac

    #* Only Active for Tau Configurations!

    if not args.beta_config_only:
        print("Tau Config", new_baseline_mac,new_baseline_mac-remaining_mac)
        idx = 0
        while curr_mac > new_baseline_mac-remaining_mac:
            score, neuron_layer, neuron_index, name = reversed_neuron_rank[idx]
            idx +=1

            #* Skips non-patch indices -- i.e we only search for patch configuraiton
            if int(neuron_index) < prunedProps["inter_size"] or int(neuron_layer)==0: 
                continue
            
            #* Re-Compute the Current mac information
            final_patch_mask[(int(neuron_layer))][int(neuron_index)-prunedProps["inter_size"]] = 0
            proc_patch_mask = process_patch_mask(final_patch_mask)
            pruningParams["patch_mask"] = proc_patch_mask
            curr_mac = compute_pruned_mac(args, prunedProps, pruningParams, skipConv=True)

    
    if args.beta_config_only:
        assert (final_patch_mask).sum() == final_patch_mask.numel()
        
    final_patch_mask = process_patch_mask(final_patch_mask)
    
    

    masks = {
        "head_mask": final_head_mask,
        "intermediate_mask": final_neuron_mask,
        "patch_mask": final_patch_mask
    }
    

    flop_red = ammount_to_be_reduced/baseline_mac
    
    
    print("-----------------------------------------------")
    print("||")
    print(f"|| Planned Total Flop Reduction is {flop_red}") 
    print(f"|| Actual Total Flop Reduction is {1- (curr_mac/baseline_mac)}") 
    print("||")
    print("-----------------------------------------------")
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
                 
                curr_indices = np.where(patch_mask[idx] == 1)[0] # out of 197
                
                
                mask = torch.zeros((len(prev_sel_ind)))
                
                new_curr_indices = []
                for ind in curr_indices:
                    
                    if ind in prev_sel_ind:
                        new_curr_indices.append(ind)
                        mask[np.where(prev_sel_ind ==ind)[0]] = 1
                newPatchMask.append(mask)
                prev_sel_ind = np.array(new_curr_indices)
                
                
           
                
    return newPatchMask