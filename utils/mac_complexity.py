"""
[리팩토링 노트]
이 스크립트는 모델의 MACs(Multiply-Accumulate Operations)를 계산하여
가지치기(pruning) 중요도 결정에 활용됩니다. `CATANet` 모델에 맞게 수정되었습니다.

[원본과의 주요 차이점]
1.  초기 컨볼루션 MAC 계산:
    -   이전: `query_conv_mac` 함수는 Vision Transformer(ViT) 모델에 특화된
      초기 패치 임베딩 컨볼루션의 MAC을 계산했습니다.
    -   현재: `catanet_first_conv_mac` 함수를 새로 추가하여, `CATANet`의
      `first_conv` 레이어(3x3 컨볼루션)에 대한 MAC을 올바르게 계산하도록 수정했습니다.
      이는 `compute_base_mac` 및 `compute_pruned_mac`에서 사용됩니다.

2.  `query_conv_mac` 함수 제거:
    -   `CATANet`과 호환되지 않는 `query_conv_mac` 함수는 더 이상 사용되지 않으므로 삭제되었습니다.

3.  `patch_mask` 로직:
    -   `patch_mask` 관련 로직은 OPTIN에서 ViT의 패치 프루닝을 위해 사용되었던 것으로,
      `CATANet`의 아키텍처와는 직접적인 관련이 적습니다. 현재는 구조적 무결성을
      유지하기 위해 남아있지만, `CATANet`에서는 주로 FFN 뉴런 프루닝에 중점을 둡니다.
"""
import torch

def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac


def mac_per_neuron(seq_len, hidden_size):
    return 2 * seq_len * hidden_size


def compute_mac(
    head_mask,
    neuron_mask,
    patch_mask,
    seq_len,
    hidden_size,
    attention_head_size
):
    
    mac = 0.0
    num_layers = len(head_mask)
    for layer in range(num_layers):
        
        current_num_heads = head_mask[layer]
        current_num_neurons = neuron_mask[layer]
        
        
        current_num_patch = patch_mask[layer]
        attention_mac = current_num_heads * mac_per_head(current_num_patch, hidden_size, attention_head_size)
        
        if layer == num_layers -1:
            current_num_patch = patch_mask[num_layers-1]
            ffn_mac = current_num_neurons * mac_per_neuron(current_num_patch, hidden_size)
        else:
            ffn_mac = current_num_neurons * mac_per_neuron(current_num_patch, hidden_size)
         
        mac += attention_mac + ffn_mac
    return mac

def catanet_first_conv_mac(prunedProps, lq_size):
    """
    Calculates the MACs for the first 3x3 convolution in CATANet.
    MODIFIED: Now accepts lq_size dynamically.
    """
    in_channels = 3
    out_channels = prunedProps["hidden_size"] # In CATANet, this is `dim`
    kernel_h, kernel_w = 3, 3
    # With padding=1, stride=1, output size is the same as input size
    output_h, output_w = lq_size, lq_size
    
    mac_count = in_channels * out_channels * kernel_h * kernel_w * output_h * output_w
    return mac_count

def compute_base_mac(args, prunedProps, skipConv):
    
    head_mask = [prunedProps["num_att_head"]] * prunedProps["num_layers"]
    neuron_mask = [prunedProps["inter_size"]] * prunedProps["num_layers"]
    seq_length = prunedProps["patch_size"] - 1
    patch_mask = [seq_length] * prunedProps["num_layers"]
    
    
    attention_head_size = int(prunedProps["hidden_size"] / prunedProps["num_att_head"])
    
    original_mac = compute_mac(
        head_mask,
        neuron_mask,
        patch_mask,
        seq_length,
        prunedProps["hidden_size"],
        attention_head_size
    )
    if not skipConv and args.task_name == "vision":
        # MODIFIED: Dynamically calculate lq_size and pass it
        gt_size = args.datasets['train']['gt_size']
        scale = args.scale
        lq_size = gt_size // scale
        original_mac += catanet_first_conv_mac(prunedProps, lq_size)
    return original_mac


def compute_pruned_mac(args, prunedProps, pruningParams, skipConv):
    
    head_mask = pruningParams["head_mask"].sum(-1)
    neuron_mask = pruningParams["neuron_mask"].sum(-1)
    seq_length = prunedProps["patch_size"] - 1
    
    
    if args.task_name == "vision":
        patch_mask = [seq_length] * prunedProps["num_layers"]
    else:
        patch_mask = [seq_length] * prunedProps["num_layers"]
    
    
    attention_head_size = int(prunedProps["hidden_size"] / prunedProps["num_att_head"])
    
    pruned_mac = compute_mac(
        head_mask,
        neuron_mask,
        patch_mask,
        seq_length,
        prunedProps["hidden_size"],
        attention_head_size
    )
    
    if not skipConv and args.task_name == "vision":
        # MODIFIED: Dynamically calculate lq_size and pass it
        gt_size = args.datasets['train']['gt_size']
        scale = args.scale
        lq_size = gt_size // scale
        pruned_mac += catanet_first_conv_mac(prunedProps, lq_size)
    return pruned_mac.item()



def compute_patch_mac(args, prunedProps, mac_details):
    """Computes the effect of patch removal by layer"""
    
    layerwise_mac = mac_details["head_mac"] + mac_details["neuron_mac"]
    
    reduced_head_mac = mac_per_head(prunedProps["patch_size"] - 1 -1, 
                                 prunedProps["hidden_size"], 
                                 int(prunedProps["hidden_size"] / prunedProps["num_att_head"]))
    reduced_neuron_mac = mac_per_neuron(prunedProps["patch_size"] - 1 - 1, prunedProps["hidden_size"])
    
    reduced_layerwise_mac = reduced_head_mac + reduced_neuron_mac
    
    total_mac_reduction_per_layer = layerwise_mac - reduced_layerwise_mac
    
    patch_mac = []
    for layer in range(1, prunedProps["num_layers"]):
        patch_mac.append((prunedProps["num_layers"]-layer)*total_mac_reduction_per_layer)
        
    final_layer_reduction = mac_details["neuron_mac"] - reduced_neuron_mac
    patch_mac.append(final_layer_reduction)
    
    return patch_mac
    

def get_mac_details(args, prunedProps):
    
    
    mac_details = {
        "base_mac": compute_base_mac(args, prunedProps, skipConv=False),
        "head_mac": mac_per_head(prunedProps["patch_size"] - 1, 
                                 prunedProps["hidden_size"], 
                                 int(prunedProps["hidden_size"] / prunedProps["num_att_head"])),
        "neuron_mac": mac_per_neuron(prunedProps["patch_size"] - 1, prunedProps["hidden_size"]),
    }
    
    if args.task_name == "vision":
        patch_mac = compute_patch_mac(args, prunedProps, mac_details)
        mac_details["patch_mac"] = patch_mac
        
    return mac_details
