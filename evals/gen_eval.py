import torch
from copy import deepcopy
from tqdm import tqdm
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img

def _apply_pruning_in_memory(model, pruning_params):
    """
    Applies pruning by zeroing out weights in the model's state_dict.
    MODIFIED: Now handles both neuron and head pruning for CATANet.
    """
    pruned_state_dict = deepcopy(model.state_dict())
    
    # --- 1. Apply Neuron Pruning ---
    neuron_mask = pruning_params.get('neuron_mask')
    if neuron_mask is not None:
        print("Applying neuron pruning...")
        for i in range(model.block_num):
            pruned_neurons = (neuron_mask[i] == 0).nonzero(as_tuple=True)[0]
            if len(pruned_neurons) == 0: continue

            fc1_weight_name = f'blocks.{i}.0.mlp.fn.fc1.weight'
            fc1_bias_name = f'blocks.{i}.0.mlp.fn.fc1.bias'
            if fc1_weight_name in pruned_state_dict:
                pruned_state_dict[fc1_weight_name][pruned_neurons, :] = 0
                pruned_state_dict[fc1_bias_name][pruned_neurons] = 0

            fc2_weight_name = f'blocks.{i}.0.mlp.fn.fc2.weight'
            if fc2_weight_name in pruned_state_dict:
                pruned_state_dict[fc2_weight_name][:, pruned_neurons] = 0
    
    # --- 2. Apply Head Pruning ---
    head_mask = pruning_params.get('head_mask')
    if head_mask is not None:
        print("Applying head pruning...")
        q_head_dim = model.qk_dim // model.heads
        v_head_dim = model.dim // model.heads

        for i in range(model.block_num):
            pruned_heads = (head_mask[i] == 0).nonzero(as_tuple=True)[0]
            for head_idx in pruned_heads:
                start_q = head_idx * q_head_dim
                end_q = start_q + q_head_dim
                start_v = head_idx * v_head_dim
                end_v = start_v + v_head_dim

                # Define paths for both Attention modules in CATANet's blocks
                attention_modules_paths = [
                    f'blocks.{i}.0.iasa_attn', # IASA in TAB
                    f'blocks.{i}.1.layer.0.fn'  # Attention in LRSA
                ]

                for attn_path in attention_modules_paths:
                    # Prune Q and K weights (output dimension)
                    for layer in ['to_q', 'to_k']:
                        key = f'{attn_path}.{layer}.weight'
                        if key in pruned_state_dict:
                            pruned_state_dict[key][start_q:end_q, :] = 0
                    
                    # Prune V weights (output dimension)
                    key = f'{attn_path}.to_v.weight'
                    if key in pruned_state_dict:
                        pruned_state_dict[key][start_v:end_v, :] = 0
                    
                    # Prune projection layer weights (input dimension)
                    key = f'{attn_path}.proj.weight'
                    if key in pruned_state_dict:
                        pruned_state_dict[key][:, start_v:end_v] = 0
            
    return pruned_state_dict

def _evaluate_performance(model, dataloader, args):
    """
    Helper function to evaluate a model on a given dataloader.
    Calculates average PSNR and SSIM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_psnr = 0
    total_ssim = 0
    num_images = 0
    
    val_opts = args.datasets.get('val', {})
    metric_opts = val_opts.get('metrics', {
        'psnr': {'crop_border': 4, 'test_y_channel': True},
        'ssim': {'crop_border': 4, 'test_y_channel': True}
    })
    psnr_opt = metric_opts.get('psnr', {'crop_border': 4, 'test_y_channel': True})
    ssim_opt = metric_opts.get('ssim', {'crop_border': 4, 'test_y_channel': True})

    pbar = tqdm(total=len(dataloader), unit='image', desc='Evaluating')
    
    for val_data in dataloader:
        lq_tensor = val_data['lq'].to(device)
        gt_tensor = val_data['gt'].to(device)

        with torch.no_grad():
            output_tensor = model(lq_tensor)

        sr_img = tensor2img(output_tensor.detach().cpu())
        gt_img = tensor2img(gt_tensor.detach().cpu())
        
        total_psnr += calculate_psnr(sr_img, gt_img, **psnr_opt)
        total_ssim += calculate_ssim(sr_img, gt_img, **ssim_opt)
        
        num_images += 1
        pbar.update(1)

    pbar.close()

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    
    return {'psnr': avg_psnr, 'ssim': avg_ssim}

def evalModel(args, model, train_dataset, val_dataset, pruningParams, prunedProps):
    """
    Evaluates the performance of the baseline and pruned models.
    """
    print("--- Evaluating Baseline Model ---")
    baseline_model = model
    baseline_performance = _evaluate_performance(baseline_model, val_dataset, args)
    print(f"Baseline Performance: PSNR={baseline_performance['psnr']:.4f}, SSIM={baseline_performance['ssim']:.4f}")

    print("\n--- Evaluating Pruned Model ---")
    pruned_model = deepcopy(baseline_model)
    
    pruned_state_dict = _apply_pruning_in_memory(pruned_model, pruningParams)
    
    pruned_model.load_state_dict(pruned_state_dict)
    
    final_performance = _evaluate_performance(pruned_model, val_dataset, args)
    print(f"Pruned Performance: PSNR={final_performance['psnr']:.4f}, SSIM={final_performance['ssim']:.4f}")
    
    # Calculate non-zero parameters for the pruned model
    total_params = sum(p.numel() for p in pruned_model.parameters())
    non_zero_params = sum(torch.count_nonzero(p.data).item() for p in pruned_model.parameters())
    
    print(f"Pruned Model Parameters: Total {total_params / 1e6:.2f}M, Non-zero {non_zero_params / 1e6:.2f}M")
    
    return baseline_performance, final_performance