import torch
from copy import deepcopy
from tqdm import tqdm
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img

# 메모리 내에서 가지치기를 적용하는 함수
def _apply_pruning_in_memory(model, pruning_params):
    """
    CATANet에 대한 뉴런 및 헤드 가지치기 모두를 처리합니다. 
    (그렇다고 삭제하는건 아니고 마스킹만 함)
    """
    pruned_state_dict = deepcopy(model.state_dict())
    
    # --- 1. 뉴런 가지치기 적용 ---
    neuron_mask = pruning_params.get('neuron_mask')
    if neuron_mask is not None:
        print("뉴런 가지치기 적용 중...")
        for i in range(model.block_num):
            # 마스크에서 0인 뉴런(가지치기된 뉴런)을 찾습니다.
            pruned_neurons = (neuron_mask[i] == 0).nonzero(as_tuple=True)[0]
            if len(pruned_neurons) == 0: continue

            # FC1 레이어의 가중치와 편향에 가지치기를 적용합니다.
            fc1_weight_name = f'blocks.{i}.0.mlp.fn.fc1.weight'
            fc1_bias_name = f'blocks.{i}.0.mlp.fn.fc1.bias'
            if fc1_weight_name in pruned_state_dict:
                pruned_state_dict[fc1_weight_name][pruned_neurons, :] = 0
                pruned_state_dict[fc1_bias_name][pruned_neurons] = 0

            # FC2 레이어의 가중치에 가지치기를 적용합니다.
            fc2_weight_name = f'blocks.{i}.0.mlp.fn.fc2.weight'
            if fc2_weight_name in pruned_state_dict:
                pruned_state_dict[fc2_weight_name][:, pruned_neurons] = 0
    
    # --- 2. 헤드 가지치기 적용 ---
    head_mask = pruning_params.get('head_mask')
    if head_mask is not None:
        print("헤드 가지치기 적용 중...")
        q_head_dim = model.qk_dim // model.heads
        v_head_dim = model.dim // model.heads

        for i in range(model.block_num):
            # 마스크에서 0인 헤드(가지치기된 헤드)를 찾습니다.
            pruned_heads = (head_mask[i] == 0).nonzero(as_tuple=True)[0]
            for head_idx in pruned_heads:
                start_q = head_idx * q_head_dim
                end_q = start_q + q_head_dim
                start_v = head_idx * v_head_dim
                end_v = start_v + v_head_dim

                # CATANet 블록 내의 두 Attention 모듈 경로를 정의합니다.
                attention_modules_paths = [
                    f'blocks.{i}.0.iasa_attn', # TAB 내의 IASA
                    f'blocks.{i}.1.layer.0.fn'  # LRSA 내의 Attention
                ]

                for attn_path in attention_modules_paths:
                    # Q 및 K 가중치 (출력 차원)에 가지치기를 적용합니다.
                    for layer in ['to_q', 'to_k']:
                        key = f'{attn_path}.{layer}.weight'
                        if key in pruned_state_dict:
                            pruned_state_dict[key][start_q:end_q, :] = 0
                    
                    # V 가중치 (출력 차원)에 가지치기를 적용합니다.
                    key = f'{attn_path}.to_v.weight'
                    if key in pruned_state_dict:
                        pruned_state_dict[key][start_v:end_v, :] = 0
                    
                    # 투영 레이어 가중치 (입력 차원)에 가지치기를 적용합니다.
                    key = f'{attn_path}.proj.weight'
                    if key in pruned_state_dict:
                        pruned_state_dict[key][:, start_v:end_v] = 0
            
    return pruned_state_dict

# 주어진 데이터 로더에서 모델의 성능을 평가하는 헬퍼 함수
def _evaluate_performance(model, dataloader, args):
    """
    주어진 데이터 로더에서 모델을 평가하고 평균 PSNR 및 SSIM을 계산합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 모델을 평가 모드로 설정합니다.

    total_psnr = 0
    total_ssim = 0
    num_images = 0
    
    # 평가 옵션을 가져옵니다.
    val_opts = args.datasets.get('val', {})
    metric_opts = val_opts.get('metrics', {
        'psnr': {'crop_border': 4, 'test_y_channel': True},
        'ssim': {'crop_border': 4, 'test_y_channel': True}
    })
    psnr_opt = metric_opts.get('psnr', {'crop_border': 4, 'test_y_channel': True})
    ssim_opt = metric_opts.get('ssim', {'crop_border': 4, 'test_y_channel': True})

    pbar = tqdm(total=len(dataloader), unit='image', desc='평가 중')
    
    for val_data in dataloader:
        lq_tensor = val_data['lq'].to(device) # 저품질 이미지
        gt_tensor = val_data['gt'].to(device) # 원본 이미지

        with torch.no_grad(): # 그래디언트 계산을 비활성화합니다.
            output_tensor = model(lq_tensor) # 모델 예측

        # 텐서를 이미지로 변환합니다.
        sr_img = tensor2img(output_tensor.detach().cpu())
        gt_img = tensor2img(gt_tensor.detach().cpu())
        
        # PSNR 및 SSIM을 계산하여 합산합니다.
        total_psnr += calculate_psnr(sr_img, gt_img, **psnr_opt)
        total_ssim += calculate_ssim(sr_img, gt_img, **ssim_opt)
        
        num_images += 1
        pbar.update(1)

    pbar.close()

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    
    return {'psnr': avg_psnr, 'ssim': avg_ssim}

# 기준 모델과 가지치기된 모델의 성능을 평가하는 함수
def evalModel(args, model, train_dataset, val_dataset, pruningParams, prunedProps):
    """
    기준 모델과 가지치기된 모델의 성능을 평가합니다.
    """
    print("--- 기준 모델 평가 중 ---")
    baseline_model = model
    baseline_performance = _evaluate_performance(baseline_model, val_dataset, args)
    print(f"기준 모델 성능: PSNR={baseline_performance['psnr']:.4f}, SSIM={baseline_performance['ssim']:.4f}")

    print("\n--- 가지치기된 모델 평가 중 ---")
    pruned_model = deepcopy(baseline_model)
    
    # 메모리 내에서 가지치기를 적용합니다.
    pruned_state_dict = _apply_pruning_in_memory(pruned_model, pruningParams)
    
    # 가지치기된 state_dict를 모델에 로드합니다.
    pruned_model.load_state_dict(pruned_state_dict)
    
    final_performance = _evaluate_performance(pruned_model, val_dataset, args)
    print(f"가지치기된 모델 성능: PSNR={final_performance['psnr']:.4f}, SSIM={final_performance['ssim']:.4f}")
    
    # 가지치기된 모델의 0이 아닌 매개변수를 계산합니다.
    total_params = sum(p.numel() for p in pruned_model.parameters())
    non_zero_params = sum(torch.count_nonzero(p.data).item() for p in pruned_model.parameters())
    
    print(f"가지치기된 모델 매개변수: 총 {total_params / 1e6:.2f}M, 0이 아닌 매개변수 {non_zero_params / 1e6:.2f}M")
    
    return baseline_performance, final_performance
