import sys
import os

# Add the parent directory (ai) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.utils.data
import subprocess
import yaml
from tqdm import tqdm
import numpy as np

# --- 1. 의존성 설치 및 임포트 ---
try:
    from thop import profile
except ImportError:
    print("thop 라이브러리를 설치합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile

from scripts.load_catanet import get_catanet_teacher_model
from basicsr.data import build_dataset
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img
from rebuild_pruned_model import PrunedCATANet

# --- 2. 헬퍼 함수 ---
def calculate_model_complexity(model, input_tensor):
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2
    return flops, params

def calculate_model_quality(model, dataloader, device):
    model.eval()
    model.to(device)
    total_psnr = total_ssim = 0
    pbar = tqdm(dataloader, desc=f'Evaluating Model', unit='image')
    for batch in pbar:
        lq, gt = batch['lq'].to(device), batch['gt'].to(device)
        with torch.no_grad():
            output = model(lq)
        output_img, gt_img = tensor2img(output), tensor2img(gt)
        total_psnr += calculate_psnr(output_img, gt_img, crop_border=2, test_y_channel=True)
        total_ssim += calculate_ssim(output_img, gt_img, crop_border=2, test_y_channel=True)
        pbar.set_postfix({'PSNR': f"{total_psnr / len(pbar):.4f}", 'SSIM': f"{total_ssim / len(pbar):.4f}"})
    return total_psnr / len(dataloader), total_ssim / len(dataloader)

# --- 3. 메인 스크립트 ---
def main():
    # 스크립트의 기본 디렉토리를 ai/analysis로 설정하고, 모든 경로는 ai/ 디렉토리 기준으로 재설정합니다.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working directory is relative to: {base_dir}")
    print(f"Using device: {device}\n")

    # --- 경로 설정 (ai/ 디렉토리 기준) ---
    original_model_path = os.path.join(base_dir, 'weights/CATANet-L_x2.pth')
    rebuilt_model_path = os.path.join(base_dir, 'weights/catanet_rebuilt.pth')
    masks_path = os.path.join(base_dir, 'weights/catanet_pruning_masks.pth')
    config_path = os.path.join(base_dir, 'config_catanet.yml')
    upscale = 2
    
    if not os.path.exists(rebuilt_model_path) or not os.path.exists(masks_path):
        print(f"[오류] '{rebuilt_model_path}' 또는 '{masks_path}' 파일을 찾을 수 없습니다.")
        print("먼저 'rebuild_pruned_model.py' 스크립트를 실행하여 최종 압축 모델을 생성해야 합니다.")
        return

    # --- 모델 로드 및 복잡도 계산 ---
    complexity_input_tensor = torch.randn(1, 3, 64, 64)
    print("복잡도(FLOPs/Params)는 64x64 크기의 입력 텐서를 기준으로 계산합니다.\n")

    print("1. 원본 모델 로딩 및 분석...")
    original_model = get_catanet_teacher_model(weights_path=original_model_path, upscale=upscale)
    original_flops, original_params = calculate_model_complexity(original_model, complexity_input_tensor)

    print("2. 재구성된 최종 모델 로딩 및 분석...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    masks = torch.load(masks_path, map_location='cpu')
    new_head_counts = masks['head_mask'].sum(dim=-1).int().tolist()
    new_mlp_dims = masks['neuron_mask'].sum(dim=-1).int().tolist()
    
    rebuilt_model = PrunedCATANet(
        upscale=config.get('scale', 2),
        mlp_dims=new_mlp_dims,
        head_counts=new_head_counts,
        args=config
    )
    rebuilt_data = torch.load(rebuilt_model_path, map_location='cpu')
    rebuilt_model.load_state_dict(rebuilt_data['params'])
    rebuilt_flops, rebuilt_params = calculate_model_complexity(rebuilt_model, complexity_input_tensor)

    # --- 품질 평가 (PSNR, SSIM) ---
    print("\n3. 품질(PSNR/SSIM)은 Set5 데이터셋을 기준으로 평가합니다.")
    dataset_opt = {'name': 'Set5', 'type': 'PairedImageDataset', 
                   'dataroot_gt': os.path.join(base_dir, 'datasets/Set5/HR'),
                   'dataroot_lq': os.path.join(base_dir, 'datasets/Set5/LR_bicubic/X2'),
                   'filename_tmpl': '{}x2',
                   'io_backend': {'type': 'disk'}, 'scale': upscale, 'phase': 'val'}
    
    try:
        test_set = build_dataset(dataset_opt)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    except Exception as e:
        print(f"\n[오류] 테스트 데이터셋을 로드할 수 없습니다: {e}")
        test_loader = None

    if test_loader:
        original_psnr, original_ssim = calculate_model_quality(original_model, test_loader, device)
        rebuilt_psnr, rebuilt_ssim = calculate_model_quality(rebuilt_model, test_loader, device)
    
    # --- 결과 종합 ---
    flops_reduction = (original_flops - rebuilt_flops) / original_flops * 100
    params_reduction = (original_params - rebuilt_params) / original_params * 100
    psnr_change = rebuilt_psnr - original_psnr if test_loader else None
    ssim_change = rebuilt_ssim - original_ssim if test_loader else None
    
    print("\n--- 최종 성능 종합 비교: 원본 vs 재구성된 모델 ---")
    header = f"{ 'Metric':<20} | { 'Original Model':<20} | { 'Rebuilt Model':<20} | { 'Change':<15}"
    print(header)
    print("-" * (len(header) + 5))
    
    if test_loader:
        print(f"{ 'PSNR (dB)':<20} | {original_psnr:<20.4f} | {rebuilt_psnr:<20.4f} | {psnr_change:+.4f}")
        print(f"{ 'SSIM':<20} | {original_ssim:<20.4f} | {rebuilt_ssim:<20.4f} | {ssim_change:+.4f}")
        
    print(f"{ 'GFLOPs':<20} | {original_flops / 1e9:<20.4f} | {rebuilt_flops / 1e9:<20.4f} | {f'-{flops_reduction:.2f}%'}")
    print(f"{ 'Total Params (M)':<20} | {original_params / 1e6:<20.4f} | {rebuilt_params / 1e6:<20.4f} | {f'-{params_reduction:.2f}%'}")
    print("-" * (len(header) + 5))

if __name__ == '__main__':
    main()