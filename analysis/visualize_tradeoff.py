import sys
import os
import torch
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 경로 설정 및 모듈 임포트 ---
# 이 스크립트가 ai/analysis/ 폴더에 있으므로, 상위 폴더(ai)를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from thop import profile
except ImportError:
    # thop이 없을 경우를 대비하지만, calculate_performance.py에서 이미 설치했을 가능성이 높음
    import subprocess
    print("thop 라이브러리를 설치합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile

from scripts.load_catanet import get_catanet_teacher_model
from basicsr.data import build_dataset
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img
from rebuild_pruned_model import PrunedCATANet

# --- 헬퍼 함수 ---

def get_model_metrics(model_name, model_path, model_type, args):
    """단일 모델에 대한 모든 메트릭(PSNR, SSIM, GFLOPs, Params)을 계산합니다."""
    print(f"\n--- '{model_name}' 모델 분석 중 ---")
    
    # 1. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upscale = args['config_data']['scale']
    
    if model_type == 'original':
        model = get_catanet_teacher_model(weights_path=model_path, upscale=upscale)
    elif model_type == 'rebuilt':
        masks = torch.load(args['masks_path'], map_location='cpu')
        new_head_counts = masks['head_mask'].sum(dim=-1).int().tolist()
        new_mlp_dims = masks['neuron_mask'].sum(dim=-1).int().tolist()
        model = PrunedCATANet(
            upscale=upscale,
            mlp_dims=new_mlp_dims,
            head_counts=new_head_counts,
            args=args['config_data']
        )
        model_data = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_data.get('params', model_data))
    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")

    model.eval()
    model.to(device)

    # 2. 복잡도 계산
    input_tensor = torch.randn(1, 3, 64, 64).to(device)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    gflops = (macs * 2) / 1e9
    m_params = params / 1e6

    # 3. 품질 평가
    avg_psnr, avg_ssim = 0, 0
    if args['test_loader'] is not None:
        pbar = args['test_loader']
        for batch in pbar:
            lq, gt = batch['lq'].to(device), batch['gt'].to(device)
            with torch.no_grad():
                output = model(lq)
            output_img, gt_img = tensor2img(output), tensor2img(gt)
            avg_psnr += calculate_psnr(output_img, gt_img, crop_border=upscale, test_y_channel=True)
            avg_ssim += calculate_ssim(output_img, gt_img, crop_border=upscale, test_y_channel=True)
        avg_psnr /= len(pbar)
        avg_ssim /= len(pbar)

    print(f"  - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, GFLOPs: {gflops:.4f}, Params(M): {m_params:.4f}")

    return {
        'Model': model_name,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'GFLOPs': gflops,
        'Params (M)': m_params
    }

# --- 메인 함수 ---

def main():
    parser = argparse.ArgumentParser(description="다양한 모델의 성능을 비교하고 트레이드오프 그래프를 생성합니다.")
    parser.add_argument('--config', type=str, default='config_catanet.yml', help='모델 설정 파일')
    parser.add_argument('--output_csv', type=str, default='tradeoff_results.csv', help='결과를 저장할 CSV 파일 경로')
    parser.add_argument('--output_image', type=str, default='tradeoff_visualization.png', help='그래프를 저장할 이미지 파일 경로')
    
    cli_args = parser.parse_args()

    # --- 기본 경로 및 설정 로드 ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(base_dir, cli_args.config), 'r') as f:
        config_data = yaml.safe_load(f)

    dataset_opt = {
        'name': 'Set5', 'type': 'PairedImageDataset',
        'dataroot_gt': os.path.join(base_dir, 'datasets/Set5/HR'),
        'dataroot_lq': os.path.join(base_dir, 'datasets/Set5/LR_bicubic/X2'),
        'filename_tmpl': '{}x2', 'io_backend': {'type': 'disk'},
        'scale': config_data['scale'], 'phase': 'val'
    }
    try:
        test_set = build_dataset(dataset_opt)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"[경고] 테스트 데이터셋 로드 실패: {e}. PSNR/SSIM은 0으로 표시됩니다.")
        test_loader = None

    # 분석할 모델 목록
    models_to_analyze = {
        "CATANet-L (Original)": {
            "path": os.path.join(base_dir, 'weights/CATANet-L_x2.pth'),
            "type": "original"
        },
        "Finetuned (Output KD)": {
            "path": os.path.join(base_dir, 'weights/rebuilt_output.pth'),
            "type": "rebuilt"
        },
        "Finetuned (Feature KD)": {
            "path": os.path.join(base_dir, 'weights/rebuilt_feature.pth'),
            "type": "rebuilt"
        },
        "Finetuned (FaKD)": {
            "path": os.path.join(base_dir, 'weights/rebuilt_fakd.pth'),
            "type": "rebuilt"
        }
    }

    # 공유될 인자
    shared_args = {
        'config_data': config_data,
        'masks_path': os.path.join(base_dir, 'weights/catanet_pruning_masks.pth'),
        'test_loader': test_loader
    }

    # --- 각 모델에 대한 메트릭 계산 ---
    results = []
    for name, info in models_to_analyze.items():
        if os.path.exists(info['path']):
            metrics = get_model_metrics(name, info['path'], info['type'], shared_args)
            results.append(metrics)
        else:
            print(f"[경고] 모델 파일 '{info['path']}'를 찾을 수 없어 건너뜁니다.")

    if not results:
        print("분석할 모델이 하나도 없습니다. 스크립트를 종료합니다.")
        return

    # --- 결과 저장 및 시각화 ---
    df = pd.DataFrame(results)
    output_csv_path = os.path.join(base_dir, cli_args.output_csv)
    df.to_csv(output_csv_path, index=False)
    print(f"\n결과가 '{output_csv_path}'에 저장되었습니다.")
    print(df.to_string())

    # 시각화
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # PSNR vs GFLOPs
    sns.scatterplot(data=df, x='GFLOPs', y='PSNR', hue='Model', s=200, ax=ax1, palette='viridis')
    ax1.set_xlabel('GFLOPs (Lower is Better)', fontsize=12)
    ax1.set_ylabel('PSNR (dB) (Higher is Better)', fontsize=12)
    
    for i, row in df.iterrows():
        ax1.text(row['GFLOPs'] + 0.05, row['PSNR'], f"{row['Model']}\n({row['GFLOPs']:.2f}G, {row['PSNR']:.2f}dB)", fontsize=9)

    ax1.set_title('Performance vs. Complexity Trade-off', fontsize=16)
    ax1.legend().set_visible(False)
    
    # 이미지 파일로 저장
    output_image_path = os.path.join(base_dir, cli_args.output_image)
    plt.savefig(output_image_path, dpi=300)
    print(f"트레이드오프 그래프가 '{output_image_path}'에 저장되었습니다.")
    plt.show()


if __name__ == '__main__':
    main()
