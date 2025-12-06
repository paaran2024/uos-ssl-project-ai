import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F # FaKD 구현을 위해 추가
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os

# --- 커스텀 모듈 임포트 ---
from scripts.load_catanet import get_catanet_teacher_model
from basicsr.data import build_dataset
from basicsr.metrics import calculate_psnr
from basicsr.utils import tensor2img
from utils.catanet_hooks import CATANetModelHooking

"""
finetune_pruned_model.py: 지식 증류(Knowledge Distillation)를 사용하여
                           가지치기된(pruned) 모델을 파인튜닝하는 스크립트입니다.

이 스크립트는 이제 세 가지 증류 방식을 지원합니다:
1.  Output Distillation: 교사 모델의 최종 출력을 학생 모델이 모방합니다. (기본)
2.  Feature Distillation: 교사 모델의 중간 피처맵을 학생 모델이 직접 모방합니다.
3.  FaKD: 교사 모델 피처맵의 구조적 관계(Affinity)를 학생 모델이 모방합니다.

실행 방법 (ai/ 디렉토리에서):
    # FaKD 사용 예시
    python finetune_pruned_model.py --config config_catanet.yml \
                                     --teacher_weights weights/CATANet-L_x2.pth \
                                     --pruned_weights weights/catanet_pruned.pth \
                                     --save_path weights/catanet_finetuned_fakd.pth \
                                     --distillation_type fakd \
                                     --beta 100 
"""

def calculate_fakd_loss(fm_teacher, fm_student):
    """
    Feature-Affinity based Knowledge Distillation (FaKD) 손실을 계산합니다.
    피처맵의 2차 통계 정보(Gram 행렬)를 비교합니다.
    """
    # fm_teacher, fm_student shape: (B, C, H, W)
    
    # 1. 피처맵을 (B, C, H*W) 형태로 재구성합니다.
    b, c, h, w = fm_teacher.shape
    fm_teacher_reshaped = fm_teacher.view(b, c, h * w)
    fm_student_reshaped = fm_student.view(b, c, h * w)

    # 2. 채널 차원을 따라 L2 정규화를 수행합니다.
    fm_teacher_normalized = F.normalize(fm_teacher_reshaped, p=2, dim=1)
    fm_student_normalized = F.normalize(fm_student_reshaped, p=2, dim=1)

    # 3. Gram 행렬 (Affinity Matrix)을 계산합니다.
    # (B, C, N) -> (B, N, C)로 전치 후 행렬 곱셈
    affinity_teacher = torch.bmm(fm_teacher_normalized.transpose(1, 2), fm_teacher_normalized)
    affinity_student = torch.bmm(fm_student_normalized.transpose(1, 2), fm_student_normalized)
    
    # 4. 두 Affinity 행렬 간의 L1 손실을 계산합니다.
    loss = F.l1_loss(affinity_student, affinity_teacher)
    
    return loss

def main():
    # --- 0. 스크립트 실행 위치 보정 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory changed to: {os.getcwd()}")

    # --- 1. 인자 파싱 및 설정 ---
    parser = argparse.ArgumentParser(description="Pruned CATANet Fine-tuning with Knowledge Distillation")
    parser.add_argument("--config", required=True, help="모델 및 데이터셋 설정을 담은 YAML 파일")
    parser.add_argument("--teacher_weights", required=True, help="원본 교사 모델 가중치 경로")
    parser.add_argument("--pruned_weights", required=True, help="가지치기된 학생 모델 가중치 경로")
    parser.add_argument("--save_path", required=True, help="파인튜닝된 모델을 저장할 경로")
    parser.add_argument("--epochs", type=int, default=10, help="파인튜닝 에폭 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--alpha", type=float, default=0.8, help="Output Distillation Loss 가중치")
    # MODIFIED: 'fakd'를 선택지에 추가
    parser.add_argument("--distillation_type", type=str, default="output", choices=["output", "feature", "fakd"], help="증류 타입 선택")
    parser.add_argument("--beta", type=float, default=0.5, help="Feature/FaKD Distillation Loss 가중치")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 데이터셋 로드 ---
    train_opt, val_opt = config['datasets']['train'], config['datasets']['val']
    train_opt['scale'], val_opt['scale'] = args.scale, args.scale
    train_opt['phase'], val_opt['phase'] = 'train', 'val'
    
    train_set = build_dataset(train_opt)
    val_set = build_dataset(val_opt)

    train_loader = DataLoader(train_set, batch_size=train_opt['batch_size_per_gpu'], shuffle=True, num_workers=train_opt.get('num_worker_per_gpu', 4), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    print(f"데이터셋 로드 완료. Train: {len(train_set)}개, Val: {len(val_set)}개")

    # --- 3. 모델 로드 ---
    print("교사 모델 로딩...")
    teacher_model = get_catanet_teacher_model(weights_path=args.teacher_weights, upscale=args.scale).to(device)
    teacher_model.eval()

    print("학생 모델 로딩 (가지치기된 가중치)...")
    student_model = get_catanet_teacher_model(weights_path=None, upscale=args.scale).to(device)
    pruned_state = torch.load(args.pruned_weights, map_location=device)['params']
    student_model.load_state_dict(pruned_state, strict=False)
    student_model.train()
    
    teacher_hook = CATANetModelHooking(args=None, model=teacher_model)
    student_hook = CATANetModelHooking(args=None, model=student_model)

    # MODIFIED: 'feature' 또는 'fakd'일 때 hook을 등록
    if args.distillation_type in ['feature', 'fakd']:
        print(f"'{args.distillation_type}' 증류를 위해 hook을 활성화합니다.")
        teacher_hook.apply_mask_and_hooks()
        student_hook.apply_mask_and_hooks()
    
    print("모델 로드 및 후킹 완료.")

    # --- 4. 옵티마이저 및 손실 함수 설정 ---
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    l1_loss = torch.nn.L1Loss().to(device)
    best_psnr = 0.0
    
    # --- 5. 파인튜닝 학습 루프 ---
    for epoch in range(1, args.epochs + 1):
        student_model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for batch in pbar:
            lq, gt = batch['lq'].to(device), batch['gt'].to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_output, teacher_fms = teacher_hook.forwardPass(lq)
            student_output, student_fms = student_hook.forwardPass(lq)

            loss_task = l1_loss(student_output, gt)
            loss_distill_output = l1_loss(student_output, teacher_output)
            loss = (1 - args.alpha) * loss_task + args.alpha * loss_distill_output
            
            # MODIFIED: FaKD 로직 추가
            if args.distillation_type == 'feature' or args.distillation_type == 'fakd':
                if not (student_fms and teacher_fms):
                    print("경고: 피처맵을 가져올 수 없어 중간 증류를 건너뜁니다.")
                else:
                    intermediate_loss = 0
                    for student_fm, teacher_fm in zip(student_fms, teacher_fms):
                        if args.distillation_type == 'feature':
                            intermediate_loss += l1_loss(student_fm, teacher_fm)
                        elif args.distillation_type == 'fakd':
                            intermediate_loss += calculate_fakd_loss(teacher_fm, student_fm)
                    loss += args.beta * intermediate_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch} 완료. 평균 Loss: {total_loss / len(train_loader):.4f}")

        # --- 6. 검증 (Validation) ---
        student_model.eval()
        current_psnr = 0
        with torch.no_grad():
            for batch in val_loader:
                lq, gt = batch['lq'].to(device), batch['gt'].to(device)
                student_output = student_model(lq)
                output_img, gt_img = tensor2img(student_output), tensor2img(gt)
                current_psnr += calculate_psnr(output_img, gt_img, crop_border=args.scale, test_y_channel=True)
        
        avg_psnr = current_psnr / len(val_loader)
        print(f"검증 완료. 평균 PSNR: {avg_psnr:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(student_model.state_dict(), args.save_path)
            print(f"최고 성능 달성! 모델을 '{args.save_path}'에 저장했습니다. (PSNR: {best_psnr:.4f})")
            
        # 훅 재설정 (메모리 누수 방지)
        if args.distillation_type in ['feature', 'fakd']:
            teacher_hook.purge_hooks()
            student_hook.purge_hooks()
            if epoch < args.epochs:
                 teacher_hook.apply_mask_and_hooks()
                 student_hook.apply_mask_and_hooks()

    print(f"\n--- 파인튜닝 완료 ---\n최고 PSNR: {best_psnr:.4f}")
    print(f"최종 모델은 '{args.save_path}'에 저장되었습니다.")


if __name__ == '__main__':
    main()