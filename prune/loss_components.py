"""
[리팩토링 노트]
이 스크립트의 `KLDiv` 함수는 초해상도와 같은 이미지 대 이미지(image-to-image)
작업과 호환되도록 수정되었습니다.

[원본과의 주요 차이점]
-   이전 버전: `KLDiv` 함수는 모델의 출력이 분류 작업에 적합한 `.logits` 속성을
    가진 확률 분포라고 가정했습니다. 이 함수는 쿨백-라이블러 발산(Kullback–Leibler
    divergence)을 계산했습니다.
-   현재 버전: `KLDiv` 함수는 이제 두 이미지 텐서(기준선 대 가지치기된 출력)를
    이미지 비교를 위한 표준 메트릭인 평균 제곱 오차(Mean Squared Error, MSE)를
    사용하여 직접 비교합니다. 이 함수는 음수 MSE를 반환하여, 제거되었을 때
    가장 적은 오류를 유발하는 구성 요소를 찾는(즉, 음수 점수를 최대화하는)
    메인 가지치기 스크립트의 목표와 일치시킵니다.
"""
import torch
from torch.nn import functional as F
import numpy as np

# Base MD Implementation
def manifold_Distillation(args, teacher, student):
    err = 0
    F_s = student
    F_t = teacher
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    
    
    K = 768 # Subject to change based on architecture.
    if F_s.shape[1] == 0:
        return torch.tensor(0.0).to(F_s.device)
        
    bsz, patch_num, _ = F_s.shape
    
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler]
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler]

    M_s = f_s.mm(f_s.T)
    M_t = f_t.mm(f_t.T)

    M_diff = M_t - M_s
    
    if args is None:
        loss_mf_rand = (M_diff * M_diff).sum()
    elif args.aggregate == 'sum':
        loss_mf_rand = (M_diff * M_diff).sum()
    elif args.aggregate == 'mean':
        loss_mf_rand = (M_diff * M_diff).mean()
    else:
        raise Exception('aggregate not specified')
    
    err += -1*loss_mf_rand
    
    return err


# MD Applied to CNN-based Architectures
def cnn_mmd(teacher, student):
    err = 0
    F_s = student
    F_t = teacher
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    
    
    loss_mf_rand += torch.sum((torch.mean(F_t, dim=0) - torch.mean(F_s, dim=0)))**2
    
    err += -1*loss_mf_rand
    
    return err


def KLDiv(base_output_image, pruned_output_image):
    """
    MODIFIED for Image-to-Image Tasks.
    
    Calculates the difference between two output images.
    Replaces the original KL Divergence for logits with Mean Squared Error (MSE),
    which is suitable for comparing images. 
    
    Returns a negative value to maintain the original script's expectation of 
    maximizing a negative importance score (i.e., minimizing the error).
    """
    # The original implementation expected `.logits` on the outputs.
    # We now directly compare the output tensors.
    loss = F.mse_loss(pruned_output_image, base_output_image)
    return -loss


## Alternate Patch based MD -- only for Vision
def patch_based_manifold_Distillation(teacher, student, layer):
    err = 0
    F_s = student
    F_t = teacher
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    
    # manifold loss among different samples (inter-sample) -- directly compares along token
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))

    M_diff = M_t.mean(0) - M_s.mean(0)
    loss_mf_sample = (M_diff * M_diff).sum()
    
    err += -1*loss_mf_sample
    
    return err