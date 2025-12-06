import sys
import os

# Add the parent directory (ai) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import argparse
import yaml
from collections import OrderedDict

# --- 헬퍼: 기존 아키텍처 파일에서 필요한 클래스들을 가져옵니다 ---
from basicsr.archs.catanet_arch import CATANet, TAB, LRSA, ConvFFN, Attention, IASA, PreNorm, IRCA

# =====================================================================================
# 1. 프루닝된 버전의 새로운 모듈 정의
# =====================================================================================

class PrunedConvFFN(ConvFFN):
    def __init__(self, in_features, hidden_features=None, out_features=None, **kwargs):
        super().__init__(in_features, hidden_features, out_features, **kwargs)

class PrunedAttention(Attention):
    def __init__(self, dim, heads, qk_dim, **kwargs):
        super().__init__(dim, heads, qk_dim, **kwargs)

class PrunedIASA(IASA):
    def __init__(self, dim, qk_dim, heads, group_size, **kwargs):
        super().__init__(dim, qk_dim, heads, group_size, **kwargs)

class PrunedTAB(TAB):
    def __init__(self, dim, qk_dim, mlp_dim, heads, group_size, **kwargs):
        super(TAB, self).__init__()
        self.n_iter = kwargs.get('n_iter', 3)
        self.ema_decay = kwargs.get('ema_decay', 0.999)
        self.num_tokens = kwargs.get('num_tokens', 8)
        self.norm = nn.LayerNorm(dim)
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.mlp = PreNorm(dim, PrunedConvFFN(dim, mlp_dim))
        self.iasa_attn = PrunedIASA(dim, qk_dim, heads, group_size)
        self.irca_attn = IRCA(dim,qk_dim,heads)
        self.register_buffer('means', torch.randn(self.num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        
class PrunedLRSA(LRSA):
    def __init__(self, dim, qk_dim, mlp_dim, heads, **kwargs):
        super(LRSA, self).__init__()
        self.layer = nn.ModuleList([
            PreNorm(dim, PrunedAttention(dim, heads, qk_dim)),
            PreNorm(dim, PrunedConvFFN(dim, mlp_dim))
        ])

class PrunedCATANet(CATANet):
    def __init__(self, upscale: int, mlp_dims: list, head_counts: list, args):
        super(CATANet, self).__init__()
        
        self.setting = CATANet.setting
        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim_orig = self.setting['qk_dim']
        self.heads_orig = self.setting['heads']
        self.upscale = upscale
        self.n_iters = args.get('n_iters', [5]*8)
        self.num_tokens = args.get('num_tokens', [16,32,64,128,16,32,64,128])
        self.group_size = args.get('group_size', [256,128,64,32,256,128,64,32])

        self.first_conv = nn.Conv2d(3, self.dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        q_head_dim = self.qk_dim_orig // self.heads_orig
        
        for i in range(self.block_num):
            current_heads = head_counts[i]
            current_mlp_dim = mlp_dims[i]
            
            if current_heads == 0 or current_mlp_dim == 0:
                raise ValueError(f"레이어 {i}는 0개의 헤드/뉴런을 가질 수 없습니다. 프루닝이 너무 많이 되었습니다.")

            current_qk_dim = current_heads * q_head_dim
            tab_kwargs = {'n_iter': self.n_iters[i], 'num_tokens': self.num_tokens[i], 'group_size': self.group_size[i], 'ema_decay': 0.999}
            
            self.blocks.append(nn.ModuleList([
                PrunedTAB(self.dim, current_qk_dim, current_mlp_dim, current_heads, **tab_kwargs),
                PrunedLRSA(self.dim, current_qk_dim, current_mlp_dim, current_heads)
            ]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))

        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
    
        self.last_conv = nn.Conv2d(self.dim, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.apply(self._init_weights)

# =====================================================================================
# 2. 가중치 이식 로직
# =====================================================================================

def transfer_weights(rebuilt_model, source_state_dict, masks):
    new_state_dict = rebuilt_model.state_dict()
    head_mask = masks['head_mask']
    neuron_mask = masks['neuron_mask']
    
    orig_heads = CATANet.setting['heads']
    orig_qk_dim = CATANet.setting['qk_dim']
    orig_v_dim = CATANet.setting['dim']
    
    q_head_dim = orig_qk_dim // orig_heads
    v_head_dim = orig_v_dim // orig_heads

    for name, new_param in new_state_dict.items():
        if name not in source_state_dict:
            print(f"경고: 소스에 '{name}' 가중치가 없습니다. 건너뜁니다.")
            continue

        source_param = source_state_dict[name]
        
        if 'blocks' not in name:
            new_param.data.copy_(source_param.data)
            continue
            
        layer_idx = int(name.split('.')[1])

        if 'mlp.fn.fc1' in name:
            kept_neurons = neuron_mask[layer_idx].nonzero().squeeze(-1)
            if 'weight' in name:
                new_param.data.copy_(source_param.data[kept_neurons, :])
            elif 'bias' in name:
                new_param.data.copy_(source_param.data[kept_neurons])
        elif 'mlp.fn.fc2.weight' in name:
            kept_neurons = neuron_mask[layer_idx].nonzero().squeeze(-1)
            new_param.data.copy_(source_param.data[:, kept_neurons])
        elif 'mlp.fn.dwconv' in name:
             # dwconv의 가중치는 그룹 컨볼루션이므로, 차원 변경에 맞춰 이식
            kept_neurons = neuron_mask[layer_idx].nonzero().squeeze(-1)
            if 'depthwise_conv.0.weight' in name or 'depthwise_conv.0.bias' in name:
                new_param.data.copy_(source_param.data[kept_neurons])
        
        elif 'iasa_attn' in name or 'layer.0.fn' in name:
            kept_heads = head_mask[layer_idx].nonzero().squeeze(-1)
            
            if 'to_q.weight' in name or 'to_k.weight' in name:
                new_w = torch.cat([source_param.data[h_idx * q_head_dim : (h_idx + 1) * q_head_dim, :] for h_idx in kept_heads], dim=0)
                new_param.data.copy_(new_w)
                
            elif 'to_v.weight' in name:
                new_w = torch.cat([source_param.data[h_idx * v_head_dim : (h_idx + 1) * v_head_dim, :] for h_idx in kept_heads], dim=0)
                new_param.data.copy_(new_w)

            elif 'proj.weight' in name:
                new_w = torch.cat([source_param.data[:, h_idx * v_head_dim : (h_idx + 1) * v_head_dim] for h_idx in kept_heads], dim=1)
                new_param.data.copy_(new_w)
            elif 'irca' in name: # irca_attn은 헤드 프루닝의 영향을 받지만, to_k/v가 공유되므로 조금 다름
                new_w = torch.cat([source_param.data[h_idx * q_head_dim : (h_idx + 1) * q_head_dim, :] for h_idx in kept_heads], dim=0)
                new_param.data.copy_(new_w)
            else:
                new_param.data.copy_(source_param.data)
        
        else:
            new_param.data.copy_(source_param.data)
            
    return new_state_dict

# =====================================================================================
# 3. 메인 실행 함수
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="프루닝된 CATANet 모델을 물리적으로 더 작은 모델로 재구성합니다.")
    parser.add_argument('--config', type=str, required=True, help='모델 설정이 담긴 원본 config 파일')
    parser.add_argument('--masks', type=str, required=True, help='프루닝 마스크가 저장된 .pth 파일')
    parser.add_argument('--source_weights', type=str, required=True, help='파인튜닝된 희소 모델의 가중치 파일')
    parser.add_argument('--save_path', type=str, required=True, help='재구성된 모델을 저장할 경로')
    
    args = parser.parse_args()

    # --- 실행 위치 보정 ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Argparse는 절대 경로를 받을 수 있으므로 CWD 변경은 불필요할 수 있으나, 일관성을 위해 유지
    
    # --- 1. 데이터 로드 ---
    print("프루닝 마스크와 소스 가중치를 로드합니다...")
    config_path = os.path.join(base_dir, args.config)
    masks_path = os.path.join(base_dir, args.masks)
    source_weights_path = os.path.join(base_dir, args.source_weights)
    save_path = os.path.join(base_dir, args.save_path)

    masks = torch.load(masks_path, map_location='cpu')
    source_data = torch.load(source_weights_path, map_location='cpu')
    source_state_dict = source_data['params'] if 'params' in source_data else source_data

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    head_mask = masks['head_mask']
    neuron_mask = masks['neuron_mask']
    
    # --- 2. 새로운 아키텍처 사양 계산 ---
    print("새로운 소형 아키텍처의 사양을 계산합니다...")
    new_head_counts = head_mask.sum(dim=-1).int().tolist()
    new_mlp_dims = neuron_mask.sum(dim=-1).int().tolist()

    print(f"  - 새로운 헤드 수 (레이어별): {new_head_counts}")
    print(f"  - 새로운 MLP 차원 (레이어별): {new_mlp_dims}")

    # --- 3. 새로운 소형 모델 생성 ---
    print("계산된 사양으로 새로운 소형 모델을 생성합니다...")
    rebuilt_model = PrunedCATANet(
        upscale=config.get('scale', 2),
        mlp_dims=new_mlp_dims,
        head_counts=new_head_counts,
        args=config
    )
    print("소형 모델 생성 완료.")

    # --- 4. 가중치 이식 ---
    print("가중치 이식을 시작합니다...")
    rebuilt_state_dict = transfer_weights(rebuilt_model, source_state_dict, masks)
    rebuilt_model.load_state_dict(rebuilt_state_dict)
    print("가중치 이식 완료.")

    # --- 5. 최종 모델 저장 ---
    print(f"재구성된 모델을 '{save_path}'에 저장합니다...")
    torch.save({'params': rebuilt_model.state_dict()}, save_path)
    print("저장 완료!")
    
    # --- 6. 검증 ---
    print("\n--- 재구성된 모델 검증 ---")
    total_params = sum(p.numel() for p in rebuilt_model.parameters())
    print(f"재구성된 모델의 총 파라미터 수: {total_params / 1e6:.4f}M")
    print("이제 'python analysis/calculate_performance.py'를 이 새로운 모델에 대해 실행하여 FLOPs 감소를 확인할 수 있습니다.")


if __name__ == '__main__':
    main()
