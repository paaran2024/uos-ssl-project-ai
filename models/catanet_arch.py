# -*- coding: utf-8 -*-
# 이 파일은 CATANet 모델의 아키텍처를 정의합니다.
# 원본 소스: ai/references/CATANet-main/basicsr/archs/catanet_arch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
# BasicSR의 아키텍처 레지스트리를 사용하여 이 모델을 프레임워크에 등록합니다.
# from basicsr.utils.registry import ARCH_REGISTRY
# from basicsr.archs.arch_util import trunc_normal_
import math

# --- 유틸리티 함수들 ---
# 이 함수들은 모델 내에서 반복적으로 사용되는 작은 기능들을 정의합니다.

def exists(val):
    """값이 None이 아닌지 확인합니다."""
    return val is not None

def is_empty(t):
    """텐서가 비어있는지 확인합니다."""
    return t.nelement() == 0

def expand_dim(t, dim, k):
    """텐서의 특정 차원을 k배로 확장합니다."""
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    """x가 None이면 기본값 d를 반환합니다."""
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    """Exponential Moving Average (EMA)를 계산합니다."""
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    """인플레이스(In-place) 연산으로 EMA를 수행하여 메모리를 절약합니다."""
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))

def similarity(x, means):
    """입력 x와 평균(means) 간의 유사도를 계산합니다."""
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    """유사도를 계산하고, 가장 유사한 평균(mean)에 따라 버킷(bucket)을 할당합니다."""
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    """배치 단위로 bincount를 수행하여 각 버킷에 몇 개의 토큰이 있는지 셉니다."""
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets = None):
    """각 버킷에 속한 토큰들의 평균을 계산하여 새로운 중심(center)을 찾습니다."""
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means

# --- 핵심 어텐션 모듈 ---

class IASA(nn.Module):
    """
    Intra-group Aggregation Self-Attention (그룹 내 집계 셀프 어텐션)
    - 비슷한 토큰들로 구성된 각 그룹 내부에서 셀프 어텐션을 수행합니다.
    - 로컬 문맥(local context) 정보를 강화하는 역할을 합니다.
    """
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape

        # 입력 x로부터 Query, Key, Value를 생성합니다.
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        # 그룹화된 인덱스에 따라 토큰들을 재정렬합니다.
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)  # 그룹 크기
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        # 패딩을 추가하여 그룹 크기에 맞춥니다.
        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",ng=ng,h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        
        # 1. 그룹 내 셀프 어텐션 (Intra-group self-attention)
        out1 = F.scaled_dot_product_attention(paded_q,paded_k,paded_v)

        # 2. 그룹과 전역 중심 간의 크로스 어텐션 (Inter-group cross-attention)
        k_global = k_global.reshape(1,1,*k_global.shape).expand(B,ng,-1,-1,-1)
        v_global = v_global.reshape(1,1,*v_global.shape).expand(B,ng,-1,-1,-1)
        out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
        
        # 두 어텐션 결과를 합칩니다.
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]

        # 원래 순서로 토큰을 되돌립니다.
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)

        return out

class IRCA(nn.Module):
    """
    Inter-group Cross-Attention (그룹 간 크로스 어텐션)
    - 전체 이미지의 토큰 중심(token centers)을 계산하고, 이를 Key와 Value로 사용합니다.
    - 이 Key, Value는 IASA 모듈에서 전역 정보(global information)로 활용됩니다.
    """
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        x = normed_x
        # 훈련 중에는 토큰 중심을 반복적으로 업데이트하고, 평가 중에는 고정된 중심을 사용합니다.
        if self.training:
            x_global = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
        else:
            x_global = x_means

        # 전역 중심으로부터 Key와 Value를 생성합니다.
        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k,v, x_global.detach()

class TAB(nn.Module):
    """
    Token Aggregation Block (토큰 집계 블록)
    - CATANet의 핵심 구성 요소입니다.
    - IRCA를 통해 전역적인 토큰 중심(Key, Value)을 계산합니다.
    - 입력 토큰들을 내용(content)에 따라 그룹화합니다.
    - IASA를 통해 그룹 내/그룹 간 어텐션을 수행하여 효율적으로 전역적인 상호작용을 모델링합니다.
    """
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim,mlp_dim))
        self.irca_attn = IRCA(dim,qk_dim,heads)
        self.iasa_attn = IASA(dim,qk_dim,heads,group_size)
        # 'means'는 토큰 중심을 저장하는 버퍼이며, EMA를 통해 훈련 중에 업데이트됩니다.
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)

    def forward(self, x):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1,N).expand(B,-1)
        
        # 초기화: 첫 실행 시 토큰 중심(means)을 초기화합니다.
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:,N-pad_n:N, :], dims=[-2])), dim=-2)
            x_means=torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c',cnt=self.num_tokens),dim=-2).detach()
        else:
            x_means = self.means.detach()

        # 훈련 시, 토큰 중심을 반복적으로 업데이트합니다.
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter-1):
                    x_means = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))

        # IRCA를 통해 전역 Key, Value 및 업데이트된 중심을 얻습니다.
        k_global, v_global, x_means = self.irca_attn(x, x_means)

        # 내용 기반 그룹화: 각 토큰을 가장 유사한 중심에 할당(그룹화)합니다.
        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j',
                                        F.normalize(x, dim=-1),
                                        F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)

        # IASA를 통해 그룹화된 토큰에 대해 어텐션을 수행합니다.
        y = self.iasa_attn(x, idx_last,k_global,v_global)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x

        # 훈련 시, EMA를 사용하여 전역 토큰 중심(means)을 부드럽게 업데이트합니다.
        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)

        return rearrange(x, 'b (h w) c->b c h w',h=h)

# --- 보조 모듈 및 클래스 ---

def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """ 정규화(Normalization)와 함수(fn)를 순차적으로 적용하는 래퍼(wrapper) 클래스입니다. """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class dwconv(nn.Module):
    """ Depth-wise convolution 구현 """
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    """ Convolutional Feed-Forward Network. MLP에 depth-wise convolution을 결합한 형태입니다. """
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    """ 표준적인 Multi-head Self-Attention 모듈입니다. """
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q,k,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class LRSA(nn.Module):
    """
    Local-Region Self-Attention (지역 셀프 어텐션)
    - 이미지를 겹치는 패치(patch)로 나누어 각 패치 내에서 어텐션을 수행합니다.
    - 지역적인 특징을 추출하고 강화하는 역할을 합니다.
    """
    def __init__(self, dim, qk_dim, mlp_dim,heads=1):
        super().__init__()
        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        # 이미지를 겹치는 패치로 나눕니다.
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        # 각 패치에 대해 어텐션과 FFN을 적용합니다.
        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)

        # 패치들을 다시 원래 이미지 형태로 합칩니다.
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        return x

# --- CATANet 전체 모델 ---

# @ARCH_REGISTRY.register() 데코레이터는 이 CATANet 클래스를 BasicSR 프레임워크에 등록합니다.
# 이를 통해 설정 파일에서 "type: CATANet"만으로 이 모델을 불러올 수 있게 됩니다.
# @ARCH_REGISTRY.register()
class CATANet(nn.Module):
    """
    CATANet (Content-Aware Token Aggregation Network)
    - 이미지 초해상도(Super-Resolution)를 위한 경량 트랜스포머 모델입니다.
    """
    # 모델의 기본 설정을 정의합니다.
    setting = dict(dim=40, block_num=8, qk_dim=36, mlp_dim=96, heads=4,
                     patch_size=[16, 20, 24, 28, 16, 20, 24, 28])

    def __init__(self,in_chans=3,n_iters=[5,5,5,5,5,5,5,5],
                 num_tokens=[16,32,64,128,16,32,64,128],
                 group_size=[256,128,64,32,256,128,64,32],
                 upscale: int = 4):
        super().__init__()

        # 모델 하이퍼파라미터를 설정합니다.
        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size

        # 1. Shallow Feature Extraction (얕은 특징 추출)
        # 입력 이미지의 저수준 특징을 추출하는 초기 컨볼루션 레이어입니다.
        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        # 2. Deep Feature Extraction (깊은 특징 추출)
        # 모델의 핵심 부분으로, 여러 개의 TAB와 LRSA 블록으로 구성됩니다.
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for i in range(self.block_num):
            # 각 블록은 전역 어텐션을 위한 TAB와 지역 어텐션을 위한 LRSA로 구성됩니다.
            self.blocks.append(nn.ModuleList([TAB(self.dim, self.qk_dim, self.mlp_dim,
                                                                 self.heads, self.n_iters[i],
                                                                 self.num_tokens[i],self.group_size[i]),
                                              LRSA(self.dim, self.qk_dim,
                                                             self.mlp_dim,self.heads)]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim,3,1,1))

        # 3. Image Reconstruction (이미지 재구성)
        # 추출된 깊은 특징을 사용하여 고해상도 이미지를 복원합니다.
        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 모델 가중치를 초기화합니다.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """가중치 초기화 함수"""
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """깊은 특징 추출을 수행하는 forward 함수"""
        for i in range(self.block_num):
            residual = x
            global_attn,local_attn = self.blocks[i]
            x = global_attn(x) # 전역 어텐션 (TAB)
            x = local_attn(x, self.patch_size[i]) # 지역 어텐션 (LRSA)
            x = residual + self.mid_convs[i](x)
        return x

    def forward(self, x):
        """모델의 전체 forward pass"""
        # 업스케일링된 기본 이미지를 준비합니다 (Residual learning).
        if self.upscale != 1:
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else:
            base = x
        
        # 1. 얕은 특징 추출
        x = self.first_conv(x)

        # 2. 깊은 특징 추출
        x = self.forward_features(x) + x

        # 3. 이미지 재구성
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        
        # 최종 출력에 기본 이미지를 더합니다.
        out = self.last_conv(out) + base

        return out

# 아래 코드는 이 파일이 직접 실행될 때 테스트용으로 사용됩니다.
if __name__ == '__main__':
    # 모델이 정상적으로 생성되고 실행되는지 확인하는 간단한 테스트 코드
    model = CATANet(upscale=3).cuda()
    x = torch.randn(2, 3, 128, 128).cuda()
    print(model)
    # print(model(x).shape)
