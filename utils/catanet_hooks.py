"""
[리팩토링 노트]
이 스크립트는 OPTIN 프레임워크의 기존 `_hooks.py`를 대체하는 파일입니다.
`CATANet` 아키텍처와 호환되도록 특별히 리팩토링되었습니다.

[원본(HuggingFace/ViT 용)과의 주요 차이점]
1.  모델 구조 탐색:
    -   원본 hook은 `model.encoder.layer`와 같은 모델 구조를 가정했습니다.
    -   이 버전은 `CATANet`의 구조에 맞춰져, `model.blocks` 및
      `model.mid_convs`를 통해 블록에 접근합니다. 여기서 `model`은
      `net_g` 네트워크 자체를 의미합니다.

2.  Forward Pass 시그니처:
    -   원본 `forwardPass`는 모델이 딕셔너리 언패킹 인자(예: `model(**batch)`)를
      받는다고 가정했으며, 이는 HuggingFace 모델에서 일반적입니다.
    -   이 버전의 `forwardPass`는 `CATANet`의 `forward` 메서드가 기대하는
      단일 텐서 `model(input_tensor)`를 받습니다.
      
3.  Hook 위치:
    -   중간 출력을 캡처하기 위한 hook 위치가 ViT 스타일의 출력 레이어에서
      `CATANet`의 주요 블록 출력을 나타내는 `mid_convs` 모듈로 변경되었습니다.
    -   뉴런 가지치기(neuron pruning)를 위한 pre-hook은 이제 `CATANet`의
      `ConvFFN` 블록 내의 활성화 함수를 대상으로 합니다.
"""
import torch
from collections import OrderedDict

class CATANetModelHooking:
    """
    `CATANet` 아키텍처를 위해 특별히 설계된 커스텀 모델 후킹 클래스입니다.
    PyTorch hook을 사용하여 마스크를 적용하고 가지치기 분석을 위한
    중간 레이어 출력을 캡처합니다.
    """
    def __init__(self, args, model=None, maskProps=None, disable_grad=False): # MODIFIED: disable_grad 추가
        super(CATANetModelHooking, self).__init__()
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.maskProps = maskProps
        self.disable_grad = disable_grad # MODIFIED: disable_grad 저장
        
        self.layer_outputs = []
        self.registered_hooks = []
        self.head_mask = None
        
        if self.maskProps:
            self.apply_mask_and_hooks()

    def apply_mask_and_hooks(self):
        """maskProps에 따라 마스크를 적용하고 forward hook을 등록합니다."""
        self.register_forward_hooks()
        
        if self.maskProps and self.maskProps.get('state') == 'neuron':
            self.register_neuron_pre_hook()
        elif self.maskProps and self.maskProps.get('state') == 'head':
            self.register_head_pre_hook()
            
    def purge_hooks(self):
        """등록된 모든 hook을 제거합니다."""
        for handle in self.registered_hooks:
            handle.remove()
        self.registered_hooks = []
        self.layer_outputs = []

    def _record_layer_output_hook(self, name):
        """모듈의 출력을 기록하기 위한 hook 함수입니다."""
        def hook(module, input, output):
            self.layer_outputs.append(output)
        return hook

    def _get_head_mask_hook(self, mask, num_heads):
        def hook(_, inputs):
            x = inputs[0]
            head_dim = x.shape[-1] // num_heads
            x = x.view(x.shape[0], x.shape[1], num_heads, head_dim)
            mask_reshaped = mask.view(1, 1, num_heads, 1).to(x.device)
            x = x * mask_reshaped
            x = x.view(x.shape[0], x.shape[1], -1)
            return (x,)
        return hook

    def register_head_pre_hook(self):
        layer_idx = self.maskProps["layer"]
        head_mask_for_layer = self.maskProps["mask"][layer_idx]

        try:
            attention_modules = [
                self.model.blocks[layer_idx][0].iasa_attn,
                self.model.blocks[layer_idx][1].layer[0].fn
            ]

            for attn_module in attention_modules:
                target_module = attn_module.proj
                hook_handle = target_module.register_forward_pre_hook(
                    self._get_head_mask_hook(head_mask_for_layer, attn_module.heads)
                )
                self.registered_hooks.append(hook_handle)
        except (AttributeError, IndexError) as e:
            print(f"오류: 헤드 hook을 적용할 레이어 {layer_idx}의 모듈을 찾을 수 없습니다: {e}")

    def register_forward_hooks(self):
        for i, block_module in enumerate(self.model.mid_convs):
            hook_handle = block_module.register_forward_hook(self._record_layer_output_hook(f"block_{i}"))
            self.registered_hooks.append(hook_handle)

    def register_neuron_pre_hook(self):
        layer_idx = self.maskProps["layer"]
        mask = self.maskProps["mask"]
        
        try:
            target_module = self.model.blocks[layer_idx][0].mlp.fn.act
            
            def hook(_, inputs):
                return (inputs[0] * mask.to(self.device),)
            
            hook_handle = target_module.register_forward_pre_hook(hook)
            self.registered_hooks.append(hook_handle)
        except IndexError:
            print(f"오류: 뉴런 hook을 적용할 레이어 {layer_idx}의 모듈을 찾을 수 없습니다.")
            
    def forwardPass(self, input_tensor):
        """
        hook이 적용된 모델로 forward pass를 수행합니다.
        `disable_grad` 플래그에 따라 경사도 계산을 제어합니다.
        """
        self.layer_outputs = []
        if self.disable_grad: # MODIFIED: disable_grad에 따라 조건부로 no_grad 적용
            with torch.no_grad():
                output_image = self.model(input_tensor)
        else:
            output_image = self.model(input_tensor)
        
        return output_image, self.layer_outputs