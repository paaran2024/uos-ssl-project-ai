import timm
import timm.data
from utils.utility import GLUE_TASKS
# transformers 라이브러리에서 Vision Transformer 모델과 자동 모델 분류 클래스를 가져옵니다.
from transformers import ViTForImageClassification, AutoModelForImageClassification
import torch

# 이 함수는 설정 파일(config.yaml)의 내용을 담은 args 객체를 인자로 받습니다.
# args에 포함된 task_name, model_name, dataset 등의 정보에 따라
# 적절한 사전 훈련된 비전 모델을 Hugging Face Hub에서 로드하는 역할을 합니다.
def gen_vision_model(args):
    
    # 설정 파일에 명시된 데이터셋 이름에 따라 모델 로딩 방식을 분기합니다.
    if args.dataset == "ImageNet":
        # --- ImageNet 데이터셋을 위한 모델 로딩 ---

        # model_name에 'deit'가 포함된 경우 (예: deit-base-patch16-224)
        if "deit" in args.model_name:
            # Hugging Face Hub의 'facebook' 그룹에서 해당 DeiT 모델을 로드합니다.
            # from_pretrained 메소드가 모델 아키텍처와 사전 훈련된 가중치를 모두 불러옵니다.
            model = ViTForImageClassification.from_pretrained("facebook/{}".format(args.model_name))
            # 로드된 모델의 설정값(레이어 수, 히든 사이즈 등)을 config 변수에 저장합니다.
            config = model.config
        
        # model_name에 'vit'가 포함된 경우 (예: vit-base-patch16-224)
        elif "vit" in args.model_name:
            # Hugging Face Hub의 'google' 그룹에서 해당 ViT 모델을 로드합니다.
            model = ViTForImageClassification.from_pretrained("google/{}".format(args.model_name))
            config = model.config
            
        # model_name에 'swin'이 포함된 경우 (예: swin-tiny-patch4-window7-224)
        elif "swin" in args.model_name:
            # Hugging Face Hub의 'microsoft' 그룹에서 해당 Swin Transformer 모델을 로드합니다.
            model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            config = model.config
            
    elif args.dataset == "Cifar10":
        # --- Cifar10 데이터셋을 위한 모델 로딩 ---

        if "vit-base" in args.model_name:
            # Hugging Face Hub의 'nateraw' 그룹에서 Cifar10용으로 파인튜닝된 ViT 모델을 로드합니다.
            model = ViTForImageClassification.from_pretrained('nateraw/{}'.format(args.model_name))
            config = model.config
            
        elif "deit" in args.model_name:
            model = ViTForImageClassification.from_pretrained("facebook/{}".format(args.model_name))
            config = model.config
            
    elif args.dataset == "Cifar100":
        # --- Cifar100 데이터셋을 위한 모델 로딩 ---

        if "vit-base" in args.model_name:
            # Hugging Face Hub의 'Ahmed9275' 그룹에서 Cifar100용으로 파인튜닝된 ViT 모델을 로드합니다.
            model = ViTForImageClassification.from_pretrained('Ahmed9275/Vit-Cifar100')
            config = model.config

    # --- 여기에 새로운 커스텀 모델 'CATANet-L'을 위한 코드를 추가할 수 있습니다. ---
    # 예시:
    # elif args.model_name == "CATANet-L":
    #     from models.architectures.catanet import CATANetL # 직접 만든 모델 클래스 임포트
    #     model = CATANetL(num_classes=1000) # 모델 인스턴스 생성
    #     # (필요하다면) 사전 훈련된 가중치 로드: model.load_state_dict(torch.load('path/to/weights.pth'))
    #     config = model.config # 모델의 설정을 config에 저장
    # -------------------------------------------------------------------------
    
    # 최종적으로 생성된 모델 객체와 설정을 반환합니다.
    # 이 값들은 main.py로 돌아가 prune, evaluation 등의 과정에 사용됩니다.
    return model, config