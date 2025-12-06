from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch


class ImageClassificationCollator:
   def __init__(self, feature_extractor): 
      self.feature_extractor = feature_extractor
   def __call__(self, batch):  
      encodings = self.feature_extractor([x[0] for x in batch],
      return_tensors='pt')   
      encodings['labels'] = torch.tensor([x[1] for x in batch],    
      dtype=torch.long)
      return encodings
  
  
def queryImageNetDataSets(args):
   transform = T.Compose([
         T.Resize(256, interpolation=3),
         T.CenterCrop(224),
         T.ToTensor(),
         T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

   train_dataset = ImageFolder("/nobackup1/ImageNet/train", transform=transform)
   val_dataset = ImageFolder("/nobackup1/ImageNet/val", transform=transform)
   return train_dataset, val_dataset
 
 
def imageNetDataset(args):
    
    
   train_dataset, val_dataset = queryImageNetDataSets(args)
   
   train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
   
   val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
   num_workers=2)

   return train_loader, val_loader