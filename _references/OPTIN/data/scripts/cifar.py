import torchvision
import torchvision.transforms as transforms
import torch

def cifar10Dataset(args, non_loader=False):
   
   size=224
   transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])

   transform_test = transforms.Compose([
      transforms.Resize(size),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])
   
   trainset = torchvision.datasets.CIFAR10(root='./data/storage', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

   testset = torchvision.datasets.CIFAR10(root='./data/storage', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
   
   if non_loader:
      return trainset, testset
   return trainloader, testloader



def cifar100Dataset(args, non_loader=False):
   
   size=224
   transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
   ])

   transform_test = transforms.Compose([
      transforms.Resize(size),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
   ])
   
   trainset = torchvision.datasets.CIFAR100(root='./data/storage', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

   testset = torchvision.datasets.CIFAR100(root='./data/storage', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
   
   if non_loader:
      return trainset, testset
   return trainloader, testloader