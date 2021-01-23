import torch
import torchvision.transforms as transforms

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE_TRAIN  = 64
BATCH_SIZE_TEST   = 512
num_workers = 0
input_size_CIFAR10 = (3, 32, 32)

CIFAR_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

to_tensor = transforms.Compose([transforms.ToTensor()])

train_transform  = transforms.Compose([torchvision.transforms.RandomAffine(degrees=8, translate=(0.1,0.1), scale=(0.95,1.05))
                                          ,transforms.ToTensor()
                                          ,transforms.Normalize((0.49186122, 0.48266134, 0.44720834), (0.24699295, 0.24340236, 0.26160896))])

test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.49186122, 0.48266134, 0.44720834), (0.24699295, 0.24340236, 0.26160896))])
