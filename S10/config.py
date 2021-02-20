import torch
import torchvision.transforms as transforms

# Following are the conguration parameters which will be used throughout the code

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE_TRAIN  = 64
BATCH_SIZE_TEST   = 1024
EPOCHS   = 50
num_workers = 4
input_size_CIFAR10 = (3, 32, 32)

CIFAR_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
