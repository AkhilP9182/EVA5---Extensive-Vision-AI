import torch

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE_TRAIN  = 64
BATCH_SIZE_TEST   = 512
num_workers = 0
pin_memory  = True
input_size_MNIST   = (1, 28, 28)
input_size_CIFAR10 = (3, 32, 32)

CIFAR_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
