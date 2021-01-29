import torch
import torchvision.transforms as transforms

# Following are the conguration parameters which will be used throughout the code

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE_TRAIN  = 128
BATCH_SIZE_TEST   = 1024
EPOCHS   = 25
LR       = 0.001
MOMENTUM = 0.9
num_workers = 4
LR_STEP = 5
LR_GAMMA = 0.8
input_size_CIFAR10 = (3, 32, 32)

CIFAR_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

to_tensor = transforms.Compose([transforms.ToTensor()])

train_transform  = transforms.Compose([transforms.RandomAffine(degrees=6, translate=(0.05,0.05), scale=(0.95,1.05))
                                          ,transforms.ToTensor()
                                          ,transforms.Normalize((0.49186122, 0.48266134, 0.44720834), (0.24699295, 0.24340236, 0.26160896))])

test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.49186122, 0.48266134, 0.44720834), (0.24699295, 0.24340236, 0.26160896))
