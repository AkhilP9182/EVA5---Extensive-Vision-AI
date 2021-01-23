import numpy as np
import torch
import torchvision
import torchvision.datasets
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import S7.config as config

to_tensor = transforms.Compose([transforms.ToTensor()])

def train_loader_cifar10(trainset, shuffle=True, num_workers=2,
                         mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)):
    """
    Function for getting a trainloader iterator
    """
    train_transform  = transforms.Compose([torchvision.transforms.RandomAffine(degrees=8, translate=(0.1,0.1), scale=(0.95,1.05))
                                          ,transforms.ToTensor()
                                          ,transforms.Normalize(mean, std)])

    trainloader      = torch.utils.data.DataLoader(trainset, batch_size = config.BATCH_SIZE_TRAIN, 
                                                shuffle=shuffle, num_workers=config.num_workers)

    return trainloader

def test_loader_cifar10(testset, shuffle=False, num_workers=2,
                         mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)):
    """
    Function for getting a testloader iterator
    """
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    testloader     = torch.utils.data.DataLoader(testset, batch_size = config.BATCH_SIZE_TEST,
                                                shuffle=shuffle, num_workers=config.num_workers)

    return testloader

def get_mean_std_overall(trainset,testset):
    """
    Function for getting the mean and standard devitation of dataset (train and test combined)
    """
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 1024, shuffle=False, num_workers=1)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size = 1024, shuffle=False, num_workers=1)
    channel_sum, channel_squared_sum, num_batches = 0,0,0

    for data, _ in train_loader:
        channel_sum += torch.mean(data, dim=[0,2,3])
        channel_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    for data, _ in test_loader:
        channel_sum += torch.mean(data, dim=[0,2,3])
        channel_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = (channel_sum/num_batches)
    std  = ((channel_squared_sum/num_batches) - mean**2)**0.5
    return mean,std

def dataset_info(train_set,test_set):
    """
        The following 7 lines are to assert whether both training and test sets have the same number/type of 
        classes (with the same labelling) for classification, and assign the number to a variable 
        'num_classes' which will be equal to the number of kernel that will be used later in the 
        final convolution layer.
    """
    classes_in_train = list(set(train_set.targets))
    classes_in_test  = list(set(test_set.targets))
    assert np.isin(classes_in_test,classes_in_train).all()
    num_classes = len(set(train_set.targets))
    print("Number of classes in CIFAR10   : {}".format(num_classes))
    print("Number of images for training  : {}".format(len(train_set)))
    print("Number of images for validation: {}".format(len(test_set)))
    return "Data loading done!"

def imshow(img,mean,std):
    img = (img*std) + mean     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_images(loader,mean,std):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images),nrow=5)
    print(' '.join('%5s' % classes[labels[j]] for j in range(5)))
