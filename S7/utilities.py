
import numpy as np
import torchvision.datasets
import torch.utils.data
import torchvision.transforms as transforms

def train_loader_cifar10(download_folder, batch_size=4, 
                         shuffle=True, num_workers=1, 
                         mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)):
    """
    Function for getting a trainloader iterator, as well as returns the train dataset
    """
    train_transform  = transforms.Compose([torchvision.transforms.RandomAffine(degrees=8, translate=(0.1,0.1), scale=(0.95,1.05))
                                            ,transforms.ToTensor()
                                            ,transforms.Normalize(mean, std)])

    trainset    = torchvision.datasets.CIFAR10(root=download_folder, train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers)

    return trainset, trainloader

def test_loader_cifar10(download_folder, batch_size=4, 
                         shuffle=False, num_workers=1, 
                         mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)):
    """
    Function for getting a testloader iterator, as well as returns the test dataset
    """
    test_transform = transforms.Compose([transforms.ToTensor()
                                        ,transforms.Normalize(mean, std)])

    testset    = torchvision.datasets.CIFAR10(root=download_folder, train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return testset, testloader


def get_mean_std_overall(train_loader,test_loader):
    """
    Function for getting the mean and standard devitation of data in a dataloader iterator
    """
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
    classes_in_train = train_set.targets.unique().numpy()
    classes_in_test  = test_set.targets.unique().numpy()
    assert np.isin(classes_in_test,classes_in_train).all()
    num_classes = len(train_set.targets.unique().numpy())
    print('Number of classes in MNIST: {}'.format(num_classes))
    print('Number of images for training  : {}'.format(len(train_set)))
    print('Number of images for validation: {}'.format(len(test_set)))
