import numpy as np
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import S7.config as config

def train_loader_cifar10(trainset, shuffle=True, num_workers=2,
                         mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)):
    """
    Function for getting a trainloader iterator
    """
    trainloader      = torch.utils.data.DataLoader(trainset, batch_size = config.BATCH_SIZE_TRAIN,
                                                    shuffle=shuffle, num_workers=config.num_workers)
    return trainloader

def test_loader_cifar10(testset, shuffle=False, num_workers=2,
                         mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)):
    """
    Function for getting a testloader iterator
    """

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

def plot_images(loader, rows=5, cols=5,mean=(0,0,0),std=(1,1,1),classes=[0,0,0]):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    num_row     = rows
    num_col     = cols
    num_images  = num_row*num_col

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.8*num_col,2.5*num_row))
    for i in range(num_images):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i].numpy().transpose((1,2,0))*std + mean)
        ax.set_title('Label: {}'.format(classes[labels[i]]))
    plt.show()
