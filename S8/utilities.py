import numpy as np
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import S8.config as config
from S8.resnet import ResNet18

def train_loader_cifar10(trainset, shuffle=True, num_workers=2):
    """
    Function for getting a trainloader iterator
    """
    trainloader      = torch.utils.data.DataLoader(trainset, batch_size = config.BATCH_SIZE_TRAIN,
                                                    shuffle=shuffle, num_workers=config.num_workers)
    return trainloader

def test_loader_cifar10(testset, shuffle=False, num_workers=2):
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

def plot_loss(train_loss_vals,test_loss_vals,epochs):
    x = [i for i in range(0,epochs)]
    y_train = train_loss_vals
    y_test = test_loss_vals

    loss = plt.figure()
    plt.title("Train/Validation Loss on CIFAR 10")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")

    plt.plot(x, y_train, color='b')
    plt.plot(x, y_test, color='r')
    plt.legend(loc='best',fancybox=True,shadow=True)

    plt.show()
    my_dpi = 100
    loss.savefig('S8/loss.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)


def plot_acc(train_acc_vals,test_acc_vals,epochs):
    x = [i for i in range(0,epochs)]
    y_train = train_acc_vals
    y_test = test_acc_vals

    acc = plt.figure()
    plt.title("Train/Validation Accuracy on CIFAR 10")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")

    plt.plot(x, y_train, color='b')
    plt.plot(x, y_test, color='r')
    plt.legend(loc='best',fancybox=True,shadow=True)

    plt.show()
    my_dpi = 100
    acc.savefig('S8/accuracy.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

def plot_misclassified(Net,MODEL_PATH,test_loader,
                        rows=5, cols=5, mean=(0,0,0), std=(1,1,1), classes=[0,0,0]):
    num_row     = 5
    num_col     = 5
    num_images  = num_row*num_col
    batch_size_test = config.BATCH_SIZE_TEST

    # Empty lists for storing the misclassified images, their corresponding predicitons and labels
    mis_images = []
    mis_labels = []
    mis_pred   = []
    device = config.DEVICE

    # MODEL_PATH = "S8/models/S8_best_resnet18_model.model"
    model = ResNet18()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            images,labels  = data.to(device), target.to(device)
            output         = model(images)
            pred           = (output.argmax(dim=1, keepdim=True)).view_as(labels)
            mis_classified = pred.ne(labels).tolist()
            # 'mis_classified' is a Boolean list (size same as labels). 
            # It is True where the prediction does not match label
            if any(mis_classified) == True:
                mis_images.extend(images[mis_classified].cpu())
                mis_labels.extend(labels[mis_classified].cpu())
                mis_pred.extend(pred[mis_classified].cpu())
            
            if len(mis_pred)>=num_images:
                mis_images = mis_images[:25]
                mis_labels = mis_labels[:25]
                mis_pred = mis_pred[:25]
                break

    # Plot the digit images with label and predictions
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.8*num_col,2.5*num_row))
    for i in range(num_images):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(mis_images[i].numpy().transpose((1,2,0))*std + mean)
        ax.set_title('Label: {}\nPrediction: {}'.format(classes[mis_labels[i]],
                                                        classes[mis_pred[i]]))

    plt.show()
    my_dpi = 100
    fig.savefig('S8/S8_misclassified_images.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
