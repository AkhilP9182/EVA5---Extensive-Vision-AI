import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import S9.config as config
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

to_tensor = transforms.Compose([transforms.ToTensor()])

class AlbumentateTrainData(Dataset):
    def __init__(self, image_list, labels, mean=(0, 0, 0), std=(1,1,1)):
        self.image_list = image_list
        self.labels     = labels
        self.mean       = mean
        self.std        = std
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate((-8.0, 8.0), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
            A.Cutout(num_holes=2, max_h_size=12, max_w_size=12, fill_value=self.mean, p=0.5),
            A.Normalize(self.mean, self.std)
        ])

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image = self.image_list[i].convert('RGB')
        image = self.transforms(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        label = self.labels[i]
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
class AlbumentateTestData(Dataset):
    def __init__(self, image_list, labels, mean=(0, 0, 0), std=(1,1,1)):
        self.image_list = image_list
        self.labels     = labels
        self.mean       = mean
        self.std        = std
        self.transforms = A.Compose([A.Normalize(self.mean, self.std)])

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image = self.image_list[i].convert('RGB')
        image = self.transforms(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        label = self.labels[i]
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
def train_loader_cifar10(trainset, shuffle=True, num_workers=2):
    """
    Function for getting a trainloader iterator
    """
    trainloader      = DataLoader(trainset, batch_size = config.BATCH_SIZE_TRAIN,
                                                    shuffle=shuffle, num_workers=config.num_workers)
    return trainloader

def test_loader_cifar10(testset, shuffle=False, num_workers=2):
    """
    Function for getting a testloader iterator
    """

    testloader     = DataLoader(testset, batch_size = config.BATCH_SIZE_TEST,
                                                shuffle=shuffle, num_workers=config.num_workers)

    return testloader

def get_PIL_images(dataset):
    """
    Function for getting list of PIL images from the loaded torchvision dataset, their labels and their corresponding mean and std deviation
    """
    images = [dataset[i][0] for i in range(0,len(dataset))]
    labels = [dataset[i][1] for i in range(0,len(dataset))]
    
    img_array = []
    for i in range(0,len(images)):
        img_array.append(np.asarray(images[i]))
    img_array = np.array((img_array))
    mean = img_array.mean(axis=(0,1,2))
    std  = img_array.std(axis=(0,1,2))
    return images, labels, mean, std

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

    plt.plot(x, y_train, color='b',label="Training")
    plt.plot(x, y_test, color='r',label="Validation")
    plt.legend(loc ="upper right")

    plt.show()
    my_dpi = 100
    loss.savefig('S9/S9_loss.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)


def plot_acc(train_acc_vals,test_acc_vals,epochs):
    x = [i for i in range(0,epochs)]
    y_train = train_acc_vals
    y_test = test_acc_vals

    acc = plt.figure()
    plt.title("Train/Validation Accuracy on CIFAR 10")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")

    plt.plot(x, y_train, color='b',label="Training")
    plt.plot(x, y_test, color='r',label="Validation")
    plt.legend(loc ="lower right")

    plt.show()
    my_dpi = 100
    acc.savefig('S9/S9_accuracy.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

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
    mis_gradcam = []
    device = config.DEVICE

    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()
    gradcam = GradCAM(model,model.layer4)

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
                
    for img in mis_images:
        mask, _         = gradcam(img.unsqueeze(0).to(device))
        heatmap, result = visualize_cam(mask, img)
        mis_gradcam.append(result)

    # Plot the digit images with label and predictions
    print("Following are the mis-classified images:-")
    fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.8*num_col,2.5*num_row))
    for i in range(num_images):
        ax = axes1[i//num_col, i%num_col]
        ax.imshow(mis_images[i].numpy().transpose((1,2,0))*std + mean)
        ax.set_title('Label: {}\nPrediction: {}'.format(classes[mis_labels[i]],
                                                        classes[mis_pred[i]]))
    plt.show()
    
    print("Following are the GradCam Heatmaps for those mis-classified images:-")
    fig2, axes2 = plt.subplots(num_row, num_col, figsize=(1.8*num_col,2.5*num_row))
    for i in range(num_images):
        ax = axes2[i//num_col, i%num_col]
        ax.imshow(mis_gradcam[i].numpy().transpose((1,2,0))*std + mean)
        ax.set_title('Label: {}\nPrediction: {}'.format(classes[mis_labels[i]],
                                                        classes[mis_pred[i]]))

    plt.show()
    my_dpi = 100
    fig1.savefig('S9/S9_misclassified_images.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    fig2.savefig('S9/S9_misclassified_gradcam.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    
def plot_correct_classified(Net,MODEL_PATH,test_loader,
                        rows=5, cols=5, mean=(0,0,0), std=(1,1,1), classes=[0,0,0]):
    num_row     = 5
    num_col     = 5
    num_images  = num_row*num_col
    batch_size_test = config.BATCH_SIZE_TEST

    # Empty lists for storing the misclassified images, their corresponding predicitons and labels
    correct_images = []
    correct_labels = []
    correct_pred   = []
    correct_gradcam = []
    device = config.DEVICE

    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()
    gradcam = GradCAM(model,model.layer4)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            images,labels  = data.to(device), target.to(device)
            output         = model(images)
            pred           = (output.argmax(dim=1, keepdim=True)).view_as(labels)
            correct_classified = pred.eq(labels).tolist()
            # 'mis_classified' is a Boolean list (size same as labels). 
            # It is True where the prediction does not match label
            if any(correct_classified) == True:
                correct_images.extend(images[correct_classified].cpu())
                correct_labels.extend(labels[correct_classified].cpu())
                correct_pred.extend(pred[correct_classified].cpu())
            
            if len(correct_pred)>=num_images:
                correct_images = correct_images[:25]
                correct_labels = correct_labels[:25]
                correct_pred = correct_pred[:25]
                break
                
    for img in correct_images:
        mask, _         = gradcam(img.unsqueeze(0).to(device))
        heatmap, result = visualize_cam(mask, img)
        correct_gradcam.append(result)
    
    print("Following are the correctly classified images:-")
    # Plot the digit images with label and predictions
    fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.8*num_col,2.5*num_row))
    for i in range(num_images):
        ax = axes1[i//num_col, i%num_col]
        ax.imshow(correct_images[i].numpy().transpose((1,2,0))*std + mean)
        ax.set_title('Label: {}\nPrediction: {}'.format(classes[correct_labels[i]],
                                                        classes[correct_pred[i]]))
    plt.show()

    print("Following are the GradCam Heatmaps for those correctly classified images:-")
    fig2, axes2 = plt.subplots(num_row, num_col, figsize=(1.8*num_col,2.5*num_row))
    for i in range(num_images):
        ax = axes2[i//num_col, i%num_col]
        ax.imshow(correct_gradcam[i].numpy().transpose((1,2,0))*std + mean)
        ax.set_title('Label: {}\nPrediction: {}'.format(classes[correct_labels[i]],
                                                        classes[correct_pred[i]]))
    plt.show()

    plt.show()
    my_dpi = 100
    fig1.savefig('S9/S9_correct_classified_images.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    fig2.savefig('S9/S9_correct_classified_gradcam.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
