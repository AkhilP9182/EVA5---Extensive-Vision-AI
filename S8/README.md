# **Overview:**
The objective was to train a Resnet18 model (default Resent code imported from [this](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-resnet18") github repositroy has been used to keep the number of parameters fixed) for classifying the objects in the torchvision.CIFAR10 dataset.

Our goal is to achieve 85% accuracy, with no limit on epochs (model reached **%** accuracy, crossed 85.78% test accuracy at epoch 7)

Following is the architecure of ResNet18 [source](https://duchesnay.github.io/pystatsml/_images/resnet18.png "cifar-resnet18"):
![resnet18_architecture.png](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/resnet18_architecture.png?raw=true)

*   No. of Parameters (original ResNet18): **11,173,962**

*   resnet.py     : The main resnet model file [source](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-resnet18") which contains the ResNet18 class imported into the main S8.ipynb file while training/testing
*   utilities.py  : Functions for loading datasets, making train/test loaders, plotting images and loss/accuracy, getting mean and standard deviation of data, etc.
*   train_test.py : Functions which can be called for training and testing the model.
*   config.py     : Parameter presets, static variable values which are called throughout the code - required for running the various blocks in the main S8.ipynb file.

Following are the Validation Accuracy and Validation loss graphs generated: <br/>
*   Validation and Training Accuracy v/s Epochs: <br/>
![S8_accuracy](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/S8_accuracy.png?raw=true)

*   Validation and Training Loss v/s Epochs: <br/>
![S8_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/S8_loss.png?raw=true)

*   Following are some of the images which were mis-classified by the model: <br/>
![S8_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/S8_misclassified_images.png?raw=true)

*   Following are some of the images which were mis-classified by the model: <br/>
![S8_correct_classified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/S8_correct_classified_images.png?raw=true)
