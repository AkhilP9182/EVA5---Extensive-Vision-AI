# **Overview:**
The objective was to train a ResNet18 model (default ResNet code imported from [this](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-ResNet18") github repository has been used to keep the number of parameters fixed) for classifying the objects in the torchvision.CIFAR10 dataset.

Our goal is to achieve 85% accuracy, with no limit on epochs (model reached a final validation accuracy of **91.57%**; it crossed **86.26%** test accuracy at **epoch 9**)

# **Model:**
Following is the architecure of ResNet18 [[source](https://duchesnay.github.io/pystatsml/_images/resnet18.png "cifar-resnet18")]:
<p align="center">
  <img src="https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/images/resnet18_architecture.png?raw=true">
</p>

*   No. of Parameters (same as in original ResNet18): **11,173,962**

# **Files:**
*   **resnet.py**     : The main ResNet model file [[source](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-resnet18")] which contains the ResNet18 class imported into the main S8.ipynb file while training/testing
*   **utilities.py**  : Functions for loading datasets, making train/test loaders, plotting images and loss/accuracy, getting mean and standard deviation of data, etc.
*   **train_test.py** : Functions which can be called for training and testing the model.
*   **config.py**     : Parameter presets, static variable values which are called throughout the code - required for running the various blocks in the main S8.ipynb file.

# **Plots:**
Following are the Validation Accuracy and Validation loss graphs generated: <br/>
*   Validation and Training Accuracy v/s Epochs: <br/>
![S8_accuracy](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/images/S8_accuracy.png?raw=true)

*   Validation and Training Loss v/s Epochs: <br/>
![S8_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/images/S8_loss.png?raw=true)

# **Results:**
*   Following are some of the images which were mis-classified by the model: <br/>
![S8_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/images/S8_misclassified_images.png?raw=true)

*   Following are some of the images which were correctly classified by the model: <br/>
![S8_correct_classified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S8/images/S8_correct_classified_images.png?raw=true)

# **References:**
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    
[2] https://github.com/kuangliu/pytorch-cifar
