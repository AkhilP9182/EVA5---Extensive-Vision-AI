# **Overview:**
The objective was to use data augmentation techniques in the Albumentation library and train the ResNet18 model from assignment S8 (default ResNet code imported from [this](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-ResNet18") repository is used) for classifying the objects in the torchvision.CIFAR10 dataset. Then, using the method known as GradCAM (Gradient weighted Class Activation Mapping) we will obtain the heatmaps for the layer **x** of the resent 18 model for both correctly classified images and misclassified images, to see what the model is learning from the given images.

Our goal is to achieve **87%** accuracy, with no limit on epochs (model reached a final validation accuracy of **91.57%**; it crossed **86.26%** test accuracy at **epoch 9**). <br/>
No. of Parameters (same as in original ResNet18): **11,173,962**
___

# **Files:**
*   **resnet.py**     : The main ResNet model file [[source](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-resnet18")] which contains the ResNet18 class imported into the main S9.ipynb file while training/testing
*   **utilities.py**  : Functions for loading datasets (and creating Albumentation transformation), making train/test loaders, plotting images (dataset images along with GradCAM heatmaps) and loss/accuracy, getting mean and standard deviation of data, etc.
*   **train_test.py** : Functions which can be called for training and testing the model.
*   **config.py**     : Parameter presets, static variable values which are called throughout the code - required for running the various blocks in the main S8.ipynb file.

___

# **Albumentations used:**
*   Random HorizontalFlip
*   RandomRotate
*   RandomRandomBrightnessContrast
*   Cutout
*   Normalize

___

# **Plots:**
Following are the Validation Accuracy and Validation loss graphs generated: <br/>
*   Validation and Training Accuracy v/s Epochs: <br/>
![S9_accuracy](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S9/images/S9_accuracy.png?raw=true)

*   Validation and Training Loss v/s Epochs: <br/>
![S9_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S9/images/S9_loss.png?raw=true)
___

# **Results:**
*   Following are some of the images which were mis-classified by the model:- <br/>
![S9_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S9/images/S9_misclassified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S9_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S9/images/S9_misclassified_gradcam.png?raw=true)

*   Following are some of the images which were correctly classified by the model: <br/>
![S9_correct_classified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S9/images/S9_correct_classified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S9_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S9/images/S9_misclassified_gradcam.png?raw=true)

___

# **References:**
[1] albumentations: https://github.com/albumentations-team/albumentations
[2] resent18:  https://github.com/kuangliu/pytorch-cifar
[3] GradCAM: https://github.com/vickyliin/gradcam_plus_plus-pytorch
