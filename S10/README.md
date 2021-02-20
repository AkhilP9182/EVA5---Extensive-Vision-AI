# **Overview:**
The objective is to implement the LR finder function (from [this](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py "lr_finder.py") repository) for training a ResNet18 Deep Learning model on CIFAR10 dataset using (mini-batch) stochastic gradient descent.

Our goal is to achieve **88%** accuracy within 50 epochs (model reached a final highest validation accuracy of **%**; it crossed **%** validation accuracy at **epoch **). <br/>
No. of Parameters (same as in original ResNet18 Model): **11,173,962**
___

# **Files:**
*   **resnet.py**     : The main ResNet model file [[source](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-resnet18")] which contains the ResNet18 class imported into the main S10.ipynb file while training/testing
*   **utilities.py**  : Functions for loading datasets (and creating Albumentation transformation), making train/test loaders, plotting images (dataset images along with GradCAM heatmaps) and loss/accuracy, getting mean and standard deviation of data, etc.
*   **train_test.py** : Functions which can be called for training and testing the model.
*   **config.py**     : Parameter presets, static variable values which are called throughout the code - required for running the various blocks in the main S10.ipynb file.
*   **lr_finder.py**     : The .py file which contains the LRfinder class. [[source](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py "lr_finder.py")]
___

# **Plots:**
Following are the Validation Accuracy and Validation loss graphs generated: <br/>
*   Validation and Training Accuracy v/s Epochs: <br/>
![S10_accuracy](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_accuracy.png?raw=true)

*   Validation and Training Loss v/s Epochs: <br/>
![S10_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_loss.png?raw=true)
___

# **Results:**
*   Following are some of the images which were mis-classified by the model:- <br/>
![S10_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_misclassified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S10_misclassified_gradcam](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_misclassified_gradcam.png?raw=true)

*   Following are some of the images which were correctly classified by the model: <br/>
![S10_correct_classified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_correct_classified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S10_correct_classified_gradcam](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_correct_classified_gradcam.png?raw=true)

___

# **References:**
[1] albumentations: https://github.com/albumentations-team/albumentations
[2] resent18:  https://github.com/kuangliu/pytorch-cifar
[3] GradCAM: https://github.com/vickyliin/gradcam_plus_plus-pytorch
[3] LRFinder: https://github.com/davidtvs/pytorch-lr-finder
