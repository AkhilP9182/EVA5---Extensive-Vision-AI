# **Overview:**
The objective is to implement the OneCycleLR to gradually update the learning rate from a minimum to a maximum throughout the training process.

Our goal is to achieve **90%** accuracy within **ONLY 24 epochs** where the learning rate reaches its max at epoch 5 (model reached a final highest validation accuracy of **91.34%**; it crossed **88%** validation accuracy at **epoch 28**).<br/>

No. of Parameters (same as in original ResNet18 Model): **11,173,962**
___

# **Files:**
*   **resnet.py**     : The main ResNet model file [[source](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py "pytorch-cifar-resnet18")] which contains the ResNet18 class imported into the main `S10.ipynb` file while training/testing
*   **utilities.py**  : Functions for loading datasets (and creating Albumentation transformation), making train/test loaders, plotting images (dataset images along with GradCAM heatmaps) and loss/accuracy, getting mean and standard deviation of data, etc.
*   **train_test.py** : Functions which can be called for training and testing the model.
*   **config.py**     : Parameter presets, static variable values which are called throughout the code - required for running the various blocks in the main `S10.ipynb` file.
*   **lr_finder.py**     : The file which contains the LRfinder class. [[source](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py "lr_finder.py")]
___

# **LR finder:**
*   The `lr_finder.range_test()` function is a a function for obtaining the most optimal learning rate for a given loss criterion, optimization method and model. 
*   During a pre-training run, the learning rate is exponentially increased between two boundaries (here, from `1e-7` to `1.0`). 
*   The low initial learning rate allows the network to start converging and as the learning rate is increased, it would eventually become large enough upon when the network diverges.
*   Usually, a good static learning rate can be found **half-way on the descending loss curve**. In the plot below that would be between lr=`1E10-4` and lr=`1E10-2` (which is also the range within which we got our suggested lr=`2.12E-03`)

![S11_LR_finder](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/LRFinder.png?raw=true)
___

# **Plots:**
Following are the Validation Accuracy and Validation loss graphs generated (the training loss curve diverges from the test loss curve after a while, and it reaches almost 100%, indiciating an overfit. This could be reduced by increasing the L2 Regularization parameter while training): <br/>
*   Validation and Training Accuracy v/s Epochs: <br/>
![S11_accuracy](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_accuracy.png?raw=true)

*   Validation and Training Loss v/s Epochs: <br/>
![S11_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_loss.png?raw=true)
___

# **Results:**
*   Following are some of the images which were mis-classified by the model:- <br/>
![S11_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_misclassified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S11_misclassified_gradcam](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S10/images/S10_misclassified_gradcam.png?raw=true)

*   Following are some of the images which were correctly classified by the model: <br/>
![S11_correct_classified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_correct_classified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S11_correct_classified_gradcam](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_correct_classified_gradcam.png?raw=true)

___

# **References:**
[1] albumentations: https://github.com/albumentations-team/albumentations

[2] resent18:  https://github.com/kuangliu/pytorch-cifar

[3] GradCAM: https://github.com/vickyliin/gradcam_plus_plus-pytorch

[4] LRFinder: https://github.com/davidtvs/pytorch-lr-finder
