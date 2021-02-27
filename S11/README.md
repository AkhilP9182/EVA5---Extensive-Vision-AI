# **Overview:**
The objective is to implement the OneCycleLR scheduler to gradually update the learning rate from a minimum learning rate to a maximum, and reverse till the final epoch. This process is called 'cyclic training' and is used to push networks to find the optimum much faster than using a very small learning rate from the beginning itself.

Below is an image of how learning rate changes in a cyclic LR updater schedule: <br/>
![S11_cyclicLRschedule.png](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_cyclicLRschedule.png?raw=true)

We will be using the OneCycleLR() function from the torch.optim.lr_scheduler() class to obtain an LR scheduler which reaches its peak at the 5th epoch. Our goal is to achieve **90%** accuracy within **ONLY 24 epochs** where the learning rate reaches its max at epoch 5 (model reached a final highest validation accuracy of **91.34%**; it crossed **88%** validation accuracy at **epoch 28**).<br/>

*   No. of Parameters (same as in original ResNet18 Model): **6,573,130**
*   The only transforms applied on the training dataset are [Padding > RandomCrop > HorizontalFlip > Cutout] as shown [here](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/0b0750c6c16a9c500f646257427f8607dcc7f4ec/S11/utilities.py#L22).
___

# **Files:**
*   **utilities.py**  : Functions for loading datasets (and creating Albumentation transformation), making train/test loaders, plotting images (dataset images along with GradCAM heatmaps) and loss/accuracy, getting mean and standard deviation of data, etc.
*   **train_test.py** : Functions which can be called for training and testing the model.
*   **config.py**     : Parameter presets, static variable values which are called throughout the code - required for running the various blocks in the main `S11.ipynb` file.
*   **lr_finder.py**     : The file which contains the LRfinder class. [[source](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py "lr_finder.py")]
*   **model_s11.py** : This contains the model_s11() class which contains the architecture for the model used in this code. This model uses 2 resnet basicblocks within it.
___

# **Plots:**
Following are the Validation Accuracy and Validation loss graphs generated: <br/>
*   Validation and Training Accuracy v/s Epochs: <br/>
![S11_accuracy](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_accuracy.png?raw=true)

*   Validation and Training Loss v/s Epochs: <br/>
![S11_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_loss.png?raw=true)
___

# **Results:**
*   Following are some of the images which were mis-classified by the model:- <br/>
![S11_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_misclassified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S11_misclassified_gradcam](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_misclassified_gradcam.png?raw=true)

*   Following are some of the images which were correctly classified by the model: <br/>
![S11_correct_classified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_correct_classified_images.png?raw=true)
... and their corresponding GradCAM heatmaps:-
![S11_correct_classified_gradcam](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S11/images/S11_correct_classified_gradcam.png?raw=true)

___

# **References:**
[1] albumentations: https://github.com/albumentations-team/albumentations

[2] GradCAM: https://github.com/vickyliin/gradcam_plus_plus-pytorch

[3] LRFinder: https://github.com/davidtvs/pytorch-lr-finder

[4] CyclicLR: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
