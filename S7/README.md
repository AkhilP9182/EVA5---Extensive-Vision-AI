# **Overview:**
The objective was to train a CNN model for predicting the classes of objects in the torchvision.CIFAR10 dataset.
**Constrains:**
*  Final Receptive field at output should be greater than 44 (current model has 64)
*  One of the model's layers must be a Depthwise Separable Convolution Layer(Used 2)
*  One of the model's layers must be a Dilated Convolution Layer (Used 1)
*  Use GAP at the final layer (**no Linear layer** was used)
*  Total parameters should be less than 1 Million (current model has **650,464** parameters)
*  Must achieve 80% accuracy, no limit on epochs (model reached **82.03%** accuracy, crossed 80% test accuracy at epoch 15)

The required files are cloned via Github for running the model. Snippet is available in the S7.ipynb file. Following are the names and descriptions of the files imported:- <br/>
*   model.py : The main model file which contains the Net() class which can be imported while running.
*   layers.py : This file contains custom model layers (Batchnorm, GBM, etc.) which can be used in the model.
*   utilities.py : Functions for loading datasets, making train/test loaders, plotting images and loss/accuracy, getting mean and standard deviation of data, etc.
*   train_test.py : Functions which can be called for training and testing the model.
*   config.py: Parameter presets, which are called throughout the code - required for running the various blocks in the main S7.ipynb file.


Following are the Validation Accuracy and Validation loss graphs generated: <br/>
*   Validation and Training Accuracy v/s Epochs: <br/>
![validation_accuracy_7](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S6/images/acc_7.png?raw=true)

*   Validation and Training Accuracy v/s Epochs: <br/>
![validation_loss_7](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S6/images/val_loss_7.png?raw=true)


Following are some of the images which were mis-classified by the model: <br/>
![mis_classified_7](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S6/images/S7_misclassified_images.png?raw=true)
