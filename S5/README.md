# **Overview:**
Our objective is to obtain create a CNN model which acheived >=99.4% validation accuracy on the MNIST Handwritten digit recognition dataset. The constraints are:-
*	Should have less than **10,000** Parameters in total.
*	Should be trained within **15** epochs.
*	Must acheive >=99.4% (and should maintain it in the last few epochs of training)

|---:|---:|---:|---:|---:|
| **Model Version** | **Maximum Test Accuracy** | **Maximum Train Accuracy** | **No. of Parameters** | **Decription of Change from Previous Version**  |
|---:|---:|---:|---:|---:|
| Version 1| 98.99% | 99.48% | 28,090  |  Base version of CNN with reduced parameters |
| Version 2| 99.27% | 99.65% | 9,304  |  Addition of Batchnorm and GAP Layer, with further parameter reduction |
| Version 3| 99.31% | 99.50% | 9,304  |  Addition of Dropout and LR (step) Scheduler |
| Version 4| 99.44% | 99.39% | 9,304  |  Use of Image Augmentation |

--- 

# **Version 1**
### **Notebook Link**: ([S5_1.ipynb](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/S5_1.ipynb)) <br/>
### **Target**: <br/>
Get the CNN digit recognition pipeline ready, and use kernels in multiples of 10 [10,20,30] for obtaining a basic model architecture with fewer parameters.

### **Results**: <br/>
*	Parameters : **28,090**
*	Best Train Accuracy: 99.48% (Epoch 14)
*	Best Test Accuracy: **98.99%** (Epoch 14)

### **Analysis:** <br/>
*	Model performs well, but takes a while to reach 99% training accuracy (about 9 epochs), which will be addressed using Batch Normalization.
*	There is a gap in the train and test accuracy, indicating a problem of overfitting (can be made more robust using Dropout).
*	The number of kernels at each step can still be reduced to obtain a model having less than 10,000 total parameters (currently there are 2 layers with 5.4k parameters each, and 1 layer with 8k parameters).
*	The convolution layer at the end is for reducing the 5x5 feature maps to 1x1 values as the output of the model, but it is using a large sized kernel to do so. We can replace that with a Global Average Pooling layer in next version and hence, reduce parameters.

![version_1_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/images/S5_1.png?raw=true)

---

# **Version 2**
### **Notebook Link**: ([S5_2.ipynb](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/S5_2.ipynb)) <br/>
### **Target**: <br/>
Reduce the parameters below 10k and add a GAP layer for the same at the end (instead of a conv layer). Also add a BatchNorm layer after every conv layer to achieve faster covnergence.

### **Results**: <br/>
*	Parameters: **9,304**
*	Best Train Accuracy: 99.65% (Epoch 13)
*	Best Test Accuracy: 99.27% (Epoch 12)

### **Analysis:** <br/>
*	Addition of batchnorm allowed the model to reach a higher train accuracy much faster than before (within 5 epochs).
*   The kernel channels are now in multiples of 4 [8,12,16] instead of 10, the number of parameters has been reduced to below 10k (it could be further reduced)
*	There is still an overfitting problem, since there is significant difference between train and test accuracy. The next version will have a Dropout layer after every conv layer to address this issue.
*   Looking at the loss (or accuracy) plot, it's visible that the training process is not gradually converging, but fluctuates a lot towards the middle. We will try to increase convergence midway through the training by using a learning rate schedule (reducing the Learning rate)

![version_2_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/images/S5_2.png?raw=true)

---

# **Version 3**
### **Notebook Link**: ([S5_3.ipynb](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/S5_3.ipynb)) <br/>
### **Target**: <br/>
Reduce overfitting by introducing Dropout after every conv layer, and reduce learning rate as the training progresses to obtain greater accuracy.

### **Results**: <br/>
*	Parameters: **9,304**
*	Best Train Accuracy: 99.50% (Epoch 14)
*	Best Test Accuracy: **99.31%** (Epoch 14)

### **Analysis**: <br/>
*	The introduction of dropout (of p=0.05) after every convolution layer and an LR step schedule certainly helped in curbing the problem of overfitting. Looking at the plots, it is visible that the convergence is much smoother towards the end, and the train and test accuracy difference is also very small for each epoch.
*	We started with an LR of 0.02, and the LR was reduced every 4 epochs, so it was decreased 3 times during training by a factor of 0.5 after every 4 epochs (0.02 to 0.01 to 0.005).
*	Test Accuracy seems to plateau, but that is because of the reduction in learning rate. We will use image augmentation techniques (random roation) to make the model more robust to variation in the validation data and increase test accuracy beyond 99.4%  within 15 epochs, in the next version.

![version_3_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/images/S5_3.png?raw=true)

---

# **Version 4**
### **Notebook Link**: ([S5_4.ipynb](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/S5_4.ipynb)) <br/>
### **Target**: <br/>
*	Increase validation accuracy by introducing variations in the training dataset using Image Augmentation (random rotation by 7 degrees clockwise and anticlockwise)

### **Results**: <br/>
*	Parameters: **9,304**
*	Best Train Accuracy: 99.39% (Epoch 14)
*	Best Test Accuracy: **99.44%** (Epoch 14)

### **Analysis**: <br/>
*	After adding a small random rotation of only 7 degrees, the test accuracy is now consistently higher than the train accuracy for all epochs (which is a slighlty underfitting conditiong), meaning our test data had few images which were transformed in an affine manner w.r.t the training images.
*	The target of 99.4% test accuracy was achieved in Epoch 11, and it was maintained till epoch 15.
*	From the accuracy plots it is visible that after every 4 epochs, the rate of increase in accuracy is decreasing, but the curve is still going up, meaning training for more epochs could possibly increase the validation accuracy.

![version_4_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S5/images/S5_4.png?raw=true)
