# **Objective:**
Our objective is to build a CNN model with less than 20,000 total parameters which is able to acheive a validation accuracy of 99.4% on the MNIST test data. We will go through 5 versions of CNN models and see the effect of intorducing a new technique/method at each step. The final test accuracy obtained is : **99.52%** with **17,730** total parameters  at Version 5.

Batch Size = 128 (training) and 1024 (testing) across all versions. LR = 0.01 till version 3.

### **Results on the final version (Version 5):**

<center>
![prediction_vs_groundTruth](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S4/prediction_vs_groundTruth.png?raw=true)
</center>

Here is a summary table:- <br />
<center>
| **Model Version** | **Maximum Validation Accuracy** | **Epoch No.** | **No. of Parameters** | **Decription of Change from Previous Version**  |
|---:|---:|---:|---:|---:|
| Version 0| 98.80% | 19 | 13,680  |  None (Base Version) |
| Version 1| 99.20% | 18 | 17,490  |  Network Architechure Change |
| Version 2| 99.30% | 11 | 17,730  |  Addition of Batchnorm |
| Version 3| 99.47% | 12 | 17,730  |  Addition of Dropout |
| Version 4| 99.51% | 11 | 17,730  |  Addition of LR Scheduler |
| Version 5| 99.52% | 16 | 17,730  |  Addition of Data Augmentation |
</center>

## **Version 0 (Base Case):**

This is the first version of the CNN model without any additional changes. Following is the architecture for reference:- <br />
Total No. of Parameters: **13,680**

<center>
|        **Layer (type)**     |         **Output Shape**       |  **Parameter**   |
|-------------------------:|----------------------------:|----------:|
|           Conv2d-1      |    [-1, 10, 28, 28]        |     100  |
|            Conv2d-2     |     [-1, 20, 28, 28]       |    1,820 |
|         MaxPool2d-3     |     [-1, 20, 14, 14]       |        0 |
|            Conv2d-4     |     [-1, 20, 14, 14]       |    3,620 |
|            Conv2d-5     |     [-1, 30, 14, 14]       |    5,430 |
|         MaxPool2d-6     |       [-1, 30, 7, 7]       |        0 |
|            Conv2d-7     |       [-1, 10, 5, 5]       |    2,710 |
|         AvgPool2d-8     |       [-1, 10, 1, 1]       |        0 |
</center>

Maximum Test Accuracy Acheived : **98.8%** (at Epoch 19) <br />
Corresponding Train Accuracy   : **99.1%** (at Epoch 19)

**Observations**: We see that the maxpool layer is very close to the final output Global Average pooling layer. There are also some architecural changes which can be implemented (change in the number of kernels, positions of layers, etc.). The next version addresses these issues.


## **Version 1 (Architecture Change):**

The number of kernels in each conv layers was changed (so that greater number of kernels are used as we go deeper) and the kernels are in multiples of 10. The final maxpool layer is mooved further up the netowrk (away from the output layer). <br />
Total No. of Parameters: **17,490**

<center>
|       **Layer (type)**      |         **Output Shape**       |   **Parameter**   |
|-------------------------:|----------------------------:|----------:|
|           Conv2d-1      |     [-1, 10, 28, 28]       |     90   |
|           Conv2d-2      |     [-1, 20, 28, 28]       |   1,800  |
|           **Conv2d-3**      |     [-1, 30, 28, 28]       |    5,400 |
|        MaxPool2d-4      |     [-1, 30, 14, 14]       |       0  |
|           Conv2d-5      |     [-1, 10, 14, 14]       |     300  |
|           Conv2d-6      |     [-1, 20, 12, 12]       |   1,800  |
|        **MaxPool2d-7**      |       [-1, 20, 6, 6]       |     0    |
|           Conv2d-8      |       [-1, 30, 4, 4]       | 5,400    |
|           Conv2d-9      |       [-1, 10, 2, 2]       | 2,700    |
|       AvgPool2d-10      |       [-1, 10, 1, 1]       |     0    |
</center>
  
Maximum Test Accuracy Acheived : **99.2%** (at Epoch 18) <br />
Corresponding Train Accuracy   : **99.5%** (at Epoch 18)

**Observations**: There is a marginal improvement in test accuracy with the architecture change, but train accuracy has improved considerably more. We also observe that the max accuracy is reached at nearly the last epoch, meaning the convergence to optima is quite slow. Hence, we may need to whiten the outputs at each step, i.e, use batchnorm, after each convolution layer in the next step.


## **Version 2 (Addition of BatchNorm after every convolution layer):**

A batchnorm layer was added after every convolutional layer to reduce the covariate shift in the weights of the layer, and help the model in converging faster. <br />
Total No. of Parameters: **17,730**

<center>
|       **Layer (type)**      |         **Output Shape**       |   **Parameter**   |
|-------------------------:|----------------------------:|----------:|
|            Conv2d-1     |      [-1, 10, 28, 28]      |        90|
|      **BatchNorm2d-2**     |      [-1, 10, 28, 28]      |        20|
|            Conv2d-3     |      [-1, 20, 28, 28]      |     1,800|
|      **BatchNorm2d-4**     |      [-1, 20, 28, 28]      |        40|
|            Conv2d-5     |      [-1, 30, 28, 28]      |     5,400|
|      **BatchNorm2d-6**     |      [-1, 30, 28, 28]      |        60|
|         MaxPool2d-7     |      [-1, 30, 14, 14]      |         0|
|            Conv2d-8     |      [-1, 10, 14, 14]      |       300|
|      **BatchNorm2d-9**     |      [-1, 10, 14, 14]      |        20|
|           Conv2d-10     |      [-1, 20, 12, 12]      |     1,800|
|      **BatchNorm2d-11**     |      [-1, 20, 12, 12]      |        40|
|        MaxPool2d-12     |        [-1, 20, 6, 6]      |         0|
|           Conv2d-13     |        [-1, 30, 4, 4]      |     5,400|
|      **BatchNorm2d-14**     |        [-1, 30, 4, 4]      |        60|
|           Conv2d-15     |        [-1, 10, 2, 2]      |     2,700|
|        AvgPool2d-16     |        [-1, 10, 1, 1]      |         0|
</center>
 
Maximum Test Accuracy Acheived : **99.9%** (at Epoch 11) <br />
Corresponding Train Accuracy   : **99.3%** (at Epoch 11)

**Observations**: With the introduction of batchnorm, the test accuracy reached 98.7% within 1 epoch. We were able to reach an accuracy of 99.3 for validation data, but also observed that at many instances the the accuracy on training data was nearly 100%, which points to the problem of overfitting. To combat that issue, we will see the use of droput in the next layer.


## **Version 3 (BatchNorm + 1 Dropout Layer):**

A droput layer was added just before the 1st maxpool layer in the network, which would randomly dropuout 5% of the input channels in a forward pass. This helped in acheiving our goal of obtaining greater than 99.4% accuracy in the validation set. We will further explore the effect of learning rate and data augmentation. <br />
Total No. of Parameters: **17,730**

<center>
|       **Layer (type)**      |         **Output Shape**       |   **Parameter**   |
|-------------------------:|----------------------------:|----------:|
|           Conv2d-1      |    [-1, 10, 28, 28]        |     90   |
|      BatchNorm2d-2      |    [-1, 10, 28, 28]        |     20   |
|           Conv2d-3      |    [-1, 20, 28, 28]        |  1,800   |
|      BatchNorm2d-4      |    [-1, 20, 28, 28]        |     40   |
|           Conv2d-5      |    [-1, 30, 28, 28]        |  5,400   |
|      BatchNorm2d-6      |    [-1, 30, 28, 28]        |     60   |
|        **Dropout2d-7**      |    [-1, 30, 28, 28]        |      0   |
|        MaxPool2d-8      |    [-1, 30, 14, 14]        |      0   |
|           Conv2d-9      |    [-1, 10, 14, 14]        |    300   |
|     BatchNorm2d-10      |    [-1, 10, 14, 14]        |     20   |
|          Conv2d-11      |    [-1, 20, 12, 12]        |  1,800   |
|     BatchNorm2d-12      |    [-1, 20, 12, 12]        |     40   |
|       MaxPool2d-13      |      [-1, 20, 6, 6]        |      0   |
|          Conv2d-14      |      [-1, 30, 4, 4]        |  5,400   |
|     BatchNorm2d-15      |      [-1, 30, 4, 4]        |     60   |
|          Conv2d-16      |      [-1, 10, 2, 2]        |  2,700   |
|       AvgPool2d-17      |      [-1, 10, 1, 1]        |      0   |
</center>

Maximum Test Accuracy Acheived : **99.47%** (at Epoch 12) <br />
Corresponding Train Accuracy   : **99.83%** (at Epoch 12)

**Observations**: Introducing a small dropout of 5% after only 1 layer enabled us to cross the threshold of 99.4% accuracy on the validation set in one of the test runs. We also observed that the difference between train and test accuracy is lower now, meaning the model has become more robust.

## **Case 4 (BatchNorm + 1 Dropout Layer + Learning Rate Update):**

The next change we will introduce is updating the learning rate (reducing LR by a factor of 0.2) after 10 epoch of training. This will help to reduce the loss even further on the training set, and increase the accuracy on validation set. We will use `optim.lr_scheduler.StepLR()` for updating the learning rate.

LR = `0.01` till epoch 10, then `0.008` till epoch 19 (last epoch)}

Total No. of Parameters: **17,730** <br />
*[Network architecure the same as Version 3]*

Maximum Test Accuracy Acheived : **99.51%** (at Epoch 11) <br />
Corresponding Train Accuracy   : **99.92%** (at Epoch 11)

**Observations**: We have now reached a validation accuracy of 99.51 %, the difference between train and test accuracy is still relatively large. In order to introudce more variation in the training data and make the model more robust, we will introduce image augmentations in the next version.

## **Case 5(BatchNorm + 1 Dropout Layer + Learning Rate Update + Data Augmentation):**

We will now apply a number of affine transformation to the training data (`torch.transforms.RandomAffine()`) in order to introudce greater variance in training data. We will be Rotating the image in the range `(-20°,+20°)`, translate by `(0.1) x dimension length` pixels in the 4 cardinal directions, and random scaling between `(0.95,1.05)` times the image size.

Total No. of Parameters: **17,730** <br />
*[Network architecure the same as Version 3]*

Maximum Test Accuracy Acheived : **99.52%** (at Epoch 16) <br />
Corresponding Train Accuracy   : **99.08%** (at Epoch 16)

**Observations**: We are now seeing a a completely opposite trend in terms of loss and accuracy when compared to the other models. The test accuracy is actually higher (for all epochs) than the train accuracy now. We have, thus, addressed the issue of overfitting, and acheived a maximum accuracy of **99.52%** on validation dataset.

Below are the loss and accuracy plots for the final version 5 of the network.
<center>
![case_5_loss](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S4/case_5_loss.png?raw=true)

![case_5_acc](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S4/case_5_acc.png?raw=true 'Version 5 Accuracy')
</center>
