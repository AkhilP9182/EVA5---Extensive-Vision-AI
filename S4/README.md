# **Objective:**
Our objective is to build a CNN model with less than 20,000 total parameters which is able to acheive a validation accuracy of 99.4% on the MNIST test data. We will go through 5 versions of CNN models and see the effect of intorducing a new technique/method at each step. The final test accuracy obtained is : **99.52%** when we reach Version 5.


## **Version 0 (Base Case):**

This is the first version of the CNN model without any additional changes. Following is the architecture for reference:- <br />
Total No. of Parameters: **13,680**

|        Layer (type)     |         Output Shape       |  Param  |
|-------------------------|----------------------------|----------|
|           Conv2d-1      |    [-1, 10, 28, 28]        |     100  |
|            Conv2d-2     |     [-1, 20, 28, 28]       |    1,820 |
|         MaxPool2d-3     |     [-1, 20, 14, 14]       |        0 |
|            Conv2d-4     |     [-1, 20, 14, 14]       |    3,620 |
|            Conv2d-5     |     [-1, 30, 14, 14]       |    5,430 |
|         MaxPool2d-6     |       [-1, 30, 7, 7]       |        0 |
|            Conv2d-7     |       [-1, 10, 5, 5]       |    2,710 |
|         AvgPool2d-8     |       [-1, 10, 1, 1]       |        0 |

Maximum Test Accuracy Acheived : **98.8%** (at Epoch 19) <br />
Corresponding Train Accuracy   : **99.1%** (at Epoch 19)

**Observations**: We see that the maxpool layer is very close to the final output Global Average pooling layer. There are also some architecural changes which can be implemented (change in the number of kernels, positions of layers, etc.). The next version addresses these issues.


## **Version 1 (Architecture Change):**

The number of kernels in each conv layers was changed (so that greater number of kernels are used as we go deeper) and the kernels are in multiples of 10. The final maxpool layer is mooved further up the netowrk (away from the output layer). <br />
Total No. of Parameters: 17,490

|-------------------------|----------------------------|----------| 
|       Layer (type)      |         Output Shape       |   Param #|
|-------------------------|----------------------------|----------|
|           Conv2d-1      |     [-1, 10, 28, 28]       |     90   |
|           Conv2d-2      |     [-1, 20, 28, 28]       |   1,800  |
|           Conv2d-3      |     [-1, 30, 28, 28]       |    5,400 |
|        MaxPool2d-4      |     [-1, 30, 14, 14]       |       0  |
|           Conv2d-5      |     [-1, 10, 14, 14]       |     300  |
|           Conv2d-6      |     [-1, 20, 12, 12]       |   1,800  |
|        MaxPool2d-7      |       [-1, 20, 6, 6]       |     0    |
|           Conv2d-8      |       [-1, 30, 4, 4]       | 5,400    |
|           Conv2d-9      |       [-1, 10, 2, 2]       | 2,700    |
|       AvgPool2d-10      |       [-1, 10, 1, 1]       |     0    |

Maximum Test Accuracy Acheived : **99.5%** (at Epoch 18) <br />
Corresponding Train Accuracy   : **99.2%** (at Epoch 18)

**Observations**: There is a marginal improvement in test accuracy with the architecture change, but train accuracy has improved considerably more. We also observe that the max accuracy is reached at nearly the last epoch, meaning the convergence to optima is quite slow. Hence, we may need to whiten the outputs at each step, i.e, use batchnorm, after each convolution layer in the next step.


## **Version 2 (Addition of BatchNorm after every convolution layer):**

A batchnorm layer was added after every convolutional layer to reduce the covariate shift in the weights of the layer, and help the model in converging faster. <br />
Total No. of Parameters: **17,730**

|       Layer (type)      |         Output Shape       |   Param #|
|-------------------------|----------------------------|----------|
|            Conv2d-1     |      [-1, 10, 28, 28]      |        90|
|       BatchNorm2d-2     |      [-1, 10, 28, 28]      |        20|
|            Conv2d-3     |      [-1, 20, 28, 28]      |     1,800|
|       BatchNorm2d-4     |      [-1, 20, 28, 28]      |        40|
|            Conv2d-5     |      [-1, 30, 28, 28]      |     5,400|
|       BatchNorm2d-6     |      [-1, 30, 28, 28]      |        60|
|         MaxPool2d-7     |      [-1, 30, 14, 14]      |         0|
|            Conv2d-8     |      [-1, 10, 14, 14]      |       300|
|       BatchNorm2d-9     |      [-1, 10, 14, 14]      |        20|
|           Conv2d-10     |      [-1, 20, 12, 12]      |     1,800|
|      BatchNorm2d-11     |      [-1, 20, 12, 12]      |        40|
|        MaxPool2d-12     |        [-1, 20, 6, 6]      |         0|
|           Conv2d-13     |        [-1, 30, 4, 4]      |     5,400|
|      BatchNorm2d-14     |        [-1, 30, 4, 4]      |        60|
|           Conv2d-15     |        [-1, 10, 2, 2]      |     2,700|
|        AvgPool2d-16     |        [-1, 10, 1, 1]      |         0|

Maximum Test Accuracy Acheived : **99.9%** (at Epoch 11) <br />
Corresponding Train Accuracy   : **99.3%** (at Epoch 11)

**Observations**: With the introduction of batchnorm, the test accuracy reached 98.7% within 1 epoch. We were able to reach an accuracy of 99.3 for validation data, but also observed that at many instances the the accuracy on training data was nearly 100%, which points to the problem of overfitting. To combat that issue, we will see the use of droput in the next layer.


## **Case 3 (BatchNorm + 1 Dropout Layer):**



## **Case 4 (BatchNorm + 1 Dropout Layer + Learning Rate Update):**

## **Case 4 (BatchNorm + 1 Dropout Layer + Learning Rate Update + Data Augmentation):**
