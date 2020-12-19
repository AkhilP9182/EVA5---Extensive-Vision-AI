# **Objective:**
Our objective is to build a CNN model with less than 20,000 total parameters which is able to acheive a validation accuracy of 99.4% on the MNIST test data. We will go through 5 versions of CNN models and see the effect of intorducing a new technique/method at each step.

## **Version 0 (Base Case):**
This is the first version of the CNN model without any additional changes. Following is the architecture for reference:-
Total No. of Parameters: 13,680
|--------------------------|----------------------------|----------|
|        Layer (type)      |         Output Shape       |  Param # |
|--------------------------|----------------------------|----------|
|           Conv2d-1       |    [-1, 10, 28, 28]        |     100  |
|            Conv2d-2      |     [-1, 20, 28, 28]       |    1,820 |
|         MaxPool2d-3      |     [-1, 20, 14, 14]       |        0 |
|            Conv2d-4      |     [-1, 20, 14, 14]       |    3,620 |
|            Conv2d-5      |     [-1, 30, 14, 14]       |    5,430 |
|         MaxPool2d-6      |       [-1, 30, 7, 7]       |        0 |
|            Conv2d-7      |       [-1, 10, 5, 5]       |    2,710 |
|         AvgPool2d-8      |       [-1, 10, 1, 1]       |        0 |

Maximum Test Accuracy Acheived : **98.8%** (at Epoch 19)
Corresponding Train Accuracy   : **99.1%** (at Epoch 19)

Observations: We see that the maxpool layer is very close to the final output Global Average pooling layer. There are also some architecural changes which can be implemented (change in the number of kernels, positions of layers, etc.). The next version addresses these issues.

## **Case 1 (Architecture Change):**

## **Case 2 (Addition of BatchNorm after every convolution layer):**

## **Case 3 (BatchNorm + 1 Dropout Layer):**

## **Case 4 (BatchNorm + 1 Dropout Layer + Learning Rate Update):**

## **Case 4 (BatchNorm + 1 Dropout Layer + Learning Rate Update + Data Augmentation):**
