# **Overview:**
Our objective is to obtain create a CNN model which acheived >=99.4% validation accuracy on the MNIST Handwritten digit recognition dataset. The constraints are:-
*	Should have less than **10,000** Parameters in total.
*	Should be trained within **15** epochs.
*	Must acheive >=99.4% (and should maintain it in the last few epochs of training)

---

| **Model Version** | **Maximum Test Accuracy** | **Maximum Test Accuracy** | **Epoch No.** | **No. of Parameters** | **Decription of Change from Previous Version**  |
|---:|---:|---:|---:|---:|---:|
| Version 1| 98.99% | 99.48% | 18 | 28,090  |  Base version of CNN |
| Version 2| 99.27% | 99.65% | 11 | 9,304  |  Addition of Batchnorm and GAP Layer |
| Version 3| 99.31% | 99.50% | 12 | 9,304  |  Addition of Dropout and LR Scheduler |
| Version 4| 99.51% || 11 | 9,304  |  Use of Image Augmentation |

# **Version 1**
### **Target**: <br/>
Get the entire prediciton pipeline ready, and use kernels in multiples of 10 [10,20,30] for obtaining a basic model architecture with fewer parameters.

### **Results**: <br/>
*	Parameters : **28,090**
*	Best Train Accuracy: 99.48 % (Epoch 14)
*	Best Test Accuracy: 98.99 % (Epoch 14)
