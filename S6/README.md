# **Overview:**
The objective was to use regularization (techniques such as BatchNorm, GhostBatchNorm, L1 Regularization and L2 Regularization) on top of the best model obtained from the last assignment (S4) and see the effects of various regularization techniques on training and accuracy. <br/>

No. of Parameters in the model (including the BN or GBN layer): **9,304** (of which 9,128 are trainable and 176 are non-trainable parameters)

| **Model Version** | **Maximum Train Accuracy** | **Maximum Test Accuracy** | **Version Descirption**  |
|---:|---:|---:|---:|
| Version 1| 98.83% | 99.01% |  Model with L1 Regularization and BN |
| Version 2| 99.44% | 99.45% |  Model with L2 Regularization and BN |
| Version 3| 98.62% | 98.96% |  Model with L1 & L2 Regularization and BN |
| Version 4| 99.48% | 99.43% |  Model with GBN |
| Version 5| 98.41% | 98.81% |  Model with L1 & L2 Regularization and BN |

*	**Epochs**: Each model was trained for 25 epochs (Dropout and RandomRotation was kept intact for all the models).
*	**Batch Size**: Model versions which had a normal Batchnorm layer (1,2,3) were trained with a training batch size of 128. Model versions which had a Ghost Batchnorm layer (4,5) were trained with a lower training batch size of 32. A lower batch size increases the stochastic variance in GBN while training, thus increasing the regularization effect of GBN.
*	**Learning Rate**: The base learning rate was set to 0.01, which was multiplied by a reducing factor of 0.8 after every 4 steps (using a StepLRScheduler)
* **L1/L2 Regularization Parameters**: Both L1 and L2 Regularization parameters were set to 0.001 (a smaller number can also be used, depends on how much of a regularizing impact is needed) <br/>

Following are the Validation Accuracy and Validation loss graphs generated on training the different model version for 25 epochs :- <br/>

![validation_accuracy_5_versions](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S6/images/val_acc_5_versions.png?raw=true)

![validation_loss_5_versions](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S6/images/val_loss_5_versions.png?raw=true)

From the above graph it can be seen that:-
*	L2 regularization (combined with BN) gave the best result of all the five versions.
*	The combination of L1 and L2 gave worse accuracy than when either of the regularizations was taken one at a time, for both model with BN and GBN. It could also be that the coefficients of regularization was large, and the penalty applied by the combination of regularizers made it harder for the model to reach the most optimum state.
*	GBN alone gave almost the best result, but on combining with L1 and L2, it gave the worst accuracy result, indicating that over-regularization could also lead to adverse effects. GBN, when used alone, was able to reach the 99.0% accuracy within 5 epochs, which the combination of L1+L2+GBN could not reach in 25 epochs.
*	GBN alone and L2 + BN were almost overlapping in their training accuracies obtained, while the other versions were always below these 2 versions for every epoch.
*	There were persistent fluctuations seen in the accuracy values when L1 Regularization is used, while L2 regularization gave a relatively smoother curve. This is due to the fact that the L2 regularization levies a quadratic penalty to large weights, and the objective function still remains a smooth surface. 
*	The fluctuations could also be due to the L1 coefficient of 0.001 being higher than what it should be. We can further experiment with setting a lower L1 coefficient to see if this effect goes away.

Following is the graph of misclassified images (prediciton which were incorrect) by the Version 4 (GBN) only model:- <br/>
![S6_GBN_misclassified_images](https://github.com/AkhilP9182/EVA5---Extensive-Vision-AI/blob/main/S6/images/S6_GBN_misclassified_images.png?raw=true)
