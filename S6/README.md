# **Overview:**
The objective was to use regularization (techniques such as BatchNorm, GhostBatchNorm, L1 Regularization and L2 Regularization) on top of the best model obtained from the last assignment (S4) and see the effects of various regularization techniques on training and accuracy.
No. of Parameters in the model (including the BN or GBN layer): **9,304** (of which 9,128 are trainable and 176 are non-trainable parameters)

| **Model Version** | **Maximum Train Accuracy** | **Maximum Test Accuracy** | **Version Descirption**  |
|---:|---:|---:|---:|
| Version 1| 98.83% | 99.01% |  Model with L1 Regularization and BN |
| Version 2| 99.44% | 99.45% |  Model with L2 Regularization and BN |
| Version 3| 98.62% | 98.96% |  Model with L1 & L2 Regularization and BN |
| Version 4| 99.48% | 99.43% |  Model with GBN |
| Version 5| 98.41% | 98.81% |  Model with L1 & L2 Regularization and BN |

