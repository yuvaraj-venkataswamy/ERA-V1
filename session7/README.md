# Session 7 - In-Depth Coding Practice Assignment
## Objective:
- 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.

## Model: 1 Basic structure
### Target
Construct `Basic Structure` of model with the prior knowledge that
- Structures in MNIST images are visible at a level of 5x5, i.e Max pooling must be performed at receptive field of 5x5.
- Go up to 32 channels to reduce total number of parameters.
- No Batch Normalization, Augmentation or Regularization to be used.

### Results
- `Total parameters: 13,584`
- No dropout, Batch normalization or additional regularization used
- Best training accuracy: 99.21% at 15th Epoch
- Best testing accuracy: 99.01% at 14th Epoch

### Analysis
- Both training and testing accuracy gradually increase together up to ~99%
- Overfitting is minimal considering low difference between testing and training accuracies (except in the last 2 epochs).
- Since the base model performs well without batch normalization or other significant additions, we can further reduce the number of parameters to improve model efficiency.

