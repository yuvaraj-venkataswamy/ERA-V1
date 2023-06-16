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

## Model: 2 Parameter Reduction
### Target
- Reduce total number of parameters to below 10k by reducing number of channels in expansions after image input and post max-pooling.
- Maintain max-pooling step at RF of 5x5
- Number of channels for conv layers to be reduced
- Additional parameter expansion to be added post GAP layer (will allow for additional learning)
### Results
- `Total Paratemers: 7,400`
- Best training accuracy: 98.79% at 15th Epoch
- B#est testing accuracy: 98.72% at 14th Epoch
## Analysis
- Model learns slowly, and slight underfitting is present.
- Accuracy convergence can be made faster with the addition of Batch Normalization, which we will do in the next step.

## Model: 3 Batch Normalization
### Target
- Add `Batch Normalization` to improve rate of convergence and overall accuracy after 15 epochs.
### Results
- Total parameters: 7,624
- Best Training Accuracy: 99.32% at 15th Epoch
- Best Testing Accuracy: 99.28% at 14th Epoch
### Analysis
- Using of Batch Normalization the accuracy increase through the epochs.
- Very slight overfitting is observed with a difference of ~0.05 between training and testing accuracies in the last 3 epochs.

## Model: 4 Adding Dropout
### Target
- Add dropout to all convolution layers (except last one) to reduce overfitting. Reduction in accuracy is expected.
### Results
- `Total parameters: 7,624`
- Best Training Accuracy: 98.93% - 14th Epoch
- Best Testing Accuracy: 99.22% - 13th Epoch
### Analysis
- Overfitting is no longer present, and slight underfitting is present.
- Overall accuracy has reduced which is expected considering dropout has been added.

## Model: 5 Data Augmentations
### Target
- Add data augmentations to improve overall testing accuracy.
- Based on analysis of wrong predictions in the last step, random rotations, 
- Random affine transormations and Color jitter augmentations have been added.
### Results
- `Total parameters:` 7,624
- Best Training Accuracy: 97.63% at 15th Epoch
- Best Testing Accuracy: 99.34% at 13th Epoch
### Analysis
- Underfitting is clearly evident, training accuracy is consistently lower than testing accuracy.

## Model: 6 Learning Rate (LR) Scheduling
### Target
- Add learning rate (LR) scheduling to improve overall accuracy with consistant of 99.4% for atleast 2 epoch.
### Results
- `Total parameters: 7,624`
- Best Training Accuracy: 97.60% - 15th Epoch
- Best Testing Accuracy: 99.48% - 11th Epoch
### Analysis
- Lambda scheduling with lambda set to 0.65^epoch helps us achieve our target of 99.4% testing accuracy starting from the 11th Epoch.

## Final Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
         MaxPool2d-9           [-1, 16, 12, 12]               0
           Conv2d-10            [-1, 8, 12, 12]             128
           Conv2d-11           [-1, 16, 10, 10]           1,152
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 24, 8, 8]           3,456
             ReLU-16             [-1, 24, 8, 8]               0
      BatchNorm2d-17             [-1, 24, 8, 8]              48
          Dropout-18             [-1, 24, 8, 8]               0
        AvgPool2d-19             [-1, 24, 1, 1]               0
           Conv2d-20             [-1, 32, 1, 1]             768
             ReLU-21             [-1, 32, 1, 1]               0
      BatchNorm2d-22             [-1, 32, 1, 1]              64
          Dropout-23             [-1, 32, 1, 1]               0
           Conv2d-24             [-1, 16, 1, 1]             512
             ReLU-25             [-1, 16, 1, 1]               0
      BatchNorm2d-26             [-1, 16, 1, 1]              32
          Dropout-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 7,624
Trainable params: 7,624
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.57
Params size (MB): 0.03
Estimated Total Size (MB): 0.60
----------------------------------------------------------------
```

## Receptive Field Calculation
![image](https://github.com/yuvaraj-venkataswamy/ERA-V1/assets/44864608/b049d257-ccfe-4601-ac08-66d8e29473c9)

## Training and Test logs of Final Model
```
EPOCH: 1
Loss=0.4880262017250061 Batch_id=937 Accuracy=84.46: 100%|██████████| 938/938 [01:39<00:00,  9.41it/s]

Test set: Average loss: 0.0890, Accuracy: 9707/10000 (97.07%)

EPOCH: 2
Loss=0.24310088157653809 Batch_id=937 Accuracy=94.74: 100%|██████████| 938/938 [01:36<00:00,  9.76it/s]

Test set: Average loss: 0.0397, Accuracy: 9869/10000 (98.69%)

EPOCH: 3
Loss=0.12208434194326401 Batch_id=937 Accuracy=95.74: 100%|██████████| 938/938 [01:34<00:00,  9.88it/s]

Test set: Average loss: 0.0454, Accuracy: 9859/10000 (98.59%)

EPOCH: 4
Loss=0.49917522072792053 Batch_id=937 Accuracy=96.36: 100%|██████████| 938/938 [01:35<00:00,  9.83it/s]

Test set: Average loss: 0.0336, Accuracy: 9896/10000 (98.96%)

EPOCH: 5
Loss=0.16808158159255981 Batch_id=937 Accuracy=96.57: 100%|██████████| 938/938 [01:34<00:00,  9.92it/s]

Test set: Average loss: 0.0269, Accuracy: 9903/10000 (99.03%)

EPOCH: 6
Loss=0.13319484889507294 Batch_id=937 Accuracy=96.80: 100%|██████████| 938/938 [01:38<00:00,  9.53it/s]

Test set: Average loss: 0.0256, Accuracy: 9916/10000 (99.16%)

EPOCH: 7
Loss=0.04289092496037483 Batch_id=937 Accuracy=96.92: 100%|██████████| 938/938 [01:36<00:00,  9.71it/s]

Test set: Average loss: 0.0277, Accuracy: 9909/10000 (99.09%)

EPOCH: 8
Loss=0.15397752821445465 Batch_id=937 Accuracy=97.19: 100%|██████████| 938/938 [01:37<00:00,  9.66it/s]

Test set: Average loss: 0.0209, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.14810331165790558 Batch_id=937 Accuracy=97.23: 100%|██████████| 938/938 [01:37<00:00,  9.66it/s]

Test set: Average loss: 0.0232, Accuracy: 9932/10000 (99.32%)

EPOCH: 10
Loss=0.006525579374283552 Batch_id=937 Accuracy=97.21: 100%|██████████| 938/938 [01:35<00:00,  9.79it/s]

Test set: Average loss: 0.0208, Accuracy: 9932/10000 (99.32%)

EPOCH: 11
Loss=0.0516677051782608 Batch_id=937 Accuracy=97.34: 100%|██████████| 938/938 [01:36<00:00,  9.71it/s]

Test set: Average loss: 0.0191, Accuracy: 9948/10000 (99.48%)

EPOCH: 12
Loss=0.3774683177471161 Batch_id=937 Accuracy=97.40: 100%|██████████| 938/938 [01:36<00:00,  9.72it/s]

Test set: Average loss: 0.0175, Accuracy: 9941/10000 (99.41%)

EPOCH: 13
Loss=0.022144075483083725 Batch_id=937 Accuracy=97.39: 100%|██████████| 938/938 [01:36<00:00,  9.77it/s]

Test set: Average loss: 0.0185, Accuracy: 9945/10000 (99.45%)

EPOCH: 14
Loss=0.07377509772777557 Batch_id=937 Accuracy=97.59: 100%|██████████| 938/938 [01:37<00:00,  9.66it/s]

Test set: Average loss: 0.0209, Accuracy: 9941/10000 (99.41%)

EPOCH: 15
Loss=0.03509518876671791 Batch_id=937 Accuracy=97.60: 100%|██████████| 938/938 [01:36<00:00,  9.68it/s]

Test set: Average loss: 0.0184, Accuracy: 9942/10000 (99.42%)
```
