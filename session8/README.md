# SESSION 8 - BATCH NORMALIZATION & REGULARIZATION - Assignment

## Objective
- Change the dataset to CIFAR10
- Make this network:
  - C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
  - Keep the parameter count less than 50000
  - Try and add one layer to another
  - Max Epochs is 20
- You are making 3 versions of the above code (in each case achieve above 70% accuracy):
  - Network with Group Normalization
  - Network with Layer Normalization
  - Network with Batch Normalization
- Share these details
  - Training accuracy for 3 models
  - Test accuracy for 3 models
  - Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images.

## Normalization
Normalization refers to the process of standardizing inputs to a neural network. Different normalization techniques can standardize different segments of the input.

1. `Batch normalization` - standardizes each mini-batch input to a layer.
2. `Layer normalization` - normalizes the activations along the feature/channel direction instead of the batch direction. Removes depdency on batch size.
3. `Group normalization` - similar to layer normalization, however, it divides the features/channels into groups and normalizes each group separately.

## Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 32, 32]             216
              ReLU-2            [-1, 8, 32, 32]               0
       BatchNorm2d-3            [-1, 8, 32, 32]              16
           Dropout-4            [-1, 8, 32, 32]               0
            Conv2d-5           [-1, 16, 32, 32]           1,152
              ReLU-6           [-1, 16, 32, 32]               0
       BatchNorm2d-7           [-1, 16, 32, 32]              32
           Dropout-8           [-1, 16, 32, 32]               0
            Conv2d-9            [-1, 8, 16, 16]             136
        MaxPool2d-10              [-1, 8, 8, 8]               0
           Conv2d-11             [-1, 16, 8, 8]           1,152
             ReLU-12             [-1, 16, 8, 8]               0
      BatchNorm2d-13             [-1, 16, 8, 8]              32
          Dropout-14             [-1, 16, 8, 8]               0
           Conv2d-15           [-1, 32, 10, 10]             512
             ReLU-16           [-1, 32, 10, 10]               0
      BatchNorm2d-17           [-1, 32, 10, 10]              64
        Dropout2d-18           [-1, 32, 10, 10]               0
           Conv2d-19             [-1, 16, 5, 5]             528
        MaxPool2d-20             [-1, 16, 2, 2]               0
           Conv2d-21             [-1, 32, 2, 2]           4,608
             ReLU-22             [-1, 32, 2, 2]               0
      BatchNorm2d-23             [-1, 32, 2, 2]              64
          Dropout-24             [-1, 32, 2, 2]               0
           Conv2d-25             [-1, 64, 2, 2]          18,432
             ReLU-26             [-1, 64, 2, 2]               0
      BatchNorm2d-27             [-1, 64, 2, 2]             128
          Dropout-28             [-1, 64, 2, 2]               0
           Conv2d-29             [-1, 32, 1, 1]           2,080
             ReLU-30             [-1, 32, 1, 1]               0
AdaptiveAvgPool2d-31             [-1, 32, 1, 1]               0
           Conv2d-32             [-1, 10, 1, 1]             320
================================================================
Total params: 29,472
Trainable params: 29,472
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.91
Params size (MB): 0.11
Estimated Total Size (MB): 1.04
----------------------------------------------------------------
```

## Accuracy and Loss plots
1. BATCH NORMALIZATION
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/images/BN.png)
   
2. LAYER NORMALIZATION
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/images/LN.png)
   
3. GROUP NORMALIZATION
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/images/LN.png)

## (Accuracy ia not crossed better level, I will go through into more details)
