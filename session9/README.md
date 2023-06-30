# Session 9 - Advanced Convolutions, Data Augmentation and Visualization

## Objective 
1. Build network that
  - has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) 
  - total RF must be more than 44
  - one of the layers must use Depthwise Separable Convolution
  - one of the layers must use Dilated Convolution
  - use GAP (compulsory):- add FC after GAP to target #of classes (optional)
2. albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
3. Achieve 85% accuracy, as many epochs as you want.
4. Total Params to be less than 200k.
  
## Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes. There are 50000 training images and 10000 test images. Classes include plane, car, bird, cat, deer, dog, frog, horse, ship and truck.

## Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 12, 30, 30]             324
              ReLU-2           [-1, 12, 30, 30]               0
       BatchNorm2d-3           [-1, 12, 30, 30]              24
           Dropout-4           [-1, 12, 30, 30]               0
            Conv2d-5           [-1, 24, 28, 28]           2,592
              ReLU-6           [-1, 24, 28, 28]               0
       BatchNorm2d-7           [-1, 24, 28, 28]              48
           Dropout-8           [-1, 24, 28, 28]               0
            Conv2d-9           [-1, 48, 14, 14]          10,368
             ReLU-10           [-1, 48, 14, 14]               0
      BatchNorm2d-11           [-1, 48, 14, 14]              96
          Dropout-12           [-1, 48, 14, 14]               0
           Conv2d-13           [-1, 48, 14, 14]          20,736
             ReLU-14           [-1, 48, 14, 14]               0
      BatchNorm2d-15           [-1, 48, 14, 14]              96
          Dropout-16           [-1, 48, 14, 14]               0
           Conv2d-17           [-1, 12, 30, 30]             324
             ReLU-18           [-1, 12, 30, 30]               0
      BatchNorm2d-19           [-1, 12, 30, 30]              24
          Dropout-20           [-1, 12, 30, 30]               0
           Conv2d-21           [-1, 24, 28, 28]           2,592
             ReLU-22           [-1, 24, 28, 28]               0
      BatchNorm2d-23           [-1, 24, 28, 28]              48
          Dropout-24           [-1, 24, 28, 28]               0
           Conv2d-25           [-1, 48, 14, 14]          10,368
             ReLU-26           [-1, 48, 14, 14]               0
      BatchNorm2d-27           [-1, 48, 14, 14]              96
          Dropout-28           [-1, 48, 14, 14]               0
           Conv2d-29           [-1, 48, 14, 14]          20,736
             ReLU-30           [-1, 48, 14, 14]               0
      BatchNorm2d-31           [-1, 48, 14, 14]              96
          Dropout-32           [-1, 48, 14, 14]               0
           Conv2d-33           [-1, 96, 12, 12]             864
           Conv2d-34           [-1, 60, 12, 12]           5,760
             ReLU-35           [-1, 60, 12, 12]               0
      BatchNorm2d-36           [-1, 60, 12, 12]             120
          Dropout-37           [-1, 60, 12, 12]               0
           Conv2d-38             [-1, 30, 6, 6]          16,200
             ReLU-39             [-1, 30, 6, 6]               0
      BatchNorm2d-40             [-1, 30, 6, 6]              60
          Dropout-41             [-1, 30, 6, 6]               0
           Conv2d-42             [-1, 10, 6, 6]           2,700
             ReLU-43             [-1, 10, 6, 6]               0
      BatchNorm2d-44             [-1, 10, 6, 6]              20
          Dropout-45             [-1, 10, 6, 6]               0
        AvgPool2d-46             [-1, 10, 1, 1]               0
================================================================
Total params: 94,292
Trainable params: 94,292
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.37
Params size (MB): 0.36
Estimated Total Size (MB): 3.74
----------------------------------------------------------------
```
## Receptive Field Calculation

![image](https://github.com/yuvaraj-venkataswamy/ERA-V1/assets/44864608/d2103a9f-3057-48ad-b9ac-4e98fdd9de77)

## Training & Testing Logs

## Accuracy and Loss Graphs
