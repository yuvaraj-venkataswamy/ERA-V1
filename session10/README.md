# Session 10 - Residual Connections in CNNs and One Cycle Policy

## Objective
1. Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    
    2. Layer1 -
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        - Add(X, R1)
    
    3. Layer 2 -
        - Conv 3x3 [256k]
        - MaxPooling2D
        - BN
        - ReLU
    
    4. Layer 3 -
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        - Add(X, R2)
    5. MaxPooling with Kernel Size 4
    6. FC Layer 
    7. SoftMax

2. Uses One Cycle Policy such that:
    - Total Epochs = 24
    - Max at Epoch = 5
    - LRMIN = FIND
    - LRMAX = FIND
    - NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Use ADAM, and CrossEntropyLoss
6. Target Accuracy: 90%

## Custom ResNet Model

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
         Dropout2d-3           [-1, 64, 32, 32]               0
       ConvBNBlock-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
         Dropout2d-8          [-1, 128, 16, 16]               0
   TransitionBlock-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
        Dropout2d-12          [-1, 128, 16, 16]               0
      ConvBNBlock-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,456
      BatchNorm2d-15          [-1, 128, 16, 16]             256
        Dropout2d-16          [-1, 128, 16, 16]               0
      ConvBNBlock-17          [-1, 128, 16, 16]               0
         ResBlock-18          [-1, 128, 16, 16]               0
           Conv2d-19          [-1, 256, 16, 16]         294,912
        MaxPool2d-20            [-1, 256, 8, 8]               0
      BatchNorm2d-21            [-1, 256, 8, 8]             512
        Dropout2d-22            [-1, 256, 8, 8]               0
  TransitionBlock-23            [-1, 256, 8, 8]               0
           Conv2d-24            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-25            [-1, 512, 4, 4]               0
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
        Dropout2d-27            [-1, 512, 4, 4]               0
  TransitionBlock-28            [-1, 512, 4, 4]               0
           Conv2d-29            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-30            [-1, 512, 4, 4]           1,024
        Dropout2d-31            [-1, 512, 4, 4]               0
      ConvBNBlock-32            [-1, 512, 4, 4]               0
           Conv2d-33            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-34            [-1, 512, 4, 4]           1,024
        Dropout2d-35            [-1, 512, 4, 4]               0
      ConvBNBlock-36            [-1, 512, 4, 4]               0
         ResBlock-37            [-1, 512, 4, 4]               0
        MaxPool2d-38            [-1, 512, 1, 1]               0
           Linear-39                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.32
Params size (MB): 25.07
Estimated Total Size (MB): 33.40
----------------------------------------------------------------

```

- No of Parameters - 6.5M
- Highest Train Accuracy -96.21%
- Highest test Accuracy - 90.88%

## LR Search Plot
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session10/images/LR.png)

## Training and Testing Logs
```
EPOCH: 1
Loss=1.5430887937545776 Batch_id=97 Accuracy=40.42: 100%|██████████| 98/98 [00:23<00:00,  4.21it/s]

Test set: Average loss: 1.4836, Accuracy: 4872/10000 (48.72%)

Test Accuracy: 48.72 has increased. Saving the model
EPOCH: 2
Loss=1.1940348148345947 Batch_id=97 Accuracy=57.93: 100%|██████████| 98/98 [00:21<00:00,  4.46it/s]

Test set: Average loss: 2.1974, Accuracy: 5086/10000 (50.86%)

Test Accuracy: 50.86 has increased. Saving the model
EPOCH: 3
Loss=0.7818192839622498 Batch_id=97 Accuracy=66.78: 100%|██████████| 98/98 [00:21<00:00,  4.48it/s]

Test set: Average loss: 1.2094, Accuracy: 6317/10000 (63.17%)

Test Accuracy: 63.17 has increased. Saving the model
EPOCH: 4
Loss=0.9456764459609985 Batch_id=97 Accuracy=69.72: 100%|██████████| 98/98 [00:23<00:00,  4.13it/s]

Test set: Average loss: 0.9750, Accuracy: 7246/10000 (72.46%)

Test Accuracy: 72.46 has increased. Saving the model
EPOCH: 5
Loss=1.0293563604354858 Batch_id=97 Accuracy=73.38: 100%|██████████| 98/98 [00:22<00:00,  4.39it/s]

Test set: Average loss: 0.9240, Accuracy: 7363/10000 (73.63%)

Test Accuracy: 73.63 has increased. Saving the model
EPOCH: 6
Loss=0.7259105443954468 Batch_id=97 Accuracy=76.41: 100%|██████████| 98/98 [00:22<00:00,  4.43it/s]

Test set: Average loss: 0.6481, Accuracy: 7997/10000 (79.97%)

Test Accuracy: 79.97 has increased. Saving the model
EPOCH: 7
Loss=0.5471834540367126 Batch_id=97 Accuracy=81.11: 100%|██████████| 98/98 [00:23<00:00,  4.13it/s]

Test set: Average loss: 0.6024, Accuracy: 8176/10000 (81.76%)

Test Accuracy: 81.76 has increased. Saving the model
EPOCH: 8
Loss=0.5164240002632141 Batch_id=97 Accuracy=83.28: 100%|██████████| 98/98 [00:22<00:00,  4.38it/s]

Test set: Average loss: 0.5348, Accuracy: 8245/10000 (82.45%)

Test Accuracy: 82.45 has increased. Saving the model
EPOCH: 9
Loss=0.41215404868125916 Batch_id=97 Accuracy=85.72: 100%|██████████| 98/98 [00:22<00:00,  4.40it/s]

Test set: Average loss: 0.4951, Accuracy: 8460/10000 (84.60%)

Test Accuracy: 84.6 has increased. Saving the model
EPOCH: 10
Loss=0.3196747303009033 Batch_id=97 Accuracy=87.17: 100%|██████████| 98/98 [00:23<00:00,  4.13it/s]

Test set: Average loss: 0.4537, Accuracy: 8600/10000 (86.00%)

Test Accuracy: 86.0 has increased. Saving the model
EPOCH: 11
Loss=0.31176382303237915 Batch_id=97 Accuracy=88.48: 100%|██████████| 98/98 [00:22<00:00,  4.37it/s]

Test set: Average loss: 0.4146, Accuracy: 8660/10000 (86.60%)

Test Accuracy: 86.6 has increased. Saving the model
EPOCH: 12
Loss=0.313408225774765 Batch_id=97 Accuracy=89.65: 100%|██████████| 98/98 [00:22<00:00,  4.45it/s]

Test set: Average loss: 0.3941, Accuracy: 8751/10000 (87.51%)

Test Accuracy: 87.51 has increased. Saving the model
EPOCH: 13
Loss=0.2645241618156433 Batch_id=97 Accuracy=90.91: 100%|██████████| 98/98 [00:23<00:00,  4.16it/s]

Test set: Average loss: 0.4235, Accuracy: 8694/10000 (86.94%)

EPOCH: 14
Loss=0.19798119366168976 Batch_id=97 Accuracy=91.82: 100%|██████████| 98/98 [00:22<00:00,  4.37it/s]

Test set: Average loss: 0.3636, Accuracy: 8905/10000 (89.05%)

Test Accuracy: 89.05 has increased. Saving the model
EPOCH: 15
Loss=0.21440164744853973 Batch_id=97 Accuracy=92.93: 100%|██████████| 98/98 [00:22<00:00,  4.43it/s]

Test set: Average loss: 0.3379, Accuracy: 8982/10000 (89.82%)

Test Accuracy: 89.82 has increased. Saving the model
EPOCH: 16
Loss=0.19927218556404114 Batch_id=97 Accuracy=93.40: 100%|██████████| 98/98 [00:23<00:00,  4.20it/s]

Test set: Average loss: 0.3238, Accuracy: 9019/10000 (90.19%)

Test Accuracy: 90.19 has increased. Saving the model
EPOCH: 17
Loss=0.17854858934879303 Batch_id=97 Accuracy=94.25: 100%|██████████| 98/98 [00:22<00:00,  4.36it/s]

Test set: Average loss: 0.3314, Accuracy: 9048/10000 (90.48%)

Test Accuracy: 90.48 has increased. Saving the model
EPOCH: 18
Loss=0.13835105299949646 Batch_id=97 Accuracy=94.67: 100%|██████████| 98/98 [00:22<00:00,  4.45it/s]

Test set: Average loss: 0.3185, Accuracy: 9032/10000 (90.32%)

EPOCH: 19
Loss=0.1482393890619278 Batch_id=97 Accuracy=95.21: 100%|██████████| 98/98 [00:23<00:00,  4.17it/s]

Test set: Average loss: 0.3118, Accuracy: 9058/10000 (90.58%)

Test Accuracy: 90.58 has increased. Saving the model
EPOCH: 20
Loss=0.14366328716278076 Batch_id=97 Accuracy=95.23: 100%|██████████| 98/98 [00:22<00:00,  4.34it/s]

Test set: Average loss: 0.3156, Accuracy: 9073/10000 (90.73%)

Test Accuracy: 90.73 has increased. Saving the model
EPOCH: 21
Loss=0.15959899127483368 Batch_id=97 Accuracy=95.64: 100%|██████████| 98/98 [00:22<00:00,  4.42it/s]

Test set: Average loss: 0.3105, Accuracy: 9089/10000 (90.89%)

Test Accuracy: 90.89 has increased. Saving the model
EPOCH: 22
Loss=0.15333612263202667 Batch_id=97 Accuracy=95.74: 100%|██████████| 98/98 [00:23<00:00,  4.16it/s]

Test set: Average loss: 0.3118, Accuracy: 9087/10000 (90.87%)

EPOCH: 23
Loss=0.11260347068309784 Batch_id=97 Accuracy=95.93: 100%|██████████| 98/98 [00:23<00:00,  4.15it/s]

Test set: Average loss: 0.3086, Accuracy: 9106/10000 (91.06%)

Test Accuracy: 91.06 has increased. Saving the model
EPOCH: 24
Loss=0.13081176578998566 Batch_id=97 Accuracy=95.89: 100%|██████████| 98/98 [00:22<00:00,  4.36it/s]

Test set: Average loss: 0.3077, Accuracy: 9098/10000 (90.98%)
```

## Accuracy Plot

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%209/Images/misclassified_images.png)

Accuracy of plane : 85 %
Accuracy of car : 100 %
Accuracy of bird : 100 %
Accuracy of cat : 75 %
Accuracy of deer : 100 %
Accuracy of dog : 66 %
Accuracy of frog : 81 %
Accuracy of horse : 83 %
Accuracy of ship : 91 %
Accuracy of truck : 100 %

## Misclassified Images
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session10/images/misclassified_images.png)

## Misclassified Images with Grad-CAM
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session10/images/misclassified_gradcam.png)
