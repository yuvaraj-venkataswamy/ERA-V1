# Session 11 - CAMs, LRs, and Optimizers

## 1. Objective
1. Train ResNet18
2. Apply these transforms while training:
    - RandomCrop(32, padding=4)
    - CutOut(16x16)
3. Use Cifar10 for 20 Epochs
4. show loss curves for test and train datasets
5. show a gallery of 10 misclassified images
6. show gradcam output on 10 misclassified images


## 2. ResNet18 Model

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                  [-1, 200]         102,600
================================================================
Total params: 11,271,432
Trainable params: 11,271,432
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 43.00
Estimated Total Size (MB): 54.26
----------------------------------------------------------------

```

## 3. Training and Testing Logs
```
EPOCH: 1
Loss=1.5754945278167725 Batch_id=390 LR=0.01000 Accuracy=33.05: 100%|██████████| 391/391 [00:44<00:00,  8.86it/s]

Test set: Average loss: 0.0131, Accuracy: 4261/10000 (42.61%)

EPOCH: 2
Loss=1.4961639642715454 Batch_id=390 LR=0.01000 Accuracy=47.88: 100%|██████████| 391/391 [00:44<00:00,  8.89it/s]

Test set: Average loss: 0.0095, Accuracy: 5635/10000 (56.35%)

EPOCH: 3
Loss=1.1374207735061646 Batch_id=390 LR=0.01000 Accuracy=57.71: 100%|██████████| 391/391 [00:44<00:00,  8.76it/s]

Test set: Average loss: 0.0084, Accuracy: 6183/10000 (61.83%)

EPOCH: 4
Loss=0.9487029910087585 Batch_id=390 LR=0.01000 Accuracy=64.12: 100%|██████████| 391/391 [00:45<00:00,  8.65it/s]

Test set: Average loss: 0.0070, Accuracy: 6866/10000 (68.66%)

EPOCH: 5
Loss=0.7605847120285034 Batch_id=390 LR=0.01000 Accuracy=68.48: 100%|██████████| 391/391 [00:45<00:00,  8.60it/s]

Test set: Average loss: 0.0072, Accuracy: 6953/10000 (69.53%)

EPOCH: 6
Loss=0.8113424181938171 Batch_id=390 LR=0.01000 Accuracy=71.79: 100%|██████████| 391/391 [00:45<00:00,  8.59it/s]

Test set: Average loss: 0.0050, Accuracy: 7811/10000 (78.11%)

EPOCH: 7
Loss=0.5964615345001221 Batch_id=390 LR=0.01000 Accuracy=73.94: 100%|██████████| 391/391 [00:45<00:00,  8.58it/s]

Test set: Average loss: 0.0053, Accuracy: 7691/10000 (76.91%)

EPOCH: 8
Loss=0.5564097166061401 Batch_id=390 LR=0.01000 Accuracy=75.34: 100%|██████████| 391/391 [00:45<00:00,  8.58it/s]

Test set: Average loss: 0.0059, Accuracy: 7436/10000 (74.36%)

EPOCH: 9
Loss=0.6229034066200256 Batch_id=390 LR=0.01000 Accuracy=76.78: 100%|██████████| 391/391 [00:45<00:00,  8.59it/s]

Test set: Average loss: 0.0044, Accuracy: 8118/10000 (81.18%)

EPOCH: 10
Loss=0.5678735971450806 Batch_id=390 LR=0.01000 Accuracy=77.47: 100%|██████████| 391/391 [00:45<00:00,  8.58it/s]

Test set: Average loss: 0.0050, Accuracy: 7801/10000 (78.01%)

EPOCH: 11
Loss=0.6253218650817871 Batch_id=390 LR=0.01000 Accuracy=78.32: 100%|██████████| 391/391 [00:45<00:00,  8.61it/s]

Test set: Average loss: 0.0041, Accuracy: 8181/10000 (81.81%)

EPOCH: 12
Loss=0.5251783132553101 Batch_id=390 LR=0.01000 Accuracy=78.96: 100%|██████████| 391/391 [00:45<00:00,  8.63it/s]

Test set: Average loss: 0.0044, Accuracy: 8086/10000 (80.86%)

EPOCH: 13
Loss=0.4598936140537262 Batch_id=390 LR=0.01000 Accuracy=79.68: 100%|██████████| 391/391 [00:45<00:00,  8.59it/s]

Test set: Average loss: 0.0041, Accuracy: 8285/10000 (82.85%)

EPOCH: 14
Loss=0.6058142781257629 Batch_id=390 LR=0.01000 Accuracy=80.08: 100%|██████████| 391/391 [00:45<00:00,  8.64it/s]

Test set: Average loss: 0.0039, Accuracy: 8385/10000 (83.85%)

EPOCH: 15
Loss=0.5530171394348145 Batch_id=390 LR=0.01000 Accuracy=80.33: 100%|██████████| 391/391 [00:45<00:00,  8.65it/s]

Test set: Average loss: 0.0042, Accuracy: 8171/10000 (81.71%)

EPOCH: 16
Loss=0.6223279237747192 Batch_id=390 LR=0.01000 Accuracy=80.90: 100%|██████████| 391/391 [00:45<00:00,  8.62it/s]

Test set: Average loss: 0.0044, Accuracy: 8220/10000 (82.20%)

EPOCH: 17
Loss=0.5162665843963623 Batch_id=390 LR=0.01000 Accuracy=81.07: 100%|██████████| 391/391 [00:44<00:00,  8.71it/s]

Test set: Average loss: 0.0039, Accuracy: 8311/10000 (83.11%)

EPOCH: 18
Loss=0.5065620541572571 Batch_id=390 LR=0.01000 Accuracy=81.24: 100%|██████████| 391/391 [00:45<00:00,  8.67it/s]

Test set: Average loss: 0.0037, Accuracy: 8433/10000 (84.33%)

EPOCH: 19
Loss=0.8082538843154907 Batch_id=390 LR=0.01000 Accuracy=81.88: 100%|██████████| 391/391 [00:45<00:00,  8.63it/s]

Test set: Average loss: 0.0040, Accuracy: 8375/10000 (83.75%)

EPOCH: 20
Loss=0.577003002166748 Batch_id=390 LR=0.01000 Accuracy=82.09: 100%|██████████| 391/391 [00:44<00:00,  8.69it/s]

Test set: Average loss: 0.0037, Accuracy: 8422/10000 (84.22%)

```

## 4. Accuracy & Loss Plot

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session11/images/accuracy_plot_s11.png)


## 5. Misclassified Images

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session11/images/misclassified_images_s11.png)


## 6. GRADCOM-Misclassified Images

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session11/images/misclassified_gradcam_s11.png)

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session11/images/misclassified_gradcam1_s11.png)

