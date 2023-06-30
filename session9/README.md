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

![image](https://github.com/yuvaraj-venkataswamy/ERA-V1/assets/44864608/93ea4e53-6cef-4a73-9b2d-a11eeab95057)

## Training & Testing Logs

```
EPOCH: 1
  0%|          | 0/391 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
Loss=1.3866239786148071 Batch_id=390 LR=0.00064 Accuracy=41.06: 100%|██████████| 391/391 [00:16<00:00, 23.34it/s]

Test set: Average loss: 0.0106, Accuracy: 5536/10000 (55.36%)

EPOCH: 2
Loss=1.376880407333374 Batch_id=390 LR=0.00076 Accuracy=53.18: 100%|██████████| 391/391 [00:16<00:00, 23.52it/s]

Test set: Average loss: 0.0089, Accuracy: 6248/10000 (62.48%)

EPOCH: 3
Loss=1.1449631452560425 Batch_id=390 LR=0.00095 Accuracy=58.19: 100%|██████████| 391/391 [00:16<00:00, 23.25it/s]

Test set: Average loss: 0.0078, Accuracy: 6701/10000 (67.01%)

EPOCH: 4
Loss=1.2006714344024658 Batch_id=390 LR=0.00122 Accuracy=61.35: 100%|██████████| 391/391 [00:16<00:00, 23.27it/s]

Test set: Average loss: 0.0076, Accuracy: 6681/10000 (66.81%)

EPOCH: 5
Loss=1.195878505706787 Batch_id=390 LR=0.00156 Accuracy=63.60: 100%|██████████| 391/391 [00:16<00:00, 23.51it/s]

Test set: Average loss: 0.0068, Accuracy: 7051/10000 (70.51%)

EPOCH: 6
Loss=1.0278211832046509 Batch_id=390 LR=0.00198 Accuracy=65.00: 100%|██████████| 391/391 [00:17<00:00, 22.48it/s]

Test set: Average loss: 0.0066, Accuracy: 7081/10000 (70.81%)

EPOCH: 7
Loss=0.9210250973701477 Batch_id=390 LR=0.00245 Accuracy=66.50: 100%|██████████| 391/391 [00:16<00:00, 23.72it/s]

Test set: Average loss: 0.0060, Accuracy: 7358/10000 (73.58%)

EPOCH: 8
Loss=0.8296180963516235 Batch_id=390 LR=0.00298 Accuracy=68.03: 100%|██████████| 391/391 [00:16<00:00, 23.97it/s]

Test set: Average loss: 0.0060, Accuracy: 7358/10000 (73.58%)

EPOCH: 9
Loss=0.764093816280365 Batch_id=390 LR=0.00357 Accuracy=68.76: 100%|██████████| 391/391 [00:16<00:00, 24.01it/s]

Test set: Average loss: 0.0057, Accuracy: 7516/10000 (75.16%)

EPOCH: 10
Loss=0.8374002575874329 Batch_id=390 LR=0.00420 Accuracy=69.35: 100%|██████████| 391/391 [00:16<00:00, 23.57it/s]

Test set: Average loss: 0.0052, Accuracy: 7728/10000 (77.28%)

EPOCH: 11
Loss=0.8652185201644897 Batch_id=390 LR=0.00487 Accuracy=70.40: 100%|██████████| 391/391 [00:16<00:00, 23.40it/s]

Test set: Average loss: 0.0053, Accuracy: 7701/10000 (77.01%)

EPOCH: 12
Loss=0.7065510749816895 Batch_id=390 LR=0.00558 Accuracy=70.55: 100%|██████████| 391/391 [00:16<00:00, 23.79it/s]

Test set: Average loss: 0.0057, Accuracy: 7567/10000 (75.67%)

EPOCH: 13
Loss=0.7191526293754578 Batch_id=390 LR=0.00630 Accuracy=71.41: 100%|██████████| 391/391 [00:16<00:00, 24.07it/s]

Test set: Average loss: 0.0054, Accuracy: 7634/10000 (76.34%)

EPOCH: 14
Loss=0.910452663898468 Batch_id=390 LR=0.00705 Accuracy=71.45: 100%|██████████| 391/391 [00:16<00:00, 24.23it/s]

Test set: Average loss: 0.0055, Accuracy: 7640/10000 (76.40%)

EPOCH: 15
Loss=0.7917444109916687 Batch_id=390 LR=0.00780 Accuracy=71.68: 100%|██████████| 391/391 [00:16<00:00, 23.47it/s]

Test set: Average loss: 0.0052, Accuracy: 7712/10000 (77.12%)

EPOCH: 16
Loss=0.8000218272209167 Batch_id=390 LR=0.00855 Accuracy=72.07: 100%|██████████| 391/391 [00:16<00:00, 24.31it/s]

Test set: Average loss: 0.0056, Accuracy: 7628/10000 (76.28%)

EPOCH: 17
Loss=0.6852819919586182 Batch_id=390 LR=0.00930 Accuracy=72.57: 100%|██████████| 391/391 [00:16<00:00, 24.07it/s]

Test set: Average loss: 0.0049, Accuracy: 7852/10000 (78.52%)

EPOCH: 18
Loss=0.8959468007087708 Batch_id=390 LR=0.01003 Accuracy=72.93: 100%|██████████| 391/391 [00:16<00:00, 24.17it/s]

Test set: Average loss: 0.0047, Accuracy: 7918/10000 (79.18%)

EPOCH: 19
Loss=0.6952611804008484 Batch_id=390 LR=0.01073 Accuracy=73.31: 100%|██████████| 391/391 [00:16<00:00, 24.22it/s]

Test set: Average loss: 0.0048, Accuracy: 7951/10000 (79.51%)

EPOCH: 20
Loss=0.8318716287612915 Batch_id=390 LR=0.01140 Accuracy=73.28: 100%|██████████| 391/391 [00:16<00:00, 23.75it/s]

Test set: Average loss: 0.0052, Accuracy: 7717/10000 (77.17%)

EPOCH: 21
Loss=0.5587401986122131 Batch_id=390 LR=0.01203 Accuracy=73.36: 100%|██████████| 391/391 [00:16<00:00, 23.96it/s]

Test set: Average loss: 0.0048, Accuracy: 7956/10000 (79.56%)

EPOCH: 22
Loss=0.655518114566803 Batch_id=390 LR=0.01262 Accuracy=73.72: 100%|██████████| 391/391 [00:16<00:00, 23.94it/s]

Test set: Average loss: 0.0047, Accuracy: 7988/10000 (79.88%)

EPOCH: 23
Loss=0.7268597483634949 Batch_id=390 LR=0.01315 Accuracy=74.13: 100%|██████████| 391/391 [00:16<00:00, 23.67it/s]

Test set: Average loss: 0.0047, Accuracy: 7975/10000 (79.75%)

EPOCH: 24
Loss=1.0040514469146729 Batch_id=390 LR=0.01363 Accuracy=73.96: 100%|██████████| 391/391 [00:16<00:00, 23.26it/s]

Test set: Average loss: 0.0046, Accuracy: 7984/10000 (79.84%)

EPOCH: 25
Loss=0.8473714590072632 Batch_id=390 LR=0.01404 Accuracy=74.09: 100%|██████████| 391/391 [00:16<00:00, 23.80it/s]

Test set: Average loss: 0.0052, Accuracy: 7775/10000 (77.75%)

EPOCH: 26
Loss=0.7950946092605591 Batch_id=390 LR=0.01438 Accuracy=74.36: 100%|██████████| 391/391 [00:16<00:00, 23.80it/s]

Test set: Average loss: 0.0049, Accuracy: 7875/10000 (78.75%)

EPOCH: 27
Loss=0.7588018178939819 Batch_id=390 LR=0.01465 Accuracy=74.48: 100%|██████████| 391/391 [00:16<00:00, 24.16it/s]

Test set: Average loss: 0.0052, Accuracy: 7825/10000 (78.25%)

EPOCH: 28
Loss=0.6660417318344116 Batch_id=390 LR=0.01484 Accuracy=74.42: 100%|██████████| 391/391 [00:16<00:00, 23.11it/s]

Test set: Average loss: 0.0046, Accuracy: 8013/10000 (80.13%)

EPOCH: 29
Loss=0.6954179406166077 Batch_id=390 LR=0.01496 Accuracy=74.79: 100%|██████████| 391/391 [00:16<00:00, 23.65it/s]

Test set: Average loss: 0.0048, Accuracy: 7927/10000 (79.27%)

EPOCH: 30
Loss=0.9274923205375671 Batch_id=390 LR=0.01500 Accuracy=75.25: 100%|██████████| 391/391 [00:16<00:00, 23.79it/s]

Test set: Average loss: 0.0050, Accuracy: 7914/10000 (79.14%)

EPOCH: 31
Loss=0.4732210040092468 Batch_id=390 LR=0.01499 Accuracy=75.19: 100%|██████████| 391/391 [00:16<00:00, 23.73it/s]

Test set: Average loss: 0.0046, Accuracy: 8036/10000 (80.36%)

EPOCH: 32
Loss=0.562386691570282 Batch_id=390 LR=0.01497 Accuracy=75.25: 100%|██████████| 391/391 [00:16<00:00, 23.78it/s]

Test set: Average loss: 0.0045, Accuracy: 8095/10000 (80.95%)

EPOCH: 33
Loss=0.809414267539978 Batch_id=390 LR=0.01493 Accuracy=75.26: 100%|██████████| 391/391 [00:16<00:00, 23.84it/s]

Test set: Average loss: 0.0044, Accuracy: 8106/10000 (81.06%)

EPOCH: 34
Loss=0.635543704032898 Batch_id=390 LR=0.01488 Accuracy=75.55: 100%|██████████| 391/391 [00:16<00:00, 23.93it/s]

Test set: Average loss: 0.0045, Accuracy: 8077/10000 (80.77%)

EPOCH: 35
Loss=0.5966061353683472 Batch_id=390 LR=0.01481 Accuracy=75.53: 100%|██████████| 391/391 [00:16<00:00, 23.97it/s]

Test set: Average loss: 0.0047, Accuracy: 7959/10000 (79.59%)

EPOCH: 36
Loss=0.784514307975769 Batch_id=390 LR=0.01473 Accuracy=75.70: 100%|██████████| 391/391 [00:16<00:00, 23.27it/s]

Test set: Average loss: 0.0044, Accuracy: 8062/10000 (80.62%)

EPOCH: 37
Loss=0.6881356239318848 Batch_id=390 LR=0.01463 Accuracy=74.22: 100%|██████████| 391/391 [00:17<00:00, 22.30it/s]

Test set: Average loss: 0.0047, Accuracy: 7920/10000 (79.20%)

EPOCH: 38
Loss=0.8033257722854614 Batch_id=390 LR=0.01452 Accuracy=74.82: 100%|██████████| 391/391 [00:16<00:00, 23.10it/s]

Test set: Average loss: 0.0046, Accuracy: 8050/10000 (80.50%)

EPOCH: 39
Loss=0.82331383228302 Batch_id=390 LR=0.01440 Accuracy=75.65: 100%|██████████| 391/391 [00:16<00:00, 23.63it/s]

Test set: Average loss: 0.0041, Accuracy: 8212/10000 (82.12%)

EPOCH: 40
Loss=0.6204553842544556 Batch_id=390 LR=0.01426 Accuracy=75.97: 100%|██████████| 391/391 [00:16<00:00, 24.20it/s]

Test set: Average loss: 0.0043, Accuracy: 8179/10000 (81.79%)

EPOCH: 41
Loss=0.6314924955368042 Batch_id=390 LR=0.01410 Accuracy=76.46: 100%|██████████| 391/391 [00:16<00:00, 23.18it/s]

Test set: Average loss: 0.0045, Accuracy: 7995/10000 (79.95%)

EPOCH: 42
Loss=0.6855339407920837 Batch_id=390 LR=0.01394 Accuracy=76.51: 100%|██████████| 391/391 [00:16<00:00, 24.05it/s]

Test set: Average loss: 0.0041, Accuracy: 8280/10000 (82.80%)

EPOCH: 43
Loss=0.839387059211731 Batch_id=390 LR=0.01376 Accuracy=76.32: 100%|██████████| 391/391 [00:16<00:00, 23.97it/s]

Test set: Average loss: 0.0044, Accuracy: 8065/10000 (80.65%)

EPOCH: 44
Loss=0.6113983988761902 Batch_id=390 LR=0.01357 Accuracy=76.44: 100%|██████████| 391/391 [00:16<00:00, 23.30it/s]

Test set: Average loss: 0.0046, Accuracy: 8045/10000 (80.45%)

EPOCH: 45
Loss=0.6492141485214233 Batch_id=390 LR=0.01336 Accuracy=76.74: 100%|██████████| 391/391 [00:17<00:00, 22.34it/s]

Test set: Average loss: 0.0043, Accuracy: 8197/10000 (81.97%)

EPOCH: 46
Loss=0.6213717460632324 Batch_id=390 LR=0.01315 Accuracy=77.00: 100%|██████████| 391/391 [00:16<00:00, 24.04it/s]

Test set: Average loss: 0.0043, Accuracy: 8136/10000 (81.36%)

EPOCH: 47
Loss=0.5823884010314941 Batch_id=390 LR=0.01292 Accuracy=77.02: 100%|██████████| 391/391 [00:17<00:00, 22.09it/s]

Test set: Average loss: 0.0042, Accuracy: 8218/10000 (82.18%)

EPOCH: 48
Loss=0.6427104473114014 Batch_id=390 LR=0.01268 Accuracy=77.35: 100%|██████████| 391/391 [00:16<00:00, 23.76it/s]

Test set: Average loss: 0.0040, Accuracy: 8308/10000 (83.08%)

EPOCH: 49
Loss=0.40768498182296753 Batch_id=390 LR=0.01243 Accuracy=77.64: 100%|██████████| 391/391 [00:17<00:00, 22.91it/s]

Test set: Average loss: 0.0040, Accuracy: 8265/10000 (82.65%)

EPOCH: 50
Loss=0.6190169453620911 Batch_id=390 LR=0.01218 Accuracy=77.48: 100%|██████████| 391/391 [00:16<00:00, 23.60it/s]

Test set: Average loss: 0.0041, Accuracy: 8213/10000 (82.13%)

EPOCH: 51
Loss=0.46336716413497925 Batch_id=390 LR=0.01191 Accuracy=77.77: 100%|██████████| 391/391 [00:16<00:00, 23.70it/s]

Test set: Average loss: 0.0040, Accuracy: 8290/10000 (82.90%)

EPOCH: 52
Loss=0.524250864982605 Batch_id=390 LR=0.01163 Accuracy=78.00: 100%|██████████| 391/391 [00:16<00:00, 23.78it/s]

Test set: Average loss: 0.0039, Accuracy: 8332/10000 (83.32%)

EPOCH: 53
Loss=0.6757458448410034 Batch_id=390 LR=0.01135 Accuracy=78.41: 100%|██████████| 391/391 [00:17<00:00, 22.68it/s]

Test set: Average loss: 0.0039, Accuracy: 8361/10000 (83.61%)

EPOCH: 54
Loss=0.6256436109542847 Batch_id=390 LR=0.01105 Accuracy=78.41: 100%|██████████| 391/391 [00:16<00:00, 23.97it/s]

Test set: Average loss: 0.0037, Accuracy: 8388/10000 (83.88%)

EPOCH: 55
Loss=0.7842350602149963 Batch_id=390 LR=0.01075 Accuracy=78.30: 100%|██████████| 391/391 [00:16<00:00, 23.75it/s]

Test set: Average loss: 0.0039, Accuracy: 8322/10000 (83.22%)

EPOCH: 56
Loss=0.5691540837287903 Batch_id=390 LR=0.01045 Accuracy=78.60: 100%|██████████| 391/391 [00:16<00:00, 24.05it/s]

Test set: Average loss: 0.0039, Accuracy: 8336/10000 (83.36%)

EPOCH: 57
Loss=0.5649817585945129 Batch_id=390 LR=0.01013 Accuracy=79.01: 100%|██████████| 391/391 [00:16<00:00, 23.21it/s]

Test set: Average loss: 0.0039, Accuracy: 8329/10000 (83.29%)

EPOCH: 58
Loss=0.580344557762146 Batch_id=390 LR=0.00982 Accuracy=79.25: 100%|██████████| 391/391 [00:16<00:00, 23.60it/s]

Test set: Average loss: 0.0038, Accuracy: 8373/10000 (83.73%)

EPOCH: 59
Loss=0.685698926448822 Batch_id=390 LR=0.00949 Accuracy=79.08: 100%|██████████| 391/391 [00:16<00:00, 23.61it/s]

Test set: Average loss: 0.0037, Accuracy: 8462/10000 (84.62%)

EPOCH: 60
Loss=0.6260854601860046 Batch_id=390 LR=0.00917 Accuracy=79.04: 100%|██████████| 391/391 [00:16<00:00, 23.64it/s]

Test set: Average loss: 0.0036, Accuracy: 8489/10000 (84.89%)

EPOCH: 61
Loss=0.7272824048995972 Batch_id=390 LR=0.00884 Accuracy=79.61: 100%|██████████| 391/391 [00:16<00:00, 23.54it/s]

Test set: Average loss: 0.0036, Accuracy: 8448/10000 (84.48%)

EPOCH: 62
Loss=0.5386489629745483 Batch_id=390 LR=0.00851 Accuracy=79.72: 100%|██████████| 391/391 [00:16<00:00, 23.22it/s]

Test set: Average loss: 0.0036, Accuracy: 8464/10000 (84.64%)

EPOCH: 63
Loss=0.5939100980758667 Batch_id=390 LR=0.00817 Accuracy=79.93: 100%|██████████| 391/391 [00:16<00:00, 23.59it/s]

Test set: Average loss: 0.0036, Accuracy: 8505/10000 (85.05%)

EPOCH: 64
Loss=0.6112222075462341 Batch_id=390 LR=0.00784 Accuracy=80.24: 100%|██████████| 391/391 [00:16<00:00, 23.63it/s]

Test set: Average loss: 0.0037, Accuracy: 8440/10000 (84.40%)

EPOCH: 65
Loss=0.629386842250824 Batch_id=390 LR=0.00750 Accuracy=80.49: 100%|██████████| 391/391 [00:16<00:00, 23.63it/s]

Test set: Average loss: 0.0034, Accuracy: 8561/10000 (85.61%)

EPOCH: 66
Loss=0.8118685483932495 Batch_id=390 LR=0.00716 Accuracy=80.46: 100%|██████████| 391/391 [00:16<00:00, 23.15it/s]

Test set: Average loss: 0.0035, Accuracy: 8485/10000 (84.85%)

EPOCH: 67
Loss=0.5720980167388916 Batch_id=390 LR=0.00683 Accuracy=80.64: 100%|██████████| 391/391 [00:16<00:00, 23.97it/s]

Test set: Average loss: 0.0035, Accuracy: 8485/10000 (84.85%)

EPOCH: 68
Loss=0.6564686298370361 Batch_id=390 LR=0.00649 Accuracy=81.00: 100%|██████████| 391/391 [00:16<00:00, 23.57it/s]

Test set: Average loss: 0.0033, Accuracy: 8574/10000 (85.74%)

EPOCH: 69
Loss=0.7067839503288269 Batch_id=390 LR=0.00616 Accuracy=81.27: 100%|██████████| 391/391 [00:16<00:00, 23.97it/s]

Test set: Average loss: 0.0034, Accuracy: 8579/10000 (85.79%)

EPOCH: 70
Loss=0.49096694588661194 Batch_id=390 LR=0.00583 Accuracy=81.05: 100%|██████████| 391/391 [00:16<00:00, 23.26it/s]

Test set: Average loss: 0.0032, Accuracy: 8614/10000 (86.14%)

EPOCH: 71
Loss=0.5617742538452148 Batch_id=390 LR=0.00550 Accuracy=81.90: 100%|██████████| 391/391 [00:16<00:00, 24.04it/s]

Test set: Average loss: 0.0033, Accuracy: 8630/10000 (86.30%)

EPOCH: 72
Loss=0.40431007742881775 Batch_id=390 LR=0.00518 Accuracy=81.87: 100%|██████████| 391/391 [00:16<00:00, 24.19it/s]

Test set: Average loss: 0.0032, Accuracy: 8646/10000 (86.46%)

EPOCH: 73
Loss=0.5194879770278931 Batch_id=390 LR=0.00486 Accuracy=81.94: 100%|██████████| 391/391 [00:16<00:00, 23.93it/s]

Test set: Average loss: 0.0032, Accuracy: 8645/10000 (86.45%)

EPOCH: 74
Loss=0.6543298959732056 Batch_id=390 LR=0.00455 Accuracy=81.93: 100%|██████████| 391/391 [00:16<00:00, 23.04it/s]

Test set: Average loss: 0.0031, Accuracy: 8671/10000 (86.71%)

EPOCH: 75
Loss=0.4472695291042328 Batch_id=390 LR=0.00425 Accuracy=82.34: 100%|██████████| 391/391 [00:16<00:00, 24.03it/s]

Test set: Average loss: 0.0031, Accuracy: 8663/10000 (86.63%)

EPOCH: 76
Loss=0.59834885597229 Batch_id=390 LR=0.00395 Accuracy=82.82: 100%|██████████| 391/391 [00:16<00:00, 23.88it/s]

Test set: Average loss: 0.0031, Accuracy: 8707/10000 (87.07%)

EPOCH: 77
Loss=0.4682682454586029 Batch_id=390 LR=0.00365 Accuracy=82.95: 100%|██████████| 391/391 [00:16<00:00, 23.81it/s]

Test set: Average loss: 0.0031, Accuracy: 8696/10000 (86.96%)

EPOCH: 78
Loss=0.37219640612602234 Batch_id=390 LR=0.00337 Accuracy=83.06: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]

Test set: Average loss: 0.0031, Accuracy: 8699/10000 (86.99%)

EPOCH: 79
Loss=0.2595481276512146 Batch_id=390 LR=0.00309 Accuracy=83.45: 100%|██████████| 391/391 [00:16<00:00, 23.44it/s]

Test set: Average loss: 0.0030, Accuracy: 8722/10000 (87.22%)

EPOCH: 80
Loss=0.44243016839027405 Batch_id=390 LR=0.00282 Accuracy=83.37: 100%|██████████| 391/391 [00:16<00:00, 23.90it/s]

Test set: Average loss: 0.0030, Accuracy: 8735/10000 (87.35%)

EPOCH: 81
Loss=0.4257698059082031 Batch_id=390 LR=0.00256 Accuracy=83.56: 100%|██████████| 391/391 [00:16<00:00, 23.82it/s]

Test set: Average loss: 0.0030, Accuracy: 8711/10000 (87.11%)

EPOCH: 82
Loss=0.322399765253067 Batch_id=390 LR=0.00232 Accuracy=84.06: 100%|██████████| 391/391 [00:16<00:00, 23.92it/s]

Test set: Average loss: 0.0029, Accuracy: 8755/10000 (87.55%)

EPOCH: 83
Loss=0.43780583143234253 Batch_id=390 LR=0.00208 Accuracy=84.37: 100%|██████████| 391/391 [00:16<00:00, 23.14it/s]

Test set: Average loss: 0.0029, Accuracy: 8780/10000 (87.80%)

EPOCH: 84
Loss=0.4165523052215576 Batch_id=390 LR=0.00185 Accuracy=84.53: 100%|██████████| 391/391 [00:16<00:00, 23.89it/s]

Test set: Average loss: 0.0029, Accuracy: 8759/10000 (87.59%)

EPOCH: 85
Loss=0.6352576613426208 Batch_id=390 LR=0.00164 Accuracy=84.55: 100%|██████████| 391/391 [00:16<00:00, 23.78it/s]

Test set: Average loss: 0.0029, Accuracy: 8779/10000 (87.79%)

EPOCH: 86
Loss=0.5346687436103821 Batch_id=390 LR=0.00143 Accuracy=84.79: 100%|██████████| 391/391 [00:16<00:00, 23.80it/s]

Test set: Average loss: 0.0028, Accuracy: 8818/10000 (88.18%)

EPOCH: 87
Loss=0.5774069428443909 Batch_id=390 LR=0.00124 Accuracy=85.09: 100%|██████████| 391/391 [00:17<00:00, 22.92it/s]

Test set: Average loss: 0.0028, Accuracy: 8840/10000 (88.40%)

EPOCH: 88
Loss=0.5059360265731812 Batch_id=390 LR=0.00106 Accuracy=85.15: 100%|██████████| 391/391 [00:16<00:00, 24.03it/s]

Test set: Average loss: 0.0028, Accuracy: 8796/10000 (87.96%)

EPOCH: 89
Loss=0.38224318623542786 Batch_id=390 LR=0.00090 Accuracy=85.30: 100%|██████████| 391/391 [00:16<00:00, 23.78it/s]

Test set: Average loss: 0.0028, Accuracy: 8810/10000 (88.10%)

EPOCH: 90
Loss=0.36603784561157227 Batch_id=390 LR=0.00074 Accuracy=85.59: 100%|██████████| 391/391 [00:16<00:00, 23.69it/s]

Test set: Average loss: 0.0028, Accuracy: 8833/10000 (88.33%)

EPOCH: 91
Loss=0.5548180341720581 Batch_id=390 LR=0.00060 Accuracy=85.76: 100%|██████████| 391/391 [00:16<00:00, 23.26it/s]

Test set: Average loss: 0.0028, Accuracy: 8828/10000 (88.28%)

EPOCH: 92
Loss=0.3807026743888855 Batch_id=390 LR=0.00048 Accuracy=85.80: 100%|██████████| 391/391 [00:16<00:00, 23.86it/s]

Test set: Average loss: 0.0028, Accuracy: 8844/10000 (88.44%)

EPOCH: 93
Loss=0.4605600833892822 Batch_id=390 LR=0.00037 Accuracy=86.02: 100%|██████████| 391/391 [00:16<00:00, 23.55it/s]

Test set: Average loss: 0.0028, Accuracy: 8850/10000 (88.50%)

EPOCH: 94
Loss=0.3903403878211975 Batch_id=390 LR=0.00027 Accuracy=86.13: 100%|██████████| 391/391 [00:16<00:00, 23.72it/s]

Test set: Average loss: 0.0028, Accuracy: 8815/10000 (88.15%)

EPOCH: 95
Loss=0.3735284209251404 Batch_id=390 LR=0.00019 Accuracy=85.99: 100%|██████████| 391/391 [00:16<00:00, 23.78it/s]

Test set: Average loss: 0.0028, Accuracy: 8845/10000 (88.45%)

EPOCH: 96
Loss=0.5211338996887207 Batch_id=390 LR=0.00012 Accuracy=85.96: 100%|██████████| 391/391 [00:16<00:00, 23.54it/s]

Test set: Average loss: 0.0027, Accuracy: 8853/10000 (88.53%)

EPOCH: 97
Loss=0.5674200654029846 Batch_id=390 LR=0.00007 Accuracy=86.15: 100%|██████████| 391/391 [00:16<00:00, 23.55it/s]

Test set: Average loss: 0.0029, Accuracy: 8848/10000 (88.48%)

EPOCH: 98
Loss=0.42610183358192444 Batch_id=390 LR=0.00003 Accuracy=85.97: 100%|██████████| 391/391 [00:16<00:00, 23.74it/s]

Test set: Average loss: 0.0028, Accuracy: 8847/10000 (88.47%)

EPOCH: 99
Loss=0.5177533626556396 Batch_id=390 LR=0.00001 Accuracy=86.19: 100%|██████████| 391/391 [00:16<00:00, 23.74it/s]

Test set: Average loss: 0.0027, Accuracy: 8864/10000 (88.64%)

EPOCH: 100
Loss=0.3460105061531067 Batch_id=390 LR=0.00000 Accuracy=86.15: 100%|██████████| 391/391 [00:16<00:00, 23.01it/s]

Test set: Average loss: 0.0028, Accuracy: 8869/10000 (88.69%)
```

## Accuracy and Loss Graphs
![image](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session9/Loss_accuracy_graph.png)
