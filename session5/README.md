# Session 5 Assignment

## Folder Structure
```
|── model.py
|── utils.py
|── S5.ipynb     
|── README.md   
```

## 1. model.py
model.py file consists two major functions defined below,
- We define our own class `class Net(nn.Module)` and we inharite nn.Module which is Base class for all neural network modules. Then we define initialize function `__init__` after we inherite all the functionality of nn.Module in our class `super(Net, self).__init__()`. After that we start building our model.
- The `forward()` function defines the process of calculating the output using the given layers and activation functions.

## 2. utils.py
utils.py file consists training and testing functions which are described below,
- `train()` function defines the process of training the dataset.
- `test()` function is used for testing the dataset.

## MNIST-Classification

### Sample MNIST dataset
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session5/image/sample_dataset.png)

### Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

### Training logs
```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.0503 Batch_id=468 Accuracy=86.69: 100%|██████████| 469/469 [00:37<00:00, 12.56it/s]
Test set: Average loss: 0.0644, Accuracy: 9800/10000 (98.00%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.0352 Batch_id=468 Accuracy=97.23: 100%|██████████| 469/469 [00:30<00:00, 15.56it/s]
Test set: Average loss: 0.0390, Accuracy: 9888/10000 (98.88%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.0160 Batch_id=468 Accuracy=98.05: 100%|██████████| 469/469 [00:27<00:00, 17.23it/s]
Test set: Average loss: 0.0432, Accuracy: 9854/10000 (98.54%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.0826 Batch_id=468 Accuracy=98.33: 100%|██████████| 469/469 [00:28<00:00, 16.64it/s]
Test set: Average loss: 0.0300, Accuracy: 9896/10000 (98.96%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.0316 Batch_id=468 Accuracy=98.60: 100%|██████████| 469/469 [00:27<00:00, 17.19it/s]
Test set: Average loss: 0.0254, Accuracy: 9913/10000 (99.13%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.0162 Batch_id=468 Accuracy=98.79: 100%|██████████| 469/469 [00:27<00:00, 17.25it/s]
Test set: Average loss: 0.0265, Accuracy: 9905/10000 (99.05%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.0972 Batch_id=468 Accuracy=98.86: 100%|██████████| 469/469 [00:26<00:00, 17.37it/s]
Test set: Average loss: 0.0260, Accuracy: 9912/10000 (99.12%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 8
Train: Loss=0.0891 Batch_id=468 Accuracy=99.05: 100%|██████████| 469/469 [00:28<00:00, 16.47it/s]
Test set: Average loss: 0.0215, Accuracy: 9924/10000 (99.24%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 9
Train: Loss=0.0458 Batch_id=468 Accuracy=99.05: 100%|██████████| 469/469 [00:27<00:00, 17.22it/s]
Test set: Average loss: 0.0220, Accuracy: 9921/10000 (99.21%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 10
Train: Loss=0.0021 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:27<00:00, 17.23it/s]
Test set: Average loss: 0.0235, Accuracy: 9929/10000 (99.29%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 11
Train: Loss=0.0176 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:27<00:00, 17.31it/s]
Test set: Average loss: 0.0224, Accuracy: 9929/10000 (99.29%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 12
Train: Loss=0.0144 Batch_id=468 Accuracy=99.24: 100%|██████████| 469/469 [00:27<00:00, 17.23it/s]
Test set: Average loss: 0.0209, Accuracy: 9929/10000 (99.29%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 13
Train: Loss=0.0270 Batch_id=468 Accuracy=99.25: 100%|██████████| 469/469 [00:27<00:00, 16.95it/s]
Test set: Average loss: 0.0239, Accuracy: 9916/10000 (99.16%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 14
Train: Loss=0.0005 Batch_id=468 Accuracy=99.33: 100%|██████████| 469/469 [00:27<00:00, 17.32it/s]
Test set: Average loss: 0.0210, Accuracy: 9932/10000 (99.32%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 15
Train: Loss=0.0023 Batch_id=468 Accuracy=99.34: 100%|██████████| 469/469 [00:27<00:00, 17.32it/s]
Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 0.0000e+00.
Epoch 16
Train: Loss=0.0019 Batch_id=468 Accuracy=99.47: 100%|██████████| 469/469 [00:27<00:00, 17.25it/s]
Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 0.0000e+00.
Epoch 17
Train: Loss=0.0112 Batch_id=468 Accuracy=99.53: 100%|██████████| 469/469 [00:27<00:00, 17.32it/s]
Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 0.0000e+00.
Epoch 18
Train: Loss=0.0060 Batch_id=468 Accuracy=99.52: 100%|██████████| 469/469 [00:27<00:00, 17.30it/s]
Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 0.0000e+00.
Epoch 19
Train: Loss=0.0032 Batch_id=468 Accuracy=99.47: 100%|██████████| 469/469 [00:27<00:00, 17.37it/s]
Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 0.0000e+00.
Epoch 20
Train: Loss=0.0052 Batch_id=468 Accuracy=99.48: 100%|██████████| 469/469 [00:27<00:00, 16.91it/s]
Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 0.0000e+00.
```
### Accuracy and Loss plots
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session5/image/accuracy_loss_plot.png)
