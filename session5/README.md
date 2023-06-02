# ERA V1 - Session 5 Assignment
   This task provides the folder structure of Neural Network classification using MNIST dataset and describes each file inside the folder.

## Folder Structure
```
session5
   |── model.py
   |── utils.py
   |── S5.ipynb     
   |── README.md   
```

### 1. model.py
   model.py file consists two major functions defined below,
   - We define our own class `class Net(nn.Module)` and then we initialize `__init__` function after we inherite all the functionality of nn.Module in our class   `super(Net, self).__init__()`. After that we start building our model.
   - The `forward()` function defines the process of calculating the output using the given layers and activation functions.

### 2. utils.py
utils.py file consists training and testing functions which are described below,
- `train()` function defines the process of training the dataset.
- `test()` function is used for testing the dataset.

### 3. S5.ipynb
This is the main notebook file which consists below process,
#### a. import packages: 
- We are importing all neccessary libraries and functions files such as model.py and utils.py. 

#### b. Load Model:
-  In the next step, we have to check the Cuda device which adds support for CUDA tensor types, that implement the same function as CPU tensors, but they utilize GPUs for computation. Then, call the `model.py` function and load model to available device.
- `summary()` function Summarize the given PyTorch model which summarized information includes Layer names, input/output shapes, kernel shape, No. of parameters.
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
#### c. Preparing the Dataset
- `batch_size`: batch size is the number of images (here, 128 images) we want to read in one go.
- `torch.utils.data.DataLoader`: we make Data iterable by loading it to a loader.
- `datasets.MNIST`: Downloading the MNIST dataset for training and testing at path ../data.
- `transform=transforms.Compose`: Composes several transforms together. This transform does not support torchscript.
- `transforms.ToTensor()`: This converts the image into numbers, that are understandable by the system. It separates the image into three color channels (separate images): red, green & blue. Then it converts the pixels of each image to the brightness of their color between 0 and 255. These values are then scaled down to a range between 0 and 1. The image is now a Torch Tensor.
- `transforms.Normalize((0.1307,), (0.3081,))`: This normalizes the tensor with a mean (0.1307,) and standard deviation (0.3081,) which goes as the two parameters respectively.
- `shuffle=True`: Shuffle the training data to make it independent of the order by making it a True.
- Sample MNIST dataset

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/images/sample_dataset.png)

#### d. Training and Testing
- The neural network iterates over the training set and updates the weights. We make use of `optim.SGD` which is a Stochastic Gradient Descent (SGD) provided by PyTorch to optimize the model.
- After each epoch the logs are provided below,
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
#### e. Accuracy and Loss plots
- In each epoch (number of times we iterate over the training set), we will be seeing a gradual decrease in training loss.
- The below figure provides training and testing loss as well as accuracy of the built model.

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/images/accuracy_loss_plot.png)
