# TASK 2
## PROBLEM STATEMENT
Buiding an architecture to achieve,
1. 99.4% validation accuracy
2. Less than 20k Parameters
3. Less than 20 Epochs
4. Have used BN, Dropout,
5. (Optional): a Fully connected layer, have used GAP. 

## MODEL ARCHITECTURE
Constructed the model in such a way that the number of parameters shoul be less than 20k.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
         Dropout2d-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           2,304
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
            Conv2d-8           [-1, 32, 22, 22]           4,608
              ReLU-9           [-1, 32, 22, 22]               0
      BatchNorm2d-10           [-1, 32, 22, 22]              64
        Dropout2d-11           [-1, 32, 22, 22]               0
           Conv2d-12           [-1, 16, 22, 22]             528
             ReLU-13           [-1, 16, 22, 22]               0
        MaxPool2d-14           [-1, 16, 11, 11]               0
           Conv2d-15             [-1, 16, 9, 9]           2,304
             ReLU-16             [-1, 16, 9, 9]               0
      BatchNorm2d-17             [-1, 16, 9, 9]              32
        Dropout2d-18             [-1, 16, 9, 9]               0
           Conv2d-19             [-1, 16, 9, 9]           2,304
             ReLU-20             [-1, 16, 9, 9]               0
      BatchNorm2d-21             [-1, 16, 9, 9]              32
        Dropout2d-22             [-1, 16, 9, 9]               0
           Conv2d-23             [-1, 16, 7, 7]           2,304
             ReLU-24             [-1, 16, 7, 7]               0
      BatchNorm2d-25             [-1, 16, 7, 7]              32
        Dropout2d-26             [-1, 16, 7, 7]               0
           Conv2d-27             [-1, 32, 5, 5]           4,608
             ReLU-28             [-1, 32, 5, 5]               0
      BatchNorm2d-29             [-1, 32, 5, 5]              64
        Dropout2d-30             [-1, 32, 5, 5]               0
           Conv2d-31             [-1, 10, 5, 5]             320
        AvgPool2d-32             [-1, 10, 1, 1]               0
================================================================
Total params: 19,712
Trainable params: 19,712
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.28
Params size (MB): 0.08
Estimated Total Size (MB): 1.35
----------------------------------------------------------------
```
- *Batch Normalization*: 
      Applied Batch Normalization at every layer to normalize the inputs to the next layer.
- *Drop-out*:
      Used dropout after batchnormlization every time to regularize our input and to prevent our model from overfitting.
- *Max Pooling*:
      Max pooling was applied after it reached a receptive field of 7. At this point, the model captures lines/edges, curves in its entirety and can now combine features into higher level objects.
- *1x1 Convolution*:
      Together with the Max Pooling layer, this makes up our transition layer. This is primarily to reduce the number of channels while effectively combining features from all channels.
- *Global Average Pooling*:
      This layer is introduced when our channel dimension is 5x5. Not only does this translate convolutional structure to linear structure, it has the added advantage of having less parameters to compute and since it doesn't have to learn anything, it helps avoid overfitting. 

## DATA PREPARATION
- The MNIST database is a large database of handwritten digits that is commonly used for training various model architectures.
![alt_text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/images/sample_dataset_S6.png) 

## TRAINING AND TESTING


## RESULTS
