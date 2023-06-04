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
- In 13th epoch, I have reached 99.4% of accuracy.
- Training and testing logs are provided below,
```
Epoch 1 : 
  0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-8-57e2fa97f87c>:77: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
loss=0.21440552175045013 batch_id=468: 100%|██████████| 469/469 [01:07<00:00,  6.98it/s]

Test set: Average loss: 0.0560, Accuracy: 9829/10000 (98.29%)


Epoch 2 : 
loss=0.050974104553461075 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.39it/s]

Test set: Average loss: 0.0389, Accuracy: 9878/10000 (98.78%)


Epoch 3 : 
loss=0.1348286122083664 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.39it/s]

Test set: Average loss: 0.0315, Accuracy: 9905/10000 (99.05%)


Epoch 4 : 
loss=0.07328923791646957 batch_id=468: 100%|██████████| 469/469 [01:04<00:00,  7.33it/s]

Test set: Average loss: 0.0275, Accuracy: 9914/10000 (99.14%)


Epoch 5 : 
loss=0.06996336579322815 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.40it/s]

Test set: Average loss: 0.0261, Accuracy: 9924/10000 (99.24%)


Epoch 6 : 
loss=0.0316704697906971 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.38it/s]

Test set: Average loss: 0.0235, Accuracy: 9928/10000 (99.28%)


Epoch 7 : 
loss=0.08884567767381668 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.35it/s]

Test set: Average loss: 0.0229, Accuracy: 9925/10000 (99.25%)


Epoch 8 : 
loss=0.01366227027028799 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.42it/s]

Test set: Average loss: 0.0207, Accuracy: 9934/10000 (99.34%)


Epoch 9 : 
loss=0.04848036542534828 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.39it/s]

Test set: Average loss: 0.0202, Accuracy: 9939/10000 (99.39%)


Epoch 10 : 
loss=0.060379188507795334 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.36it/s]

Test set: Average loss: 0.0198, Accuracy: 9941/10000 (99.41%)


Epoch 11 : 
loss=0.033541832119226456 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.36it/s]

Test set: Average loss: 0.0189, Accuracy: 9930/10000 (99.30%)


Epoch 12 : 
loss=0.005692657083272934 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.36it/s]

Test set: Average loss: 0.0215, Accuracy: 9936/10000 (99.36%)


Epoch 13 : 
loss=0.03460043668746948 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.35it/s]

Test set: Average loss: 0.0195, Accuracy: 9940/10000 (99.40%)


Epoch 14 : 
loss=0.08565816283226013 batch_id=468: 100%|██████████| 469/469 [01:04<00:00,  7.32it/s]

Test set: Average loss: 0.0172, Accuracy: 9944/10000 (99.44%)


Epoch 15 : 
loss=0.014360684901475906 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.35it/s]

Test set: Average loss: 0.0169, Accuracy: 9941/10000 (99.41%)


Epoch 16 : 
loss=0.012136665172874928 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.35it/s]

Test set: Average loss: 0.0168, Accuracy: 9945/10000 (99.45%)


Epoch 17 : 
loss=0.053760722279548645 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.33it/s]

Test set: Average loss: 0.0186, Accuracy: 9944/10000 (99.44%)


Epoch 18 : 
loss=0.01371628325432539 batch_id=468: 100%|██████████| 469/469 [01:04<00:00,  7.27it/s]

Test set: Average loss: 0.0181, Accuracy: 9941/10000 (99.41%)


Epoch 19 : 
loss=0.03255043923854828 batch_id=468: 100%|██████████| 469/469 [01:03<00:00,  7.38it/s]

Test set: Average loss: 0.0172, Accuracy: 9945/10000 (99.45%)
```
## CONCLUSION
- The architecture is built using batch normalization, drop out, fully connected layer and Global Average Pooling (GAP).
- The results attain 99.4% of accuracy within 20 epoch and number of parameters reached over 20k.

## REFERENCES
1. The architecture is build based based on this link, https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99
