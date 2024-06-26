# ERA V1 - Session 18 Assignment

## UNET Architecture
UNET is a popular neural network architecture for image segmentation tasks.

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session18/images/unet.gif)

1. It consists of a contracting path, which gradually reduces the spatial resolution of the input image while increasing the number of channels, and an expanding path, which gradually recovers the original resolution while decreasing the number of channels. 
2. The two paths are connected by skip connections, which allow the network to use information from the contracting path to better localize the segmentation in the expanding path. 
3. UNET has shown excellent performance on a wide range of image segmentation tasks, and is often used as a baseline architecture for new segmentation tasks. 

## UNet Model
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 128, 128, 3) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 128, 128, 32) 896         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 128, 32) 9248        conv2d[0][0]                     
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 128, 128, 32) 128         conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 64, 64, 32)   0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, 64, 64)   18496       max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 64, 64)   36928       conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 64, 64, 64)   256         conv2d_3[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 32, 32, 64)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 128)  73856       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 128)  147584      conv2d_4[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 128)  512         conv2d_5[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 16, 16, 128)  0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 16, 16, 256)  295168      max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 16, 16, 256)  590080      conv2d_6[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 16, 16, 256)  1024        conv2d_7[0][0]                   
__________________________________________________________________________________________________
dropout (Dropout)               (None, 16, 16, 256)  0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 8, 8, 256)    0           dropout[0][0]                    
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 8, 8, 512)    1180160     max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 8, 8, 512)    2359808     conv2d_8[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 8, 8, 512)    2048        conv2d_9[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 8, 8, 512)    0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 16, 16, 256)  1179904     dropout_1[0][0]                  
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 16, 16, 512)  0           conv2d_transpose[0][0]           
                                                                 dropout[0][0]                    
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 256)  1179904     concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 256)  590080      conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 32, 32, 128)  295040      conv2d_11[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 32, 32, 256)  0           conv2d_transpose_1[0][0]         
                                                                 batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 128)  295040      concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 32, 32, 128)  147584      conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 64, 64, 64)   73792       conv2d_13[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 64, 64, 128)  0           conv2d_transpose_2[0][0]         
                                                                 batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 64, 64, 64)   73792       concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 64, 64, 64)   36928       conv2d_14[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 128, 128, 32) 18464       conv2d_15[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 128, 128, 64) 0           conv2d_transpose_3[0][0]         
                                                                 batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 128, 128, 32) 18464       concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 128, 128, 32) 9248        conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 128, 128, 32) 9248        conv2d_17[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 128, 128, 3)  99          conv2d_18[0][0]                  
==================================================================================================
Total params: 8,643,779
Trainable params: 8,641,795
Non-trainable params: 1,984
__________________________________________________________________________________________________
```

## Training and validation

```
Epoch 1/20
185/185 [==============================] - 3177s 17s/step - loss: 0.7669 - accuracy: 0.6656 - val_loss: 0.6085 - val_accuracy: 0.7550
Epoch 2/20
185/185 [==============================] - 3177s 17s/step - loss: 0.5510 - accuracy: 0.7811 - val_loss: 0.4746 - val_accuracy: 0.8169
Epoch 3/20
185/185 [==============================] - 3171s 17s/step - loss: 0.4507 - accuracy: 0.8266 - val_loss: 0.4812 - val_accuracy: 0.8157
Epoch 4/20
185/185 [==============================] - 3210s 17s/step - loss: 0.3991 - accuracy: 0.8475 - val_loss: 0.3889 - val_accuracy: 0.8493
Epoch 5/20
185/185 [==============================] - 3193s 17s/step - loss: 0.3745 - accuracy: 0.8578 - val_loss: 0.3565 - val_accuracy: 0.8643
Epoch 6/20
185/185 [==============================] - 3193s 17s/step - loss: 0.3408 - accuracy: 0.8707 - val_loss: 0.3621 - val_accuracy: 0.8596
Epoch 7/20
185/185 [==============================] - 3190s 17s/step - loss: 0.3322 - accuracy: 0.8747 - val_loss: 0.3892 - val_accuracy: 0.8570
Epoch 8/20
185/185 [==============================] - 3188s 17s/step - loss: 0.3093 - accuracy: 0.8834 - val_loss: 0.3279 - val_accuracy: 0.8758
Epoch 9/20
185/185 [==============================] - 3174s 17s/step - loss: 0.2928 - accuracy: 0.8897 - val_loss: 0.3363 - val_accuracy: 0.8797
Epoch 10/20
185/185 [==============================] - 3179s 17s/step - loss: 0.2802 - accuracy: 0.8941 - val_loss: 0.3271 - val_accuracy: 0.8793
Epoch 11/20
185/185 [==============================] - 3160s 17s/step - loss: 0.2682 - accuracy: 0.8989 - val_loss: 0.3298 - val_accuracy: 0.8803
Epoch 12/20
185/185 [==============================] - 3174s 17s/step - loss: 0.2588 - accuracy: 0.9023 - val_loss: 0.3352 - val_accuracy: 0.8796
Epoch 13/20
185/185 [==============================] - 3172s 17s/step - loss: 0.2515 - accuracy: 0.9049 - val_loss: 0.3621 - val_accuracy: 0.8706
Epoch 14/20
185/185 [==============================] - 3150s 17s/step - loss: 0.2457 - accuracy: 0.9071 - val_loss: 0.3197 - val_accuracy: 0.8819
Epoch 15/20
185/185 [==============================] - 3143s 17s/step - loss: 0.2350 - accuracy: 0.9112 - val_loss: 0.3352 - val_accuracy: 0.8804
Epoch 16/20
185/185 [==============================] - 3151s 17s/step - loss: 0.2352 - accuracy: 0.9107 - val_loss: 0.3212 - val_accuracy: 0.8877
Epoch 17/20
185/185 [==============================] - 3147s 17s/step - loss: 0.2341 - accuracy: 0.9113 - val_loss: 0.3387 - val_accuracy: 0.8790
Epoch 18/20
185/185 [==============================] - 3144s 17s/step - loss: 0.2257 - accuracy: 0.9146 - val_loss: 0.3059 - val_accuracy: 0.8920
Epoch 19/20
185/185 [==============================] - 3148s 17s/step - loss: 0.2126 - accuracy: 0.9191 - val_loss: 0.3085 - val_accuracy: 0.8909
Epoch 20/20
185/185 [==============================] - 3152s 17s/step - loss: 0.2250 - accuracy: 0.9142 - val_loss: 0.3081 - val_accuracy: 0.8906
```

## Training accuracy and loss

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session18/images/Loss_accuracy_graph_S18.png)

## Predicted Segmentation

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session18/images/output.png)

## VAE MNIST 

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session18/images/MNISTsample1_.png)

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session18/images/MNISTsample2_.png)


## Reference
1. https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406
2. https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset
