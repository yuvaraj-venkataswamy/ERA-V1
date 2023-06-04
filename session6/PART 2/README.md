# TASK 2
## PROBLEM STATEMENT
Buiding an architecture to achieve,
1. 99.4% validation accuracy
2. Less than 20k Parameters
3. Less than 20 Epochs
4. Have used BN, Dropout,
5. (Optional): a Fully connected layer, have used GAP. 

## MODEL ARCHITECTURE
- Batch Normalization: 
      Applied Batch Normalization at every layer to normalize the inputs to the next layer.
- Drop-out:
      Used dropout after batchnormlization every time to regularize our input and to prevent our model from overfitting.
- Max Pooling:
      Max pooling was applied after it reached a receptive field of 7. At this point, the model captures lines/edges, curves in its entirety and can now combine features into higher level objects.
- 1x1 Convolution:
      Together with the Max Pooling layer, this makes up our transition layer. This is primarily to reduce the number of channels while effectively combining features from all channels.
- Global Average Pooling
      This layer is introduced when our channel dimension is 5x5. Not only does this translate convolutional structure to linear structure, it has the added advantage of having less parameters to compute and since it doesn't have to learn anything, it helps avoid overfitting. 

## DATA PREPARATION


## TRAINING AND TESTING


## RESULTS
