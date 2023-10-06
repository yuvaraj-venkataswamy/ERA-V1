# ERA V1 - Session 19 Assignment

## CLIP Models 
CLIP or Contrastive Language - Image Pre-training, deviates from the standard practice of fine-tuning a pre-trained model by taking the path of zero-shot learning. Zero-shot learning is the ability of the model to perform tasks that it was not explicitly programmed to do. 

SAM's architecture comprises three components that work together to return a valid segmentation mask:
1. An image encoder to generate one-time image embeddings.
2. A prompt encoder that embeds the prompts.
3. A lightweight mask decoder that combines the embeddings from the prompt and image encoders.

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session19/Images/segment-anything-pipeline.gif)

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session19/Images/model_architecture.png)

## dataset
Flicker 8K dataset is used.

## Training

```
Epoch: 1
100%
1012/1012 [21:08<00:00, 2.58it/s, lr=0.0001, train_loss=1.92]
100%
253/253 [01:14<00:00, 5.25it/s, valid_loss=1.25]
/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
Saved Best Model!
Epoch: 2
100%
1012/1012 [07:58<00:00, 2.64it/s, lr=0.0001, train_loss=0.63]
100%
253/253 [01:12<00:00, 5.86it/s, valid_loss=1.04]
Saved Best Model!
Epoch: 3
100%
1012/1012 [07:58<00:00, 2.63it/s, lr=0.0001, train_loss=0.381]
100%
253/253 [01:11<00:00, 6.07it/s, valid_loss=1.02]
Saved Best Model!
Epoch: 4
100%
1012/1012 [07:56<00:00, 2.66it/s, lr=0.0001, train_loss=0.272]
100%
253/253 [01:12<00:00, 6.06it/s, valid_loss=0.995]
Saved Best Model!
```

## Finding Matches
```
find_matches(model,
             image_embeddings,
             query="a group of people dancing in a party",
             image_filenames=valid_df['image'].values,
             n=9)
```
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session19/Images/dance.png)


## Reference
1. https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2
2. https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
