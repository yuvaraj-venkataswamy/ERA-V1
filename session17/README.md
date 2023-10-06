# ERA V1 - Session 17 Assignment

## ViT

### Model
```
============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
ViT (ViT)                                                    [32, 3, 224, 224]    [32, 3]              152,064              True
├─PatchEmbedding (patch_embedding)                           [32, 3, 224, 224]    [32, 196, 768]       --                   True
│    └─Conv2d (patcher)                                      [32, 3, 224, 224]    [32, 768, 14, 14]    590,592              True
│    └─Flatten (flatten)                                     [32, 768, 14, 14]    [32, 768, 196]       --                   --
├─Dropout (embedding_dropout)                                [32, 197, 768]       [32, 197, 768]       --                   --
├─Sequential (transformer_encoder)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    └─TransformerEncoderBlock (0)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (1)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (2)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (3)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (4)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (5)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (6)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (7)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (8)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (9)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (10)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (11)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttentionBlock (msa_block)          [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
├─Sequential (classifier)                                    [32, 768]            [32, 3]              --                   True
│    └─LayerNorm (0)                                         [32, 768]            [32, 768]            1,536                True
│    └─Linear (1)                                            [32, 768]            [32, 3]              2,307                True
============================================================================================================================================
Total params: 85,800,963
Trainable params: 85,800,963
Non-trainable params: 0
Total mult-adds (G): 5.52
============================================================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3292.20
Params size (MB): 229.20
Estimated Total Size (MB): 3540.67
============================================================================================================================================
```

### Training logs

```
Epoch: 1 | train_loss: 0.8303 | train_acc: 0.6758 | test_loss: 0.5308 | test_acc: 0.8362
Epoch: 2 | train_loss: 0.3468 | train_acc: 0.9453 | test_loss: 0.3112 | test_acc: 0.8873
Epoch: 3 | train_loss: 0.2157 | train_acc: 0.9648 | test_loss: 0.2508 | test_acc: 0.8873
```

### Loss and accuracy
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session17/images/vit.png)

### Output
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session17/images/vit_pred.png)


Reference
1. https://arxiv.org/abs/1810.04805
