# Session 7 - In-Depth Coding Practice Assignment
## Objective:
- 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.

## Model: 1 Basic structure
### Target
Construct `Basic Structure` of model with the prior knowledge that
- Structures in MNIST images are visible at a level of 5x5, i.e Max pooling must be performed at receptive field of 5x5.
- Go up to 32 channels to reduce total number of parameters.
- No Batch Normalization, Augmentation or Regularization to be used.
### Results
- `Total parameters: 13,584`
- No dropout, Batch normalization or additional regularization used
- Best training accuracy: 99.21% at 15th Epoch
- Best testing accuracy: 99.01% at 14th Epoch
### Analysis
- Both training and testing accuracy gradually increase together up to ~99%
- Overfitting is minimal considering low difference between testing and training accuracies (except in the last 2 epochs).
- Since the base model performs well without batch normalization or other significant additions, we can further reduce the number of parameters to improve model efficiency.

## Model: 2 Parameter Reduction
### Target
- Reduce total number of parameters to below 10k by reducing number of channels in expansions after image input and post max-pooling.
- Maintain max-pooling step at RF of 5x5
- Number of channels for conv layers to be reduced
- Additional parameter expansion to be added post GAP layer (will allow for additional learning)
### Results
- `Total Paratemers: 7,400`
- Best training accuracy: 98.79% at 15th Epoch
- B#est testing accuracy: 98.72% at 14th Epoch
## Analysis
- Model learns slowly, and slight underfitting is present.
- Accuracy convergence can be made faster with the addition of Batch Normalization, which we will do in the next step.

## Model: 3 Batch Normalization
### Target
- Add `Batch Normalization` to improve rate of convergence and overall accuracy after 15 epochs.
### Results
- Total parameters: 7,624
- Best Training Accuracy: 99.32% at 15th Epoch
- Best Testing Accuracy: 99.28% at 14th Epoch
### Analysis
- Using of Batch Normalization the accuracy increase through the epochs.
- Very slight overfitting is observed with a difference of ~0.05 between training and testing accuracies in the last 3 epochs.

## Model: 4 Adding Dropout
### Target
- Add dropout to all convolution layers (except last one) to reduce overfitting. Reduction in accuracy is expected.
### Results
- `Total parameters: 7,624`
- Best Training Accuracy: 98.93% - 14th Epoch
- Best Testing Accuracy: 99.22% - 13th Epoch
### Analysis
- Overfitting is no longer present, and slight underfitting is present.
- Overall accuracy has reduced which is expected considering dropout has been added.

## Model: 5 Data Augmentations
### Target
- Add data augmentations to improve overall testing accuracy.
- Based on analysis of wrong predictions in the last step, random rotations, 
- Random affine transormations and Color jitter augmentations have been added.
### Results
- `Total parameters:` 7,624
- Best Training Accuracy: 97.63% at 15th Epoch
- Best Testing Accuracy: 99.34% at 13th Epoch
### Analysis
- Underfitting is clearly evident, training accuracy is consistently lower than testing accuracy.

## Model: 6 Learning Rate (LR) Scheduling
### Target
- Add learning rate (LR) scheduling to improve overall accuracy with consistant of 99.4% for atleast 2 epoch.
### Results
- `Total parameters: 7,624`
- Best Training Accuracy: 97.60% - 15th Epoch
- Best Testing Accuracy: 99.48% - 11th Epoch
### Analysis
- Lambda scheduling with lambda set to 0.65^epoch helps us achieve our target of 99.4% testing accuracy starting from the 11th Epoch.

## Final Model
```

```

Operation	nin	in_ch	out_ch	padding	kernel	stride	nout	jin	jout	rin	rout
Convolution	28	1	8	0	3	1	26	1	1	1	3
Convolution	26	8	16	0	3	1	24	1	1	3	5
Max-Pooling	24	16	16	0	2	2	12	1	2	5	6
Convolution	12	16	8	0	1	1	12	2	2	6	6
Convolution	12	8	16	0	3	1	10	2	2	6	10
Convolution	10	16	24	0	3	1	8	2	2	10	14
GAP	8	24	24	0	8	1	1	2	2	14	28
Convolution	1	24	32	0	1	1	1	2	2	28	28
Convolution	1	32	16	0	1	1	1	2	2	28	28
Convolution	1	16	10	0	1	1	1	2	2	28	28
![image](https://github.com/yuvaraj-venkataswamy/ERA-V1/assets/44864608/64475651-b060-4be9-9b11-e402916d20a1)


