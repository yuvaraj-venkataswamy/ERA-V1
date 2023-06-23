# -*- coding: utf-8 -*-
"""Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qirLNmEcKoX4GUbys-JUaqZfdS6HyTPb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.03
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Dropout(dropout_value)
        )

    
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), stride=2), 
            nn.MaxPool2d(2, 2),
        ) 

        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Dropout(dropout_value),

            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding=1, bias = False),   
            nn.ReLU(),
            nn.GroupNorm(4, 32),
            nn.Dropout2d(dropout_value),
        ) 

       
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=2), 
            nn.MaxPool2d(2, 2),
        )  


        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(4, 32),
            nn.Dropout(dropout_value),

            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(2, 64),
            nn.Dropout(dropout_value),
        )


        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=2), 
            nn.ReLU(),
        ) 
        


        # GAP
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) 
        self.convblock4 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)

        x = self.convblock2(x)
        x = self.trans2(x)

        x = self.convblock3(x)
        x = self.trans3(x)

        x = self.gap(x) 
        x = self.convblock4(x)

        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)