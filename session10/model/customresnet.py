import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np



class ConvBNBlock(nn.Module):
    

    def __init__(self, in_planes, planes, stride=1, p=0.0):
        super(ConvBNBlock, self).__init__()
        self.dropout_prob=p
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.drop_out = nn.Dropout2d(p=self.dropout_prob)        

    def forward(self, x):
        out = F.relu(self.drop_out(self.bn(self.conv(x))))
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, p=0.0):
        super(TransitionBlock, self).__init__()
        self.p = p
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(planes)
        self.drop_out = nn.Dropout2d(p=self.p)    
       

    def forward(self, x):
        x = F.relu(self.drop_out(self.bn(self.max_pool(self.conv(x)))))
         
        return x


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, p=0.0):
        super(ResBlock, self).__init__()
        self.p = p
        self.transition_block = TransitionBlock(in_planes, planes, stride, p)
        self.conv_block1 = ConvBNBlock(planes, planes, stride, p)
        self.conv_block2 = ConvBNBlock(planes, planes, stride, p)
  

    def forward(self, x):
        x = self.transition_block(x)
        r = self.conv_block2(self.conv_block1(x))
        out = x + r
        
         
        return out


class CustomResNet(nn.Module):
    def __init__(self, p=0.0, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_planes = 64
        self.p = p

        self.conv = ConvBNBlock(3, 64, 1, p)
        self.layer1 = ResBlock(64,128, 1, p)
        self.layer2 = TransitionBlock(128, 256, 1, p)
        self.layer3 =  ResBlock(256,512, 1, p)
        self.max_pool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out)

def train_test_loader(BATCH_SIZE,get_train_transform, get_test_transform,get_train_loader, get_test_loader, get_classes):
  transform_train = get_train_transform()
  transform_test = get_test_transform()

  trainloader = get_train_loader(BATCH_SIZE, transform_train)
  testloader = get_test_loader(BATCH_SIZE, transform_test)
  classes = get_classes()
  return trainloader, testloader, classes,transform_train,transform_test
  
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def get_model(CustomResNet):
  model =  CustomResNet()
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model.device = torch.device("cuda" if use_cuda else "cpu")
  model =  CustomResNet().to(model.device)
  return model
 
def find_lr_value(model,EPOCHS_TO_TRY,max_lr_list,test_accuracy_list,PATH_BASE_MODEL,max_lr_finder_schedule,BATCH_SIZE,train,test,trainloader,testloader):
  for lr_value in max_lr_list:
      model.load_state_dict(torch.load(PATH_BASE_MODEL))
      optimizer = optim.SGD(model.parameters(), lr=lr_value/10, momentum=0.9)

      lr_finder_schedule = lambda t: np.interp([t], [0, EPOCHS_TO_TRY], [lr_value/10,  lr_value])[0]
      lr_finder_lambda = lambda it: lr_finder_schedule(it * BATCH_SIZE/50176)
      max_lr_finder = max_lr_finder_schedule(no_of_images=50176, batch_size=BATCH_SIZE, base_lr=lr_value/10, max_lr=lr_value, total_epochs=5)
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[max_lr_finder])
      train_losses = []
      test_losses = []
      train_acc = []
      test_acc = []
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")      
      for epoch in range(EPOCHS_TO_TRY):
          print("MAX LR:" ,lr_value, " EPOCH:", (epoch+1))        
          train(model, device, trainloader, optimizer, epoch, train_losses,scheduler,train_acc,True )
          test(model, device, testloader, test_losses, test_acc)
      t_acc = test_acc[-1]
      test_accuracy_list.append(t_acc)
      print(" For Max LR: ", lr_value, " Test Accuracy: ", t_acc)
      
def train_model(best_test_accuracy,EPOCHS, model,trainloader,testloader,optimizer,train,test,train_losses,test_losses,scheduler,train_acc,test_acc,PATH):
  for epoch in range(EPOCHS):
      print("EPOCH:", (epoch+1))
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")  
      train(model, device, trainloader, optimizer, epoch, train_losses,scheduler,train_acc, True )
      test(model, device, testloader, test_losses, test_acc)
      t_acc = test_acc[-1]
      if t_acc > best_test_accuracy:
          print("Test Accuracy: " + str(t_acc) + " has increased. Saving the model")
          best_test_accuracy = t_acc
          torch.save(model.state_dict(), PATH)