import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, dropout_value):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1A
        self.convblock1A = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        # CONVOLUTION BLOCK 2A
        self.convblock2A = nn.Sequential(
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # CONVOLUTION BLOCK 3A
        self.convblock3A = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14

        # CONVOLUTION BLOCK 4A
        self.convblock4A = nn.Sequential(
            nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14

        # CONVOLUTION BLOCK 1B
        self.convblock1B = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        # CONVOLUTION BLOCK 2B
        self.convblock2B = nn.Sequential(
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # CONVOLUTION BLOCK 3B
        self.convblock3B = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14

        # CONVOLUTION BLOCK 4B
        self.convblock4B = nn.Sequential(
            nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 14


        ## CONVOLUTION BLOCK 4 - Depthwise Convolution
        self.depthwise_separable_block = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (3, 3), padding = 0, groups = 96, bias = False),
            nn.Conv2d(in_channels = 96, out_channels = 60, kernel_size = (1, 1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.Dropout(dropout_value)
        ) ## output_size = 12

        ## CONVOLUTION BLOCK 5 - Reduction
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels = 60, out_channels = 30, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 30, out_channels = 10, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6) ## Global Average Pooling
        ) # output_size = 1


    def forward(self, x):
        x1 = self.convblock1A(x)
        x1 = self.convblock2A(x1)
        x1 = self.convblock3A(x1)
        x1 = self.convblock4A(x1)

        x2 = self.convblock1B(x)
        x2 = self.convblock2B(x2)
        x2 = self.convblock3B(x2)
        x2 = self.convblock4B(x2)

        y = torch.cat((x1,x2), 1)

        y = self.depthwise_separable_block(y)
        y = self.convblock5(y)
        y = self.gap(y)

        y = y.view(-1, 10)
        return F.log_softmax(y, dim=-1)

def model_summary(model, input_size):
    summary(model, input_size)

def get_lr(optimizer):

    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, device, train_loader, train_acc, train_loss, optimizer, scheduler, criterion, lrs, grad_clip=None):

    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):

        ## Get data samples
        data, target = data.to(device), target.to(device)

        ## Init
        optimizer.zero_grad()

        ## Predict
        y_pred = model(data)

        ## Calculate loss
        loss = criterion(y_pred, target)

        train_loss.append(loss.data.cpu().numpy().item())

        ## Backpropagation
        loss.backward()

        # Gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        lrs.append(get_lr(optimizer))

        ## Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]:0.5f} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader, test_acc, test_losses, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    
# functions to show an image
def imshow_sample(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def draw_graphs(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")