## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 1 x 224 x 224
        self.conv1 = nn.Conv2d(1, 32, 3)
        # 32 x 222 x 222
        
        self.conv1_bn = nn.BatchNorm2d(32)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        #pool
        # 32 x 111 x 111
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 64 x 109x 109
        
        self.conv2_bn = nn.BatchNorm2d(64)
        
        # pool again
        # 64 x 54 x 54
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        # 128 x 52 x 52
        
        self.conv3_bn = nn.BatchNorm2d(128)
        # pool again
        # 128 x 26 x 26
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        # 256 x 24 x 24
        
        self.conv4_bn = nn.BatchNorm2d(256)
        
        #pool 256 x 12 x12
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(256*12*12, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 68*2)
        
        self.drop = nn.Dropout(p=0.2)
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # a modified x, having gone through all the layers of your model, should be returned
        
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        
        
        x = x.view(x.size(0), -1)
        
        x = self.drop(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.fc2(x)
        
        return x
