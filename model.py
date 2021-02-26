import torch
from torch import nn
from PIL import Image
import numpy as np

# These are largely the same components and functions that are used in the Pix2Pix notebook provided on Coursera

# Block that reduces the resolution of the image but doubles the number of channels
class ContractingBlock(nn.Module):
  def __init__(self, input_channels, use_bn=True):
    super(ContractingBlock, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3, padding=1)
    self.activation = nn.LeakyReLU(0.2)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    if use_bn:
      self.batchnorm = nn.BatchNorm2d(input_channels * 2)
    self.use_bn = use_bn

  def forward(self, x):
    x = self.conv1(x)
    if self.use_bn:
      x = self.batchnorm(x)
    x = self.activation(x)
    x = self.conv2(x)
    if self.use_bn:
      x = self.batchnorm(x)
    x = self.activation(x)
    x = self.maxpool(x)
    return x

# Crop function for link layers
def crop(image, new_shape):
  cropped_image = image[:,:,int((image.shape[2]-new_shape[2])/2):int((image.shape[2]+new_shape[2])/2),int((image.shape[3]-new_shape[3])/2):int((image.shape[3]+new_shape[3])/2)]
  return cropped_image

# Increases the resolution but halves the number of channels, on the upward half of the U-Net Generator
class ExpandingBlock(nn.Module):
  def __init__(self, input_channels, use_bn = True):
    super(ExpandingBlock, self).__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv1 = nn.Conv2d(input_channels, int(input_channels/2), kernel_size=2)
    self.conv2 = nn.Conv2d(input_channels, int(input_channels/2), kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(int(input_channels/2), int(input_channels/2), kernel_size=2, padding = 1)
    if use_bn:
      self.batchnorm = nn.BatchNorm2d(input_channels // 2)
      self.use_bn = use_bn
    self.activation = nn.ReLU()

  def forward(self, x, skip_con_x):
    x = self.upsample(x)
    x = self.conv1(x)
    skip_con_x = crop(skip_con_x, x.shape)
    x = torch.cat([x, skip_con_x], axis=1)
    x = self.conv2(x)
    if self.use_bn:
      x = self.batchnorm(x)
    x = self.activation(x)
    x = self.conv3(x)
    if self.use_bn:
      x = self.batchnorm(x)
    x = self.activation(x)
    return x

# Ensures that the number of channels going in and the number of channels going out remains compatible with the rest of the model
class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

# Generator, it's just a U-Net
class Generator(nn.Module):
  def __init__(self, input_channels, output_channels, hidden_channels=64):
    super(Generator, self).__init__()
    self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
    self.contract1 = ContractingBlock(hidden_channels)
    self.contract2 = ContractingBlock(hidden_channels * 2)
    self.contract3 = ContractingBlock(hidden_channels * 4)
    self.contract4 = ContractingBlock(hidden_channels * 8)
    self.expand1 = ExpandingBlock(hidden_channels * 16)
    self.expand2 = ExpandingBlock(hidden_channels * 8)
    self.expand3 = ExpandingBlock(hidden_channels * 4)
    self.expand4 = ExpandingBlock(hidden_channels * 2)
    self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
    self.sigmoid = torch.nn.Sigmoid()
  
  def forward(self,x):
    x0 = self.upfeature(x)
    x1 = self.contract1(x0)
    x2 = self.contract2(x1)
    x3 = self.contract3(x2)
    x4 = self.contract4(x3)
    x5 = self.expand1(x4, x3)
    x6 = self.expand2(x5, x2)
    x7 = self.expand3(x6, x1)
    x8 = self.expand4(x7, x0)
    xn = self.downfeature(x8)
    return self.sigmoid(xn)

# Discriminator, Patch-GAN. Uses the same contracting block component that is used in U-Net.
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn
