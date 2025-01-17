"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = nn.CrossEntropyLoss()(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))



class CIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs: 
        InputSize - Size of the Input (e.g., (3, 32, 32) for CIFAR-10 images)
        OutputSize - Size of the Output (10 classes for CIFAR-10)
        """
        super(CIFAR10Model, self).__init__()
        self.network = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(in_channels=InputSize[0], out_channels=32, kernel_size=3, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Down-sample to 16x16x32
            
            # Convolutional layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Down-sample to 8x8x64
            
            # Convolutional layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 8x8x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Down-sample to 4x4x128
            
            # Fully connected layers
            nn.Flatten(),  # Flatten 4x4x128 to 2048
            nn.Linear(4*4*128, 512),  # Hidden layer
            nn.ReLU(),
            nn.Linear(512, OutputSize)  # Output layer
        )
        
    def forward(self, xb):
        """
        Input:
        xb - MiniBatch of the current image (shape: [BatchSize, Channels, Height, Width])
        Output:
        out - output of the network (shape: [BatchSize, OutputSize])
        """
        out = self.network(xb)
        return out
  
class CIFAR10Model2(ImageClassificationBase): ## Improved
  def __init__(self, InputSize, OutputSize):
      """
      Inputs:
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      #############################
      # Fill your network initialization of choice here!
      #############################

      super().__init__()
      self.cn1 = nn.Conv2d(3, 10, 3, padding=1)
      self.cn1_bn = nn.BatchNorm2d(10) # Added to Improve architecture
      self.mp = nn.MaxPool2d(2, 2)
      self.cn2 = nn.Conv2d(10, 20, 3, padding=1)
      self.cn2_bn = nn.BatchNorm2d(20) # Added to Improve architecture
      self.fc1 = nn.Linear(1280, 128)
      self.fc2 = nn.Linear(128, 64)
      self.fc2_bn = nn.BatchNorm1d(64) # Added to Improve architecture
      self.fc3 = nn.Linear(64, 10)
      self.dropout = nn.Dropout(0.5) # Added to Improve architecture

  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      #############################
      # Fill your network structure of choice here!
      #############################

      out = self.cn1_bn(self.cn1(xb))
      out = self.mp(F.relu(out))
      out = self.mp(F.relu(self.cn2(out)))
      out = out.reshape(-1, 1280) #reshaping

      out = F.relu(self.fc1(out))
      out = F.relu(self.fc2_bn(self.fc2(out)))
      out = F.dropout(out)
      out = F.relu(self.fc3(out))

      return out
  
class ResidualBlock(ImageClassificationBase):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Adding the residual (skip) connection
        out = F.relu(out)
        return out


class ResNet(ImageClassificationBase):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

class BasicBlock(ImageClassificationBase):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super(BasicBlock, self).__init__()

        # Grouped convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Adding the residual (skip) connection
        out = F.relu(out)
        return out


class ResNeXt(ImageClassificationBase):
    def __init__(self, block, num_blocks, cardinality=32, num_classes=10):
        super(ResNeXt, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, cardinality=cardinality)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cardinality=cardinality)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cardinality=cardinality)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, cardinality=cardinality)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, cardinality):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, cardinality))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, cardinality=cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNeXt29_32x4d():
    return ResNeXt(BasicBlock, [3, 3, 3, 3], cardinality=32)


class DenseNet(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super().__init__()
      self.idty = nn.Identity()
      self.cn1 = nn.Conv2d(3, 10, 3, padding=1)
      self.cn2 = nn.Conv2d(10, 10, 3, padding=1)
      self.cn3 = nn.Conv2d(10, 10, 3, padding=1)
      self.cn4 = nn.Conv2d(10, 20, 3, padding=1)
      self.cn5 = nn.Conv2d(20, 20, 3, padding=1)
      self.cn6 = nn.Conv2d(20, 20, 3, padding=1)
      self.fc1 = nn.Linear(32*32*20, 1024)
      self.fc2 = nn.Linear(1024, 84)
      self.fc3 = nn.Linear(84, 10)

  def forward(self, xb):
      out = F.relu(self.cn1(xb))
      x = out
      x1 = F.relu(self.cn2(out))
      out = self.idty(x) + x1
      x2 = F.relu(self.cn3(out))
      out = self.idty(x) + x1 + x2
      x = F.relu(self.cn4(out))
      x1 = F.relu(self.cn5(x))
      out = self.idty(x) + x1
      x2 = self.cn6(out)
      out = self.idty(x) + x1 + x2
      out = out + self.idty(x)
      out = out.reshape(-1, 32*32*20)
      out = F.sigmoid(self.fc1(out))
      out = F.sigmoid(self.fc2(out))
      out = F.softmax(self.fc3(out))
      return out