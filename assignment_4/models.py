import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class point_model(nn.Module):
    ### Complete for task 1
    def __init__(self,num_classes):
        super(point_model, self).__init__()
        self.mlp1 = nn.Conv1d(3, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 8)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    def forward(self,x):
        x = x.transpose(1,2).contiguous()
        # print(x.shape)
        x = F.relu(self.bn1(self.mlp1(x)))
        # print('2', x.shape)
        x = F.relu(self.bn2(self.mlp2(x)))
        # print('3', x.shape)
        x = F.relu(self.bn3(self.mlp3(x)))
        # print('4', x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        # print('5', x.shape)
        x = x.view(-1, 1024)
        # print('6', x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        # print('7', x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        # print('8', x.shape)
        x = self.fc3(x)
        # print('9', x.shape)
        return x

        

class voxel_model(nn.Module):
    ### Complete for task 2
    def __init__(self,num_classes):
        super(voxel_model, self).__init__()
        self.conv1 = nn.Conv3d(1,8,3)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8,16,4)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(16,32,4)
        self.pool3 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(32,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,8)

        self.bn1 = nn.BatchNorm3d(8)
        self.bn2 = nn.BatchNorm3d(16)
        self.bn3 = nn.BatchNorm3d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
    def forward(self,x):
        x = x.reshape(-1,1,32,32,32)
        # print('1', x.shape)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # print('2', x.shape)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # print('3', x.shape)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # print('4', x.shape)
        x = x.view(-1, 32)
        # print('4.5', x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        # print('5', x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        # print('6', x.shape)
        x = self.fc3(x)
        # print('6.5', x.shape)
        return x
        

class spectral_model(nn.Module):
    ### Complete for task 3
    def __init__(self,num_classes):
        super(spectral_model, self).__init__()
        self.mlp1 = nn.Conv1d(6, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 8)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    def forward(self,x):
        x = x.transpose(1,2).contiguous()
        # print('1', x.shape)
        x = F.relu(self.bn1(self.mlp1(x)))
        # print('2', x.shape)
        x = F.relu(self.bn2(self.mlp2(x)))
        # a = nn.BatchNorm1d(128)
        # x = a(x)
        # print('3', x.shape)
        x = F.relu(self.bn3(self.mlp3(x)))
        # print('4', x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        # print('5', x.shape)
        x = x.view(-1, 1024)
        # print('6', x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        # print('7', x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        # print('8', x.shape)
        x = self.fc3(x)
        # print('9', x.shape)
        return x