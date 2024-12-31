import torchvision
import torch
from torch import nn
from torch.nn import Sequential,Module,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
import torch.nn.functional as F
#搭建神经网络 十分类的网络
class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.model1 = Sequential(#序列
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x
#测试模型写的对不对
if __name__ == 'main':
    n = NN()
    input = torch.ones((64,3,32,32))
    output = n(input)
    print(output.shape)