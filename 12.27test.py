import torchvision
import torch
import os
from model import * #全导入
from torch import nn
from torch.nn import Sequential,Module,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import load
from PIL import Image
import torch.nn.functional as F
# train_data = torchvision.datasets.CIFAR10('./dataset',
#                                           train = True,
#                                           transform=torchvision.transforms.ToTensor(),
#                                           download=True)
# test_data = torchvision.datasets.CIFAR10('./dataset',
#                                          train = False,
#                                          transform=torchvision.transforms.ToTensor(),
#                                          download=True)
# train_dataloader = DataLoader(train_data,batch_size=64)
# test_dataloader = DataLoader(test_data,batch_size=64)

# loss_fn = nn.CrossEntropyLoss()

# for i in range(6,10):
#     my_model = torch.load('D:\\python\\n_' + str(i) + '.pth')
#     #print(my_model)

#     test_data_size = len(test_dataloader)
#     total_test_loss = 0
#     total_accuracy = 0
#     # with torch.no_grad():
#     for data in test_dataloader:
#         imgs,targets = data
#         output = my_model(imgs)
#         loss = loss_fn(output,targets)
#         total_test_loss = total_test_loss + loss
#         accuracy = (output.argmax(1) == targets).sum()
#         total_accuracy = total_accuracy + accuracy
#     print("第{}轮参数下整体测试集上的loss: {} ".format(i,total_test_loss))
#     print('第{}轮参数下整体测试集上的正确率:{}'.format(i,total_accuracy/test_data_size))

img_path = r'Deep learning\AI-Homework\ship_test.jpg'
image = Image.open(img_path)

#转换函数 PIL->Tensor 
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
# writer = SummaryWriter('image')
# writer.add_image('logs',image)
# writer.close()
#image.show()
image = torch.reshape(image,(1,3,32,32))
print(image.shape)

#加载网络模型
my_model = torch.load('D:\\python\\n_500.pth')
my_model.eval()

with torch.no_grad():#可节约内存
    outputs = my_model(image)

probabilities = F.softmax(outputs, dim=1)
print(probabilities)
print(outputs.argmax(1))
# 0: airplane
# 1: automobile
# 2: bird
# 3: cat
# 4: deer
# 5: dog
# 6: frog
# 7: horse
# 8: ship
# 9: truck
# 10: _len_