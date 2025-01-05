#导包
import torchvision
import torch
from model import * #全导入
from torch import nn
from torch.nn import Sequential,Module,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#CIFAR-10数据集导入
train_data = torchvision.datasets.CIFAR10('./dataset',
                                          train = True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10('./dataset',
                                         train = False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

#VScode中的快捷键 ctrl + /：注释 ctrl + f ：在终端中查找
#看训练集的大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练集的长度为{}'.format(train_data_size))
print('测试集的长度为{}'.format(test_data_size))

#利用Dataloader 来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#调用网络
n = NN()

#损失函数 交叉熵损失
loss_fn = nn.CrossEntropyLoss()

#优化器 随机梯度下降
# 1e-2 =  1X10^(-2) = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(n.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 500

#添加tensorboard
writer = SummaryWriter('logs_train')

for i in range(epoch):
    print('-----第{}轮训练开始-----'.format(i+1))
    #训练开始
    for data in train_dataloader:
        imgs,targets = data
        output = n(imgs)
        loss = loss_fn(output,targets)

        #优化器优化模型 梯度清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()#?

        total_train_step += 1
        if total_train_step %100 == 0:#100次打印一次损失
            print('训练次数 {} ， loss: {} '.format(total_train_step,loss))
            writer.add_scalar('train_loss_step',loss,total_train_step)
    writer.add_scalar('train_loss_epoch',loss,i)
    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():#怎么判断训练的怎么样？？？ 直接试一轮测试集
        for data in test_dataloader:
            imgs,targets = data
            output = n(imgs)
            loss = loss_fn(output,targets)
            total_test_loss = total_test_loss + loss
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss: {} ".format(total_test_loss))
    print('整体测试集上的正确率:{}'.format(total_accuracy/test_data_size))
    writer.add_scalar('test_loss',total_test_loss,total_test_step)
    writer.add_scalar('test_accuracy',total_accuracy,total_test_step)
    total_test_step += 1

    # torch.save(n,'n_{}.pth'.format(i)) #保存每一轮的参数
    print('模型已保存')
    # argmax的用法
    # import torch
    # outputs = torch.tensor([[0.1,0.2,0.3],
    #                         [0.3,0.01,0.1]])
    # print(outputs.argmax(1))#argmax中的0和1代表竖着看还是横着看
    # preds = outputs.argmax(1)
    # targets = torch.tensor([2,0])
    # print((preds == targets).sum())
# writer.add_graph(n,train_data)
torch.save(n,'n_500.pth')
writer.close()






