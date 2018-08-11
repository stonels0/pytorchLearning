# -*- coding: utf-8 -*-
# @Author: Lishi
# @Date:   2017-11-09 18:41:13
# @Last Modified by:   Lishi
# @Last Modified time: 2017-11-14 20:00:48
'''
   注意：定义网络时，需继承nn.Module，并实现它的forward方法
   __init__：网络中 具有可学习参数的层 放到此构造函数中
   如果某一层 不具有可学习的参数 （如ReLU），既可以放到 构造函数（__init__）中，也可以不放，建议不放在构造函数
   而在forward中使用 nn.functional 代替
'''
from __future__ import print_function
import torch as t
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 优化器


class Net(nn.Module):
    def __init__(self):
        # nn.Module 子类的函数必须在构造函数中执行 父类的构造函数
        # 下式等价于 nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层‘1’表示输入图片为单通道，‘6’表示输出通道数，‘5’表示卷积核为 5*5
        self.conv1 = nn.Conv2d(1, 6, 5)

        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层， y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # nn.Module类中forward函数的输入和输出都是Variable而非Tensor
        # 因为Variable才具有自动求导功能（所有在输入时，需要将Tensor封装成 Variable）
        # 卷积 ——> 激活 ——> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # (2,2) 规定了池化层的大小;
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape, '-1' 表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    net = Net()
    # print(net)
    # 打印网络的学习参数
    params = list(net.parameters())
    print(len(params))

    for name, parameters in net.named_parameters():
        print("{} : {}".format(name, parameters.size())
              )  # fc1.weight : torch.Size([120, 400])，其中[输出，输入]

    input = Variable(t.randn(1, 1, 32, 32))
    out = net(input)
    print(out.size())

    # 将net中所有 “可学习参数” 的梯度清零
    # net.zero_grad()
    # 进行反向传播
    # out.backward(Variable(t.ones(1, 10)))
    # print("the backward data is : {}".format(input.grad))
    # 损失函数 调用
    output = net(input)
    target = Variable(t.arange(0, 10))
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss
    print("the loss of the net is : {}".format(loss))
    print(output.grad_fn)

    # 对loss进行 反向传播 溯源（使用grad_fn属性）此网络计算图如下：
    # input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
    #  -> view -> linear -> relu -> linear -> relu -> linear
    #  -> MSELoss
    #  -> loss
    # 运行 .backward(),观察调用之前和调用之后的grad (调用.backward() 时，该图会自动微分，即自动计算图中参数的导数)
    net.zero_grad()
    print("反向传播之前 conv1.bias 的梯度 \n {}".format(net.conv1.bias.grad))
    loss.backward()
    print("反向传播之后 conv1.bias 的梯度 \n {}".format(net.conv1.bias.grad))

    ## 优化器 （反向传播计算完所有梯后，需要使用优化方法 更新网络的权重和参数——例如，SGD的更新策略）
    # SGD 更新参数
    # 新建一个优化器，指定要调整的参数和学习率
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # 在训练过程中
    # 先将梯度清零（与net.zero_grad()效果一致）
    optimizer.zero_grad()

    # 计算损失
    output = net(input)
    loss = criterion(output, target)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

if __name__ == '__main__':
    main()
