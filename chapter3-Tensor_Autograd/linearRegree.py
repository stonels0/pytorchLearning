# -*- coding: utf-8 -*-
# @Author: Lishi
# @Date:   2017-11-09 18:41:13
# @Last Modified by:   Lishi
# @Last Modified time: 2017-11-14 20:00:48
'''
    线性回归利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的，其表达形式为 y=wx+b+e ，
    e 为误差服从均值为0的正态分布。
    1、首先让我们来确认线性回归的损失函数
        loss = sum(yi - (wxi+b)**2)/2
    2、参数更新（w和b）：SGD-随机梯度下降法
    3、最小化损失函数（目标）

'''

import torch as t
from torch.autograd import Variable
# %matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
import pdb

# 设置随机数种子，为了在不同人电脑上运行时下面的输出一致
t.manual_seed(1000)


def get_fake_data(batch_size=8):
    ''' 产生随机数据：y = x*2 + 3，加上了一些噪声'''
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
    return x, y


def displayData():
    # 来看看产生x-y分布是什么样的
    x, y = get_fake_data()
    plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
    plt.show()


def linearRegress():

    # 随机初始化函数,拟合函数为 「y = x*2 + 3」
    # 随机初始化参数
    w = t.rand(1, 1) 
    b = t.zeros(1, 1)

    lr =0.001 # 学习率

    for ii in range(20000):
        x, y = get_fake_data()
    
        # forward：计算loss
        y_pred = x.mm(w) + b.expand_as(y) # x@W等价于x.mm(w);for python3 only
        loss = 0.5 * (y_pred - y) ** 2 # 均方误差
        loss = loss.sum()
    
        # backward：手动计算梯度
        dloss = 1
        dy_pred = dloss * (y_pred - y)
    
        dw = x.t().mm(dy_pred)
        db = dy_pred.sum()

        # 更新参数
        w.sub_(lr * dw)
        b.sub_(lr * db)
    
        if ii%1000 ==0:
       
            # 画图
            display.clear_output(wait=True)
            x = t.arange(0, 20).view(-1, 1)
            y = x.mm(w) + b.expand_as(x)
            plt.plot(x.numpy(), y.numpy()) # predicted
        
            x2, y2 = get_fake_data(batch_size=20) 
            plt.scatter(x2.numpy(), y2.numpy()) # true data
        
            plt.xlim(0, 20)
            plt.ylim(0, 41)
            plt.show()
            plt.pause(0.5)
        
    print(w.squeeze()[0], b.squeeze()[0])


def main():
    #displayData()
    linearRegress()
    print("this is over!!!")


if __name__ == '__main__':
    main()