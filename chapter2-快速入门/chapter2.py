# -*- coding: utf-8 -*-
# @Author: Lishi
# @Date:   2017-11-09 18:41:13
# @Last Modified by:   Lishi
# @Last Modified time: 2017-11-14 20:00:48

from __future__ import print_function
import torch as t
import numpy as np

from torch.autograd import Variable  #

import os, sys
import shutil
import pdb


def testTensor():
    x = t.Tensor(5, 3)  # 仅仅分配了空间，并未初始化
    x = t.rand(5, 3)  # 使用 [0,1] 均匀分布随机初始化二维数组
    print(x.size())

    y = t.rand(5, 3)

    # 进行加法计算
    result = t.Tensor(5, 3)
    t.add(x, y, out=result)  # 输出到result

    ## 函数名后面带下划线_ 的函数会修改Tensor本身,如 add_()
    y.add(x)  # 普通加法，不会改变y的值
    y.add_(x)  #inplace 加法，y改变

    ## 类型转化
    # numpy 可以处理 Tensor不可执行的
    a_t = t.ones(5)
    b_np = a.numpy()  #Tensor => Numpy
    a_t = t.from_numpy(b)  # Numpy => Tensor

    # Tensor => GPU Tensor,加速
    if t.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x + y

    # Autograd : 自动微分
    x = Variable(t.ones(2, 2), requires_grad=True)
    y = x.sum()
    y.grad_fn
    y.backward()  # 反向传播，计算梯度;

    # y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
    # 每个值的梯度都为1
    x.grad


def main():
    print("this is a test {}".format(os.getcwd()))


if __name__ == '__main__':
    main()