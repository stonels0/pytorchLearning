# -*- coding: utf-8 -*-
# @Author: Lishi
# @Date:   2017-11-09 18:41:13
# @Last Modified by:   Lishi
# @Last Modified time: 2017-11-14 20:00:48

from __future__ import print_function
import torch as t
from torch.autograd import Variable

import os
import sys, pdb


def tensorTest():
    pass
    a = t.Tensor(2, 3)  # a的值取决于内存空间的状态

    # 使用list的数据来创建tensor
    listTemp = [[1, 2, 3], [4, 5, 6]]
    b = t.Tensor(listTemp)
    b_size = b.size()  # 返回 torch.Size对象（为tuple的子类）
    b_shape = b.shape  # 查看属性值
    b.numel()  # 返回b中元素的总个数

    # 创建 Tensor变量
    c = t.Tensor(b_size)
    d = t.Tensor((2, 3))  # 传入元组，元素值为元组值

    ### 常用操作
    ## 调整形状(Tensor.shape),tensor.view必须保证调整前后「元素数目一致」
    a = t.arange(0, 6)
    a.view(2, 3)  # 并不会改变内存结构，仅仅作为视图进行显示「内存共享」
    b = a.view(-1, 3)  # a和b共享内存
    # 维度相关，即torch.Size对象进行改变
    c = b.unsqueeze(1)  # 注意形状，在第一维（下标从0开始）上增加“1”
    c.shape  # 在第一个维度上插入维度1

    c = b.view(1, 1, 1, 1, 2, 3)
    c.shape
    c.squeeze()  # 把所有多余为1的维度去掉

    ### 索引操作：如无特殊说明，索引出来的结果与原tensor共享内存
    a = t.randn(3, 4)
    a[:, 0]  # 第0列
    a[0][2]  # 第0行第2列，等价于 a[0,2]
    a[0, -1]  # 第0行的最后一个元素
    a[:2]  # 前两列
    a[:2, 0:2]  # 前两行，第0,1列
    print(a[0:1, :2])  # 第0行，前两列
    print(a[0, :2])  # 注意两者的区别：形状不同

    a > 1  # 返回 torch.ByteTensor 对象
    a[a > 1]  # 等价于a.masked_select(a>1), 选择结果与原tensor不共享内存空间

    ###gather 取dim维度上index的值
    a = t.arange(0, 16).view(4, 4)
    ## 取对角线上的数字(输出size = index的size)
    index = t.LongTensor([[0, 1, 2, 3]])
    diaglog = a.gather(0, index)

    ## 选取两个对角线上的元素
    index = t.LongTensor([[0, 1, 2, 3], [3, 2, 1, 0]]).t()
    b = a.gather(1, index)

    ### 类型转化问题
    ## numpy 和 Tensor变量之间的转化
    b.tolist()

    ## 广播操作
    a = t.ones(3, 2)  # a.shape = torch.Size([3, 2])
    b = t.zeros(2, 3, 1)  # b.shape = torch.Size([2, 3, 1])
    # 自动广播法则
    # 第一步：a是2维,b是3维，所以先在较小的a前面补1 ，
    #            即：a.unsqueeze(0)，a的形状变成（1，3，2），b的形状是（2，3，1）,
    # 第二步: a和b在第一维和第三维形状不一样，其中一个为1 ，
    #            可以利用广播法则扩展，两个形状都变成了（2，3，2）
    a + b


def autogradTest():
    a = Variable(t.ones(3, 4), requires_grad=True)
    b = Variable(t.zeros(3, 4))
    print("this variable is : a \n{} \n b:\n{}\n".format(a, b))

    c = a.add(b)  # c = a+b,c并非为用户创建，为计算所得 c.is_leaf False;
    print(c)
    d = c.sum()
    d.backward()  # 反向传播, 因为a依赖于c，故c自动设置为c.requires_grad = True
    ## 注意二者区别：
    # 前者：取data后变为tensor，而后从tensor计算得到float
    # 后者：计算sum得到仍然为Variable
    c.data.sum(), c.sum()

    a.grad  # 反向传播后可以直接根据属性获取 grad 梯度

    # 此处虽然没有指定c需要求导，但c依赖于a，而a需要求导，
    # 因此c的requires_grad属性会自动设为True
    print(a.requires_grad, b.requires_grad, c.requires_grad)

    # 用户创建的Variable属于「叶子节点」，对应的grad_fn 为None
    print(a.is_leaf, b.is_leaf, c.is_leaf)
    # 因为c.grad = None，因c为非叶子节点，其梯度用来计算a的梯度，故其requires_grad自动设置为True，但其梯度计算完后进行释放
    print("c.grad is None : {}".format(c.grad is None))

    ## 计算导函数：y = pow(x,2)*pow(e,x)
    def f(x):
        """ 计算y """
        y = x**2 * t.exp(x)
        return y

    def gradf(x):
        """ 手动求导函数 """
        dx = 2 * x * t.exp(x) + x**2 * t.exp(x)
        return dx

    x = Variable(t.randn(3, 4), requires_grad=True)
    y = f(x)
    print("this function result is y = f(x): \n {}".format(y))

    y.backward(t.ones(
        y.size()))  # grad_variables 形状要和y保持一致（传入参数为grad_variables）
    print("(autoGrad) : this grad is : {}".format(x.grad))  # 输出结果：dy/dx
    print("(selfComp) : this grad is : {}".format(gradf(x)))  # 输出手动计算结果

    pdb.set_trace()
    ##  学习autograd的实现细节
    # 函数：z = wx + b
    x = Variable(t.ones(1))
    b = Variable(t.rand(1), requires_grad=True)
    w = Variable(t.rand(1), requires_grad=True)
    y = w * x  # 等价于y=w.mul(x),虽然未指定y.requires_grad = True,但由于y依赖于需要求导的w,故而y.requires_grad为True
    z = y + b  # 等价于z=y.add(b)
    print(x.requires_grad, y.requires_grad, z.requires_grad)
    print(y.is_leaf, z.is_leaf)

    # grad_fn可以查看这个variable的反向传播函数，
    # z是add函数的输出，所以它的反向传播函数是AddBackward
    print(z.grad_fn)
    # 「next_functions」保存grad_fn的输入，是一个tuple，tuple的元素也是Function
    # 第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数y.grad_fn是MulBackward
    # 第二个是b，它是叶子节点，由用户创建，grad_fn为None，但是有
    print(z.grad_fn.next_functions)

    # variable的grad_fn对应着和图中的function相对应（换言之：）
    z.grad_fn.next_functions[0][0] == y.grad_fn

    # 第一个是w，叶子节点，需要求导，梯度是累加的
    # 第二个是x，叶子节点，不需要进行求导，所以为None
    y.grad_fn.next_functions

    # 叶子节点的grad_fn是None
    print("w.grad_fn:{}\nx.grad_fn:{}\n".format(w.grad_fn, x.grad_fn))

    # 计算w的梯度时，需要用到x的数值，这些值在「前向传播」中会保存成buffer，在计算完梯度之后会自动清空。为了能够多次反向传播需要指定的「retain_graph」来保留这些buffer信息
    # 使用retain_graph来保存buffer
    z.backward(retain_graph=True)
    print("w.grad is : {}".format(w.grad))

    # 多次反向传播，梯度累加，这也就是w中AccumulateGrad标识的含义
    z.backward()
    print("w.grad is : {}".format(w.grad))

    # pytorch使用的是「动态图」，即它的计算图在每次「前向传播」中都是从头开始构建，所以它能够使用python控制语句（if、for等）根据需要创建计算图。
    def abs(x):
        if x.data[0] > 0: return x
        else: return -x

    x = Variable(t.ones(1), requires_grad=True)
    y = abs(x)
    y.backward()
    x.grad

    x = Variable(t.ones(1), requires_grad=True)
    y = abs(x)
    y.backward()
    x.grad

    def f(x):
        result = 1
        for ii in x:
            if ii.data[0] > 0: result = ii * result
        return result

    x = Variable(t.arange(-2, 4), requires_grad=True)
    y = f(x) # y = x[3] *x[4]*x[5]
    y.backward()
    x.grad

    # requires_grad属性设置的依赖性传递
    x = Variable(t.ones(1))
    w = Variable(t.rand(1), requires_grad=True)
    y = x * w
    # y依赖于w，而w.requires_grad = True
    print("x.requires_grad : {}, w.requires_grad : {}, y.requires_grad : {}".
          format(x.requires_grad, w.requires_grad, y.requires_grad))
    x = Variable(t.ones(1), volatile=True)
    w = Variable(t.rand(1), requires_grad = True)
    y = x * w
    # y依赖于w和x，但x.volatile = True，w.requires_grad = True
    x.requires_grad, w.requires_grad, y.requires_grad

    pdb.set_trace()
    print("this is over!!!")


def main():
    # tensorTest()
    autogradTest()


if __name__ == '__main__':
    main()