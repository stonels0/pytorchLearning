# -*- coding: utf-8 -*-
# @Author: Lishi
# @Date:   2017-11-09 18:41:13
# @Last Modified by:   Lishi
# @Last Modified time: 2017-11-14 20:00:48

from __future__ import print_function
import torch as t

import os
import sys


def tensorTest(parameter_list):
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
    c = b.unsequeeze(1)  # 主意形状，在第一维（下标从0开始）上增加“1”
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


def main():
    tensorTest()


if __name__ == '__main__':
    main()