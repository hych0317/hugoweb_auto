+++
title = 'Pytorch'
date = 2024-10-21T15:34:07+08:00
draft = false
categories = ['instruction']
+++

## 数据操作

### 张量基本操作
```python
import torch
# 1. 创建张量
x = torch.tensor([[1, 2, 3],[4,5,6]])
x0 = torch.zeros(2, 3)
x1 = torch.ones(2, 3)
xr = torch.rand(2, 3)
X = torch.arange(12, dtype=torch.float32).reshape(3,4)# 指定个数\类型\形状
# 2. 改变形状
xrs = x.reshape(3, 2)
print(y)
# 3. 查看属性
print(x.shape)
print(x.numel())# number of elements元素总数
X.sum()# 所有元素的和,产生单元素张量
# 4. 索引
X[1:3] # 索引从0开始,区间左闭右开,1:3即1、2,对应第二到第三行
X[:, 1] # 第二列
X[1:3, 1:3] # 第二到第三行,第二到第三列
X == Y # 元素比较,生成布尔张量
X[X>0] # 元素过滤
# 5. 张量连结
X = torch.cat([X, Y], dim=0) # 按行连接
X = torch.cat([X, Y], dim=1) # 按列连接
# 6. 广播机制
# 由于`a`和`b`分别是3*1矩阵和1*2矩阵，如果让它们相加，它们的形状不匹配。
# 将两个矩阵*广播*为一个更大的3*2矩阵，矩阵`a`将复制列，矩阵`b`将复制行，然后再按元素相加
a = torch.tensor([[1], [2], [3]])
b = torch.tensor([4, 5])
print(a + b)# 输出 tensor([[5, 6],[6, 7],[7, 8]])
```

### 线性代数
```python
# 1. 矩阵乘法
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
C = torch.mm(A, B) # 矩阵乘法
# 2. 按元素乘法
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a * b # 按元素乘法(hadamard积)
# 3. 向量点积(等同于按元素乘法后求和)
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.dot(a, b) # 向量点积,即torch.sum(a*b)
# 4. 矩阵求逆
A = torch.tensor([[1, 2], [3, 4]])
A_inv = torch.inverse(A) # 矩阵求逆
# 5. 矩阵转置
AT = A.T() # 矩阵转置
# 6. 矩阵降维(沿指定轴sum或mean)
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
# 7. 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
# 可利用广播机制
A/sum_A# 获得每一行间独立的概率分布
A.cumsum(axis=0)# 沿某个轴的累计总和,不会降维
# 8. 范数
A = torch.tensor([1, 2])
torch.norm(A, p=2) # 向量的L2范数,即向量元素平方和的平方根
torch.norm(A, p=1) # 向量的L1范数,即向量元素绝对值之和
torch.abs(u).sum() # L1范数的另一种表示形式
# 矩阵的Frobenius范数(矩阵元素平方和的平方根，类似于向量的L2范数)
torch.norm(torch.arange(36,dtype=torch.float32).reshape(4, 9))

```
### 导数和梯度
#### 画图
```python
# 见chapter0 calculus.py
import numpy as np
import matplotlib.pyplot as plt
 
# 设置x,y轴的数值
x1 = np.linspace(0, 15, 100)
y1 = np.sin(x1)
y2 = np.cos(x1)
 
# 在当前绘图对象中画图（x轴,y轴,给所绘制的曲线的名字，画线颜色，画线宽度）
plt.plot(x1, y1, label="$sin(x)$", color="blue", linewidth=2)
plt.plot(x1, y2, label="$cos(x)$", color="red", linewidth=2)
 
# X和Y坐标轴的表示
plt.xlabel("Domain")
plt.ylabel("Range")
 
# 图表的标题
plt.title("sin and cos")
# Y轴的范围
plt.ylim(-1.5, 1.5)
# 显示图示
plt.legend()
# 显示图
plt.show()
```
#### 自动求导
```python
import torch
x = torch.arange(4.0)# 数据类型需要为float,才可微分
x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
y = 2 * torch.dot(x, x)# 2*((x_i)**2)
y.backward() # 自动求导,应为dy/dx=4*x_i
print(x.grad) # 输出梯度张量

##1
#在默认情况下,PyTorch会累积梯度,我们需要清除之前的值  
x.grad.zero_()
y = x.sum()# ATTENTION: 这里的y是标量,因为反向传播需要损失函数上的一个特定的值,从而计算梯度.此时y相当于损失函数,需要是一个值(标量),这样才可以进行backwards()
y.backward()
print(x.grad) # 输出梯度张量

##2 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x# hadamard积,按元素
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad

##3 分离计算图
x0 = torch.arange(4.0,requires_grad=True)
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
y = x0 * x0

# 只希望使用当前y的值,然后计算z=u(y当前的值)*x0,但不希望获取y的梯度,导致z=x*x*x
u = y.detach()# 保存当前y的值,为常数,梯度不会更新
print(u)
z = u * x0
z.sum().backward()
print(x0.grad)
print(x0.grad == u)# u是常数,导数即u

x0.grad.zero_()
y.sum().backward()
print(x0.grad)
print(x0.grad == 2 * x0)# 导数为2*x0

##4 python控制流中的梯度计算
# 该例子想说明,标量在控制流中(循环,条件分支)进行运算仍会记录梯度的变化
import torch
def f(a):
    b = a
    while b.norm() < 1000:# 验证循环对梯度的影响
        b = b * 2
    if b.sum() > 0:# 验证条件分支对梯度的影响
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)

d = f(a)# d=2^n(b在循环内的次数n)*1或100(根据条件分支判断)*a
d.backward()# 因此d对a的导数就是d/a
print(a.grad)
print(a.grad == d / a)
```

### 深度学习计算
#### GPU相关
```python
import torch
# 查看是否有GPU
print(torch.cuda.is_available())
# 设置可见的GPU
import os
os.environ["CUDA_VISIBLE_DEVICES=1,3"]# 仅第二、四个GPU可见，引号可加可不加
## 也可以在运行前!export CUDA_VISIBLE_DEVICES=0.这样设置后程序中的1卡为实际的3卡
# 设置设备
device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义张量
x = torch.tensor([[1, 2, 3],[4,5,6]], device=device)
# 张量转移到GPU
x = x.to(device)
# 张量转移到CPU
x = x.to("cpu")
```
#### 层和块

#### 参数管理

#### 延迟初始化

#### 自定义层
```python

```

## 线性神经网络
### 线性回归的简洁实现
```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))# 输出第一个batch的特征和标签,进行验证

# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)# torch中，带下划线的一般指赋值,这里normal_指用均值0,方差0.01的正态分布给w的data属性(即w的值)赋值
net[0].bias.data.fill_(0)

loss = nn.MSELoss()# 损失函数:平方L2范数
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)# 优化算法和学习率

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)# 前向传播及计算损失函数
        optimizer.zero_grad()# 清空累计梯度
        l.backward()# 反向传播
        optimizer.step()# 优化参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```

### softmax回归
softmax函数返回一个概率分布,其值在0到1之间,且总和为1.因此softmax回归常用于多类别分类问题.
#### 图像分类数据集
使用Fashion-MNIST数据集,该数据集包含70,000张图像,分为10个类别,每张图像高和宽均为28像素.
```python
# softmax回归的简洁实现(从零开始实现见工程代码)
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，在线性层之前定义展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
## nn.Flatten()将输入的多维张量展平为一维向量.如28*28的图像参数向量将被展平为长784的一维向量

def init_weights(m):# 初始化权重
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs,0.03, 0)
# 包里没ch3的trainer了. d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```
## 多层感知机

## 卷积神经网络

## 循环神经网络

## 注意力机制

## 优化算法

## 计算机视觉

