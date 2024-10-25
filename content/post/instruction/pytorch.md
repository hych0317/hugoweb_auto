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

# 在默认情况下,PyTorch会累积梯度,我们需要清除之前的值  
x.grad.zero_()
y = x.sum()# ATTENTION: 这里的y是标量,因为反向传播需要损失函数上的一个特定的值,从而计算梯度.此时y相当于损失函数,需要是一个值(标量),这样才可以进行backwards()
y.backward()
print(x.grad) # 输出梯度张量

# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x# hadamard积,按元素
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```
### 概率

## 线性神经网络

## 多层感知机

## 卷积神经网络

## 循环神经网络

## 注意力机制

## 优化算法

## 计算机视觉

