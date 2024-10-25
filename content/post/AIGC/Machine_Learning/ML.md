+++
title = 'Machine Learning'
date = 2024-10-21T15:34:07+08:00
draft = false
categories = ['AIGC']
+++

## 基础概念
### 张量（Tensor）
张量（tensor）是指具有多个维度的数组，它可以用来表示向量、矩阵、高阶数组等多种数据结构。张量的元素可以是标量、向量、矩阵、张量等。

### 损失函数
损失函数（loss function）是指用来衡量模型预测值与真实值之间的差距，并反映模型的预测精度的函数。损失函数的选择对模型的训练、优化和泛化能力都有着至关重要的影响。常见的损失函数有：

#### 残差平方和(residual sum of squares, RSS)   
公式：$L(y, \hat{y}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2$  


### 梯度下降法
梯度下降法（gradient descent）是一种优化算法，它通过迭代的方式不断更新模型的参数，使得损失函数的值逐渐减小。梯度下降法的基本思想是：沿着损失函数的负梯度方向更新参数，使得损失函数的值减小。

### 反向传播
反向传播（backpropagation）是指通过计算梯度来更新模型参数的算法。反向传播算法的基本思想是：从输出层开始，沿着损失函数的梯度方向更新参数，直到更新到网络的输入层。
![pic1](post/AIGC/ML/back1.png)
![pic2](post/AIGC/ML/back2.png)
![pic3](post/AIGC/Machine_Learning/back3.png)


## 神经网络
### CNN

### RNN

### LSTM

### GAN

### Transformer
#### Attention