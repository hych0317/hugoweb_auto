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

例:softmax计算梯度:
![softmax](post/AIGC/Machine_Learning/softmax_grad.png)

### 反向传播
反向传播（backpropagation）是指通过计算梯度来更新模型参数的算法。反向传播算法的基本思想是：从输出层开始，沿着损失函数的梯度方向更新参数，直到更新到网络的输入层。
![pic1](post/AIGC/Machine_Learning/back1.png)
![pic2](post/AIGC/Machine_Learning/back2.png)
![pic3](post/AIGC/Machine_Learning/back3.png)


## 神经网络
### CNN
网络结构:
卷积核 -> 激活函数 -> 池化层 -> 全连接层 -> 激活函数 -> 输出层
使用卷积核提取图像特征，通过激活函数对特征进行非线性变换，通过池化层对特征进行降维，再通过全连接层进行分类。
### RNN
在激活函数的输出重新连接到网络的输入，使得网络能够记住之前的输入，并对当前输入做出更好的预测。但是，这条路径的权重会导致梯度消失(w<1)和梯度爆炸(w>1)的问题。由此提出了LSTM
### LSTM
遗忘门、输入门、输出门：
![LSTM1](post/AIGC/Machine_Learning/forgetgate.png)
![LSTM2](post/AIGC/Machine_Learning/inputgate.png)
![LSTM3](post/AIGC/Machine_Learning/outputgate.png)

### GAN

### Transformer
#### Embedding
词嵌入（embedding）是指将词语转换为固定维度的向量表示的过程。词嵌入可以提高文本分类、文本匹配、文本聚类等任务的性能。常见的词嵌入方法有词向量、词袋模型、GloVe、BERT等。

位置编码:给每个输入值生成特定的位置值序列,保证语序
![position encoding](post/AIGC/Machine_Learning/pos_encoding.png)
#### Self Attention
自注意力机制（self-attention）是指模型通过注意力机制来获取输入序列的全局信息。这有助于为每个输入提供上下文信息，并建立输入间的联系。  
自注意力机制使每个token计算与其它token间的相似度(Query Key)，从而得知如it指代的是哪个词之类的信息。  
注意,每个token的Q\K\V权重系数都是相同的。  
![self attention](post/AIGC/Machine_Learning/selfattention.png)

将自注意力模块的输出通过softmax函数,得到每个token的权重,再将权重与value序列相乘,得到最终的嵌入向量。
![self attention2](post/AIGC/Machine_Learning/transformer1.png)

#### Encoder-decoder Attention
