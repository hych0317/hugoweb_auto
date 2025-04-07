+++
title = 'Pytorch'
date = 2024-10-21T15:34:07+08:00
draft = false
categories = ['指令语法']
+++

## 常用概念

### 设备相关

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 创建设备
a = torch.ones(2,3)
b = a.clone()
c = a.detach()# 只想使用当前的值,为常数,避免导致原张量梯度更新
d = a.to(device)# "cuda" / "cpu"
```
### 流程简述
具体网络详解见第三部分  
```python
# 1.网络定义
import torch.nn as nn
import torch.optim as optim
# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = self.fc2(x)
        return x

# 2.创建网络实例
model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 4. 假设有训练数据 X 和 Y
X = torch.randn(10, 2)  # 10 个样本，2 个特征
Y = torch.randn(10, 1)  # 10 个目标值

# 5. 训练循环
for epoch in range(100):  # 训练 100 轮
    optimizer.zero_grad()  # 清空之前的梯度
    output = model(X)  # 前向传播
    # output可通过激活函数完成对应任务
    # import torch.nn.functional as F 
    # # ReLU 激活
    # output = F.relu(input_tensor)
    # # Sigmoid 激活
    # output = torch.sigmoid(input_tensor)
    # # Tanh 激活
    # output = torch.tanh(input_tensor)
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
   
    # 每 10 轮输出一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 6.评估
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在评估过程中禁用梯度计算
    output = model(X_test)
    loss = criterion(output, Y_test)
    print(f'Test Loss: {loss.item():.4f}')
```



## 张量操作

### 张量基本操作
#### 基本属性
张量tensor：  
属性：维度，形状，数据类型  
0维即单个数字，一维即一维数组  
形状指每个维度的大小，如(3,4)表示三行四列  

另还有维度数.dim(),启用梯度计算.requires_grad,获取元素总数.numel()等

```python
import torch
# 1. 创建张量
x = torch.tensor([[1, 2, 3],[4,5,6]])
x0 = torch.zeros(2, 3)
x1 = torch.ones(2, 3)
xr = torch.randn(2, 3)# rand随机，randn服从正态分布
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xd = torch.rand(2, 3, device = device)
X = torch.arange(12, dtype=torch.float32).reshape(3,4)# 指定个数\类型\形状
tensor_3d = torch.stack([xd,xd+3,xd+5])# 3维张量即三个二维的堆叠，用stack
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
#### 张量操作
    torch.matmul(x, y)	矩阵乘法
    torch.dot(x, y)	向量点积（仅适用于 1D 张量）
    torch.sum(x)	求和
    torch.mean(x)	求均值
    torch.max(x)	求最大值
    torch.min(x)	求最小值
    torch.argmax(x, dim)	返回最大值的索引（指定维度）
    torch.softmax(x, dim)   计算softmax（指定维度）

#### 形状操作
    x.view(shape)	改变张量的形状（不改变数据）
    x.reshape(shape)	类似于 view，但更灵活
    x.t()	转置矩阵
    x.unsqueeze(dim)	在指定维度添加一个维度，如x从[N]变成[N,1]
    x.squeeze(dim)	去掉指定维度为 1 的维度
    torch.cat((x, y), dim)	按指定维度连接多个张量

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
#### 自定义块
```python
# 可以在自定义块中定义模型参数,并在forward函数中使用这些参数
# 要实现各种层的嵌套,既可以在自定义块的forward函数中嵌套,也可以通过sequential函数来实现
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        super().__init__()# 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

#### 自定义顺序块
```python
# 对应nn.Sequential函数
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module# 该属性存放了各个连接模块的ID

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)# 按顺序传递值
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# 将两个全连接层与一个ReLU层连接在一起
```

#### 参数管理
```python
# 参数访问
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print(net[2].state_dict())# 访问nn.Linear(8, 1)的参数
print(*[(name, param.shape) for name, param in net.named_parameters()])# 访问所有参数
```

#### 参数初始化
##### Xavier初始化原理
目的：使得每层的方差相同，从而使得每层的输出方差不变，从而使得每层的输出不受其他层影响。

全连接层输出为oi，该层输入数量为Nin，输出数量为Nout，输入表示为xj，权重表示为wij(不考虑偏置项).

权重wij都是从同一分布中独立抽取的，该分布具有零均值和方差σ2。这并不意味着分布必须是高斯的。
现在,让我们假设层xj的输入也具有零均值和方差γ2,它们独立于wij并且彼此独立。

将输出进行表示：
\[o_i = \sum_{j=1}^{N_{in}} w_{ij} x_j\]
则其均值为：
\[E[o_i] = \sum_{j=1}^{N_{in}} E[w_{ij} x_j] = \sum_{j=1}^{N_{in}} E[w_{ij}] E[x_j] = 0\]

方差为：
\[Var[o_i] = \sum_{j=1}^{N_{in}} Var[w_{ij} x_j] = \sum_{j=1}^{N_{in}} E[w_{ij}^2 x_j^2] - 0
= \sum_{j=1}^{N_{in}} E[w_{ij}^2]E[x_j^2]=N_{in}σ^2γ^2\]

保持方差不变的一种方法是设置$N_{in}σ^2 = 1$;  
对于反向传播$N_{out}σ^2 = 1$,否则梯度的方差可能会增大.

**因此,需要满足$\frac{1}{2}(N_{in}+N_{out})σ^2 = 1$**

最终确定方差范围后,wij可以从高斯分布或均匀分布中进行采样。
高斯分布采样范围：
\[w_{ij} \sim \mathcal{N}(0, \sqrt{\frac{2}{N_{in}+N_{out}}})\]
均匀分布采样范围：
\[w_{ij} \sim \mathcal{U}(-\sqrt{\frac{6}{N_{in}+N_{out}}}, \sqrt{\frac{6}{N_{in}+N_{out}}})\]
```python
nn.init.xavier_uniform_(net.weight)# 函数会自行计算范围，只需传入要初始化的网络权重
```

##### 关于net.apply
```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
```
>对于net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))使用net.apply(init_normal)和init_normal(net)会有什么区别?  

net.apply(init_normal)会递归地遍历模型中的每一层，并对每一层调用init_normal函数;  
而init_normal(net)把整个net模型作为参数传递过去,又因为net的类型是sequential,所以会导致类型错误.

#### 自定义层
```python
# 不带参数的自定义层
# 功能:将输入减去均值,不需要指定网络参数,自适应输入的形状
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

# 带参数的自定义层
# 创建了一个具有in_units输入单元数和out_units输出单元数的线性层,并通过ReLU后输出
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, out_units))
        self.bias = nn.Parameter(torch.randn(out_units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

#### 保存和加载模型
```python
# 保存模型(张量、list、dict等均可)
net = MLP()
torch.save(net.state_dict(), 'net.params')
# 加载模型
clone_net = MLP()
clone_net.load_state_dict(torch.load('net.params'))
```

## 数据加载与处理
### Dataset & Dataloader
torch.utils.data.Dataset：数据集的抽象类，需要自定义并实现 __len__（数据集大小）和 __getitem__（按索引获取样本）  
torch.utils.data.DataLoader：封装 Dataset 的迭代器，提供批处理batch_size、数据打乱shuffle=True、多线程加载num_workers等功能，便于数据输入模型训练  

```python
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data, labels):
        # 数据初始化
        self.data = data
        self.labels = labels

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 按索引返回数据和标签
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# 生成示例数据
data = torch.randn(100, 5)  # 100 个样本，每个样本有 5 个特征
labels = torch.randint(0, 2, (100,))  # 100 个标签，取值为 0 或 1

# 实例化数据集
dataset = MyDataset(data, labels)

# 创建 DataLoader 实例，batch_size 设置每次加载的样本数量
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
# 遍历 DataLoader
for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"批次 {batch_idx + 1}")
    print("数据:", batch_data)
    print("标签:", batch_labels)
    if batch_idx == 2:  # 仅显示前 3 个批次
        break
```

### 数据转换
torchvision.transforms提供了基本的数据预处理（如归一化、大小调整等），还能帮助进行数据增强（如随机裁剪、翻转等），提高模型的泛化能力。  

基础变换操作：  
    transforms.ToTensor()	将PIL图像或NumPy数组转换为PyTorch张量，并自动将像素值从[0, 255]归一化到 [0, 1]。	transform = transforms.ToTensor()
    transforms.Normalize(mean, std)	对图像进行标准化，使数据符合零均值和单位方差。	transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transforms.Resize(size)	调整图像尺寸，确保输入到网络的图像大小一致。	transform = transforms.Resize((256, 256))
    transforms.CenterCrop(size)	从图像中心裁剪指定大小的区域。	transform = transforms.CenterCrop(224)

数据增强操作：
    transforms.RandomHorizontalFlip(p)	随机水平翻转图像。	transform = transforms.RandomHorizontalFlip(p=0.5)
    transforms.RandomRotation(degrees)	随机旋转图像。	transform = transforms.RandomRotation(degrees=45)
    transforms.ColorJitter(brightness, contrast, saturation, hue)	调整图像的亮度、对比度、饱和度和色调。	transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    transforms.RandomCrop(size)	随机裁剪指定大小的区域。	transform = transforms.RandomCrop(224)
    transforms.RandomResizedCrop(size)	随机裁剪图像并调整到指定大小。	transform = transforms.RandomResizedCrop(224)

自定义转换：
```python
class CustomTransform:
    def __call__(self, x):
        # 这里可以自定义任何变换逻辑
        return x * 2

transform = CustomTransform()
```

组合变换：
    transforms.Compose()	将多个变换组合在一起，按照顺序依次应用。	
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize((256, 256))])

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

## 多层感知机MLP
### 多层感知机的简洁实现
```python
# 网络部分写法
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

### 权重衰减
权重衰减(weight decay)也被称为L2正则化。使得模型参数不会过大,从而控制复杂度。正则项的权重λ是控制模型复杂度的超参数。

目标函数 = 损失函数 + 正则项：  
\[{\rm{L(w, b)  +  }}\frac{\lambda }{2}{\left\| w \right\|^2}\]
使用L2范数的一个原因是它对权重向量的大分量施加了巨大的惩罚。这使得我们的学习算法偏向于在大量特征上均匀分布权重的模型。在实践中,这可能使它们对单个变量中的观测误差更为稳定。

![wdecay.png](post/instruction/pytorch/wdecay.png)
坐标轴对应w的取值
绿点表示L(w, b)的最优点,坐标轴原点表示L2范数的最小值.因此距离这两个点越远,惩罚越大.
黄点是两者相制衡得到的惩罚函数最小值,即权重衰减的效果.

更新权重:
\[{{\rm{w}}_{{\rm{t}} + 1}} \leftarrow (1 - \eta \lambda ){{\rm{w}}_{\rm{t}}} - \frac{\eta }{B}\sum {\frac{{\partial L(w,b)}}{{\partial w}}} \]
注意$(1 - \eta \lambda)$处,表示先对wt做衰减,在进行更新.

### 丢弃法(dropout)
对每个中间活性值h以暂退概率p由随机变量h′替换。有概率p置零，其余概率扩大1-p倍，从而保证均值不变。
\[h' = \begin{cases}\frac{h}{1-p}, &\text{以概率 }1-p \\ 0, &\text{以概率 }p\end{cases}\]
如果通过许多不同的暂退法遮盖后得到的预测结果都是一致的,那么我们可以说网络发挥更稳定。

代码实现：
```python
# dropout层，连接在全连接层之后，对输出进行丢弃
def dropout_layer(X, pdropout):  
    assert 0 <= pdropout <= 1  
    # p=1,所有元素都被丢弃  
    if pdropout == 1:  
        return torch.zeros_like(X) 
    # p=0,所有元素都被保留  
    if pdropout == 0:  
        return X  
    mask = (torch.rand(X.shape) > pdropout).float()  
    return mask * X / (1.0 - pdropout)

# 简洁调用
nn.Dropout(pdropout)
```


## 卷积神经网络CNN
CNN专门用于处理具有网格状拓扑结构数据（如图像）  
部分主要流程：(卷积-池化)*N-展平-分类
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层：输入1通道，输出32通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 定义卷积层：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入大小 = 特征图大小 * 通道数
        self.fc2 = nn.Linear(128, 10)  # 10 个类别

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = x.view(-1, 64 * 7 * 7) # 展平操作
        x = F.relu(self.fc1(x))    # 全连接层 + ReLU
        x = self.fc2(x)            # 全连接层输出
        return x
```

## 循环神经网络RNN
RNN专门用于处理序列数据，能够捕捉时间序列或有序数据的动态信息，如文本、时间序列或音频  
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        # 定义 RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, hidden_size)
        # 取序列最后一个时间步的输出作为模型的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # 全连接层
        return out
```

## Transformer
实际使用时，可以直接调用nn.embedding、nn.Transformer、nn.positional_encoding层构成模型，无需自己编写  
```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))  # 假设序列长度最大为1000
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src_seq_length, tgt_seq_length = src.size(1), tgt.size(1)
        src = self.embedding(src) + self.positional_encoding[:, :src_seq_length, :]# 取实际序列长度的部分
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt_seq_length, :]
        transformer_output = self.transformer(src, tgt)
        output = self.fc(transformer_output)
        return output
```
下面为自己实现各模块：  
### 注意力机制
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model    # 模型维度（如512）
        self.num_heads = num_heads # 注意力头数（如8）
        self.d_k = d_model // num_heads # 每个头的维度（如64）
        
        # 定义线性变换层（无需偏置）
        self.W_q = nn.Linear(d_model, d_model) # 查询变换
        self.W_k = nn.Linear(d_model, d_model) # 键变换
        self.W_v = nn.Linear(d_model, d_model) # 值变换
        self.W_o = nn.Linear(d_model, d_model) # 输出变换
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        输入形状：
            Q: (batch_size, num_heads, seq_length, d_k)
            K, V: 同Q
        输出形状： (batch_size, num_heads, seq_length, d_k)
        """
        # 计算注意力分数（Q和K的点积）
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如填充掩码或未来信息掩码）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重（softmax归一化）
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 对值向量加权求和
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        """
        将输入张量分割为多个头
        输入形状: (batch_size, seq_length, d_model)
        输出形状: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """
        将多个头的输出合并回原始形状
        输入形状: (batch_size, num_heads, seq_length, d_k)
        输出形状: (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        输入形状: Q/K/V: (batch_size, seq_length, d_model)
        输出形状: (batch_size, seq_length, d_model)
        """
        # 线性变换并分割多头
        Q = self.split_heads(self.W_q(Q)) # (batch, heads, seq_len, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头并输出变换
        output = self.W_o(self.combine_heads(attn_output))
        return output
```
### 位置编码
```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        # 前馈网络的计算
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区
        
    def forward(self, x):
        # 将位置编码添加到输入中
        return x + self.pe[:, :x.size(1)]
```
### Encoder
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 自注意力机制
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout
        
    def forward(self, x, mask):
        # 自注意力机制
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接和层归一化
        return x
```

### Decoder
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 自注意力机制
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # 交叉注意力机制
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力机制
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        # 交叉注意力机制
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # 残差连接和层归一化
        return x
```

### 整体架构
```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # 编码器词嵌入
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # 解码器词嵌入
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)  # 位置编码

        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)  # 最终的全连接层
        self.dropout = nn.Dropout(dropout)  # Dropout

    def generate_mask(self, src, tgt):
        # 源掩码：屏蔽填充符（假设填充符索引为0）
        # 形状：(batch_size, 1, 1, seq_length)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
        # 目标掩码：屏蔽填充符和未来信息
        # 形状：(batch_size, 1, seq_length, 1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # 生成上三角矩阵掩码，防止解码时看到未来信息
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask  # 合并填充掩码和未来信息掩码
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器部分
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # 解码器部分
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 最终输出
        output = self.fc(dec_output)
        return output
```


## 优化算法

## 计算机视觉

