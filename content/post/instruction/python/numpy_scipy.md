+++
title = 'Numpy & Scipy'
date = 2025-03-13T09:48:39+08:00
draft = false
categories = ['指令语法']
+++

## Numpy


## Scipy
### 常量模块constants
返回各种数学常数、各类单位的值，如pi，单位等
```python
from scipy import constants
print(constants.pi)
```

### 优化器optimize
最小化函数: 使用 minimize 方法对多种优化算法进行统一接口封装。  
求解方程的根: 使用 root 方法求解非线性方程或方程组。  
曲线拟合: 使用 curve_fit 进行非线性最小二乘拟合。  
线性规划: 使用 linprog 解决线性规划问题。  

### 稀疏矩阵sparse
指大部分元素都为0的矩阵
CSR(Compressed Sparse Row)矩阵是一种高效的稀疏矩阵存储格式,主要用于存储和计算大型稀疏矩阵。

主要特点:
- 只存储非零元素及其位置信息
- 按行压缩存储
- 适合进行矩阵-向量乘法运算

基本用法示例:
```python
import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])

print(csr_matrix(arr))
输出结果为：  (0, 5) 1    (0, 6) 1    (0, 8) 2
```

### 图结构
邻接矩阵（Adjacency Matrix）：由两部分组成：V 是顶点，E 是边，边有时会有权重，表示节点之间的连接强度。  
组成：用一个一维数组存放图中所有顶点数据，用一个二维数组存放顶点间关系（边或弧）的数据，这个二维数组称为**邻接矩阵**。

```python
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# 创建一个3x3的邻接矩阵，表示一个无向图
arr = np.array([
  [0, 1, 2],  # 节点0的连接情况
  [1, 0, 0],  # 节点1的连接情况
  [2, 0, 0]   # 节点2的连接情况
])
# 将邻接矩阵转换为CSR格式的稀疏矩阵
newarr = csr_matrix(arr)

# 使用connected_components函数找出图中的连通分量
# 返回值包含两个部分：
# 1. 连通分量的数量
# 2. 每个节点所属的连通分量编号
print(connected_components(newarr))# (1, array([0, 0, 0], dtype=int32))
```

### 空间数据
#### 三角测量
多边形的三角测量是将多边形分成多个三角形。  

任何曲面都存在三角剖分。  
假设曲面上有一个三角剖分， 我们把所有三角形的顶点总个数记为 p(公共顶点只看成一个)，边数记为 a，三角形的个数记为 n，则欧拉示性数e=p-a+n 是曲面的拓扑不变量
```python
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1]
])
simplices = Delaunay(points).simplices    # 三角形中顶点的索引

plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')

plt.show()
```

#### 各类距离
·欧几里得距离：m维空间中两个点之间的真实距离
```python
from scipy.spatial.distance import euclidean

p1 = (1, 0)
p2 = (10, 2)
res = euclidean(p1, p2)

print(res)
```

·曼哈顿距离：只能上、下、左、右四个方向进行移动
```python
from scipy.spatial.distance import cityblock

p1 = (1, 0)
p2 = (10, 2)
res = cityblock(p1, p2)

print(res)
```
余弦距离：也称为余弦相似度，通过测量两个向量的夹角的余弦值来度量它们之间的相似性
```python
from scipy.spatial.distance import cosine

p1 = (1, 0)
p2 = (10, 2)
res = cosine(p1, p2)

print(res)
```
汉明距离：两个字符串对应位置的不同字符的个数
```python
from scipy.spatial.distance import hamming

p1 = (True, False, True)
p2 = (False, True, True)
res = hamming(p1, p2)

print(res)
```


### 插值
通过已知的、离散的数据点，在范围内推求新数据点的过程或方法。  
一维插值：方法 interp1d()  
返回一个可调用函数，该函数可以用新的 x 调用并返回相应的 y
```python
from scipy.interpolate import interp1d
import numpy as np

xs = np.arange(10)# 限定输入范围
ys = 2*xs + 1# 计算方法
interp_func = interp1d(xs, ys)

newarr = interp_func(np.arange(2.1, 3, 0.1))# 插值范围及步长

print(newarr)# [5.2  5.4  5.6  5.8  6.   6.2  6.4  6.6  6.8]
```