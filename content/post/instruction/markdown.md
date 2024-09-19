---
title: 'Markdown常用语法'
date: 2024-08-12T14:36:30+08:00
draft: false
categories:
    - 指令语法

---

>官方语法教程：<https://markdown.com.cn/basic-syntax/>  
## 标题
（#）数量代表几级标题，数量越小字体越大

## 段落
使用空白行分割文本，例如：

此时文本被分段。（注意：段落的首行均不可缩进，以免导致编译错误；分段与换行不同）

## 换行
不是简单的回车，应在一行的末尾添加两个或多个空格，然后再按回车键(或者使用<br>)。

## 强调
加粗两边用两个星号（**粗体**）  
加斜两边用一个星号（*斜体*）

## 引用
使用>号，使用>>可以嵌套引用

## 列表
使用1. 2. 3.进行有序列表  
使用星号，破折号-进行无序列表  
在列表中缩进tab可以嵌套包括列表项的其它元素，如：
* 一
* 二
    * 嵌套1
    * 嵌套2
* 三
    >引用1
* 四

## 代码块
1. 缩进tab表示代码块
2. 代码块前后分别用三个反引号```表示，同时可以使用语言标识符指定代码类型，如：  

python代码块：
```python（或yaml、json等）
print("Hello, world!")
```
yaml代码块：
```yaml
links:
  - title: My GitHub
    description: GitHub is the world's largest software development platform.
    website: https://github.com
    image: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png
```
这样可以使用复制粘贴代码块功能，并能高亮显示代码类型。



## 图片
    ![描述](路径 "可选title")
![example](post/instruction/als.png)  
给图片添加链接  

    [![描述](路径)](链接) //图片链接

## 链接
    [描述](<网址>) //网址间空格应使用%20替换
<>使得网址可点击