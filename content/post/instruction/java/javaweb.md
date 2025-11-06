+++
title = 'Java Web'
date = 2025-09-13T09:48:39+08:00
draft = false
categories = ['指令语法']
+++

## 目录
- 网页基础
- Servlet 原理（生命周期、请求响应、过滤器、监听器）
- JSP 简单了解（现代项目几乎不用）
- Tomcat 部署
- MVC 思想（为Spring MVC铺路）

## 网页基础
### HTTP 协议基础（请求行、请求头、请求体、响应行、响应头、响应体）
常见状态码（200、301、302、404、500）: 2xx表示成功，3xx表示重定向，4xx表示客户端引发的错误，5xx表示服务器端引发的错误

### HTML 基础
基本结构示例：
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>

</body>
</html>
```
标签不区分大**小**写，属性值用单**双**引号均可

#### 常用标签
- 标题标签：`<h1>`到`<h6>`
- 段落标签：`<p>`
- 链接标签：`<a href="https://example.com" target="_blank">Example</a>`  
  链接地址使用`href`属性，打开方式使用target属性（_blank新窗口打开，_self当前窗口打开）   
- 图像标签：`<img src="image.jpg" alt="Description">`  
  图像地址使用`src`属性，`alt`属性用于图像无法显示时的替代文本
- 列表标签：`<ul>`、`<ol>`、`<li>`
- 表格标签：`<table>`、`<tr>`、`<td>`、`<th>`
- 表单标签：`<form>`、`<input>`、`<textarea>`

#### CSS
- 内联样式：使用`style`属性直接在HTML标签中定义样式
- 内部样式表：使用`<style>`标签在HTML文档的`<head>`部分定义样式
- 外部样式表：使用`<link>`标签链接外部CSS文件

```html
<link rel="stylesheet" href="styles.css">
</code>
```
CSS选择器
- 元素选择器：选择所有指定的元素，例如`p`选择所有段落
- 类选择器：选择所有指定类的元素，例如`.className`选择所有类名为className的元素
- ID选择器：选择指定ID的元素，例如`#idName`选择ID为idName的元素
- 属性选择器：选择具有指定属性的元素，例如`[type="text"]`选择所有type属性为text的输入框
```css
p {font-size: 16px;}
[class="highlight"] {color: red;}
#header {background-color: blue;}
[type="text"] {background-color: lightyellow;}
```

### JavaScript 基础
#### 引入方式
```html
<script src="script.js"></script>
```