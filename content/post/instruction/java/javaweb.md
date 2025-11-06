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

## Web前端
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
1. 引入方式
```html
<script src="script.js"></script>
```
2. JSON格式：
```json
{
  "name": "John",
  "age": 30,
  "isStudent": false,
  "courses": ["Math", "Science"],
  "address": {
    "street": "123 Main St",
    "city": "Anytown"
  }
}
```
3. DOM（文档对象模型）
是HTML和XML文档的JS编程接口。它将文档表示为一个树结构，其中每个节点表示文档的一部分（例如元素、属性或文本）。

- **访问节点**：通过`document`对象访问文档中的元素，例如`document.getElementById()`、`document.querySelector('选择器')`。
- **修改节点**：可以更改节点的内容、属性或样式，例如`element.innerHTML`、`element.style`。
- **事件处理**：可以为节点添加事件监听器，例如`element.addEventListener("click", function)`。
- **创建和删除节点**：可以动态添加或移除文档中的元素，例如`document.createElement()`、`parentNode.appendChild()`、`parentNode.removeChild()`。

```javascript
document.getElementById("myElement").innerText = "Hello, World!";
```

4. 事件监听
```javascript
document.getElementById("myButton").addEventListener("click", function() {
    alert("Button clicked!");
});
```
常见事件类型：
- 鼠标事件：click、dblclick、mouseenter、mouseleave
- 键盘事件：keydown键盘按下、keyup键盘抬起、keypress
- 表单事件：submit表单提交、change值改变、focus获得焦点、blur失去焦点

### Vue
Vue 是一个用于构建用户界面的渐进式 JavaScript 框架。它的核心库专注于视图层，易于上手，并与其他库或现有项目轻松集成。

1. 基础格式
```html
<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
<div id="app">
  {{ message }}
</div>
<script>
  const app = Vue.createApp({
    data() {
      return {
        message: 'Hello Vue!'
      }
    }
  });
  app.mount('#app');// 将 Vue 应用挂载到具有 id "app" 的 DOM 元素上
</script>
```

2. 常用函数
- `v-for`: 列表渲染
- `v-bind`: 动态绑定 HTML 属性
- `v-model`: 创建双向数据绑定
- `v-if` / `v-else-if` / `v-else`: 条件渲染（是否创建）
- `v-show`: 条件显示（创建后选择是否显示）
- `v-on`: 事件监听

3. 生命周期钩子函数
- `created`: 实例创建后调用
- `mounted`: 实例挂载到 DOM 后调用
- `updated`: 数据更新后调用

### AJAX
在不重新加载整个网页的情况下，与服务器交换数据并更新部分网页内容。

基本使用示例：
```javascript
<script src="https://unpkg.com/axios/dist/axios.min.js"></script>
axios.get('https://api.example.com/data')
  .then(function (response) {// then异步处理成功回调
    console.log(response.data);
  })
  .catch(function (error) {
    console.error('Error fetching data:', error);
  });
```

async/await 语法：
将异步代码变成同步
```javascript
async function fetchData() {
  try {
    const response = await axios.get('https://api.example.com/data');// await 等待异步操作完成
    console.log(response.data);
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}
fetchData();
```

## Web后端
### Maven
Maven 是一个项目管理和构建自动化工具，主要用于 Java 项目。它使用 pom.xml 文件来管理项目的构建、依赖和文档。

项目结构
```
my-app
├── pom.xml // 项目对象模型文件，包含项目信息和配置
└── src // 源代码目录
    ├── main
    │   ├── java
    │   │   └── com
    │   │       └── mycompany
    │   │           └── app
    │   │               └── App.java
    │   └── resources // 资源文件目录
    └── test // 测试代码目录
        ├── java
        └── resources
```