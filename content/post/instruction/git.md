---
title: 'Git'
date: 2024-08-12T18:36:30+08:00
draft: false
categories:
    - 指令语法

---

## 初始化
```bash
    git init  
    git add .  
    git commit -m "first commit"  
    git branch -M main  
    git remote add origin {你的github仓库地址}  
    git push -u origin main
```

## 克隆
```bash
    git clone {github仓库地址} -c http.proxy="http://127.0.0.1:7890"
    # 下载大文件时
    git lfs install
    git lfs clone {github仓库地址}
```
## 后续上传
```bash
    git add .  
    git commit  
    git push //若要无视pull则添加(--force)
```
第一行将修改同步至本地缓存区，第二行将缓存区的修改提交至本地仓库，第三行将本地仓库的修改推送至远端仓库。

## 分支
### 建立分支
```bash
    git checkout -b [branchname] //git 切换分支 建立并切换至新分支 新分支名
```
### 推送分支
```bash
    git push origin [branchname]  //git 推送 远端名称 本地分支名
```
## 版本回退
```bash
    git log //查看各提交版本  
    git reset [head~n/commit ID] --soft/hard  //回退n个版本/回退到ID表示的版本  
```
>详细见：<https://www.bilibili.com/video/BV14C4y1q78x>
### 文件回退
```bash
    git checkout [版本ID] --文件名
```
## gitignore失效
使用git rm (-r) --cached file_path移除与缓存的连接
>[失效解决方法](<https://www.cnblogs.com/goloving/p/15017769.html>)