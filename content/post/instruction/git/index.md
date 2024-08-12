---
title: 'Git'
date: 2024-08-12T14:34:30+08:00
draft: false
categories:
    - 指令

---


## git指令

### 初始化
    git init  
    git add .  
    git commit -m "first commit"  
    git branch -M main  
    git remote add origin {你的github仓库地址}  
    git push -u origin main

### 后续上传
    git add .  
    git commit  
    git push //若要无视pull则添加(--force)

### 版本回退
    git log //查看各提交版本  
    git reset [head~n/commit ID] --soft/hard  //回退n个版本/回退到ID表示的版本  
>详细见：<https://www.bilibili.com/video/BV14C4y1q78x>

### gitignore失效
使用git rm (-r) --cached file_path移除与缓存的连接
>[失效解决方法]<https://www.cnblogs.com/goloving/p/15017769.html>