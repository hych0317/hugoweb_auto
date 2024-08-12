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

### 分支
#### 建立分支
    git checkout -b [branchname] //git 切换分支 建立并切换至新分支 新分支名
#### 推送分支
    git push origin [branchname]  //git 推送 远端名称 本地分支名

### 版本回退
    git log //查看各提交版本  
    git reset [head~n/commit ID] --soft/hard  //回退n个版本/回退到ID表示的版本  
>详细见：<https://www.bilibili.com/video/BV14C4y1q78x>
#### 文件回退
    git checkout [版本ID] --文件名

### gitignore失效
使用git rm (-r) --cached file_path移除与缓存的连接
>[失效解决方法]<https://www.cnblogs.com/goloving/p/15017769.html>