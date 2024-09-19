+++
title = 'Pytorch'
date = 2024-08-22T16:49:29+08:00
draft = false
categories = ['package']
+++

## Pytorch安装

## Jupyter Notebook

### 给notebook添加kernel
```bash
$ conda create -n testEnv python=3.6
$ conda activate testEnv
(testEnv)$ conda install ipykernel
(testEnv)$ ipython kernel install --user --name=testEnv
(testEnv)$ jupyter notebook
```
### 修改默认启动路径
    jupyter notebook --generate-config
查看配置文件路径
修改jupyter_notebook_config.py文件中内容：
c.ServerApp.notebook_dir