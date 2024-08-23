+++
title = 'Command'
date = 2024-08-22T14:26:16+08:00
draft = false
categories = ['指令语法']
+++


# 命令行指令

## linux

- ls：列出当前目录下的文件和目录
- cd：切换目录
- mkdir：创建目录
- touch：创建文件
- rm：删除文件或目录
- mv：移动或重命名文件或目录
- cp：复制文件或目录
- cat：查看文件内容
- sudo：以超级用户身份执行命令
- su：切换用户身份
- chmod：修改文件或目录权限
- ps：查看进程
- top：实时显示进程信息
- kill：杀死进程
- ifconfig：查看网络接口信息
- netstat：查看网络连接信息
- ssh：远程登录
- scp：远程复制文件
- git：版本控制
- vim：编辑器
- tmux：终端复用工具

## conda

- conda create -n env_name python=xx：创建环境
- conda create --name new_env --clone copied_env：复制环境
- conda env list：查看环境
- conda activate env_name：激活环境
- conda deactivate：退出环境
- conda remove -n env_name --all：删除环境

- conda config --add channels url：添加源
- conda config --remove channels url：删除源

- conda list：查看已安装的包
- conda install package_name (-c channel_name)：安装包(从指定源安装)
- conda remove package_name：删除包
- conda search package_name：搜索包
- conda update package_name：更新包

