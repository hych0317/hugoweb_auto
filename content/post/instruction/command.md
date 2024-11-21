+++
title = 'bash/cmd指令'
description = '包含Linux、Conda、vim'
date = 2024-08-22T14:26:16+08:00
draft = false
categories = ['指令语法']
+++

## Linux
### 文件操作
- ls：列出当前目录下的文件和目录
- cd：切换目录
- mkdir：创建目录
- touch：创建文件
- rm -rf：删除文件或目录（-r：递归删除文件夹及其下文件；-f：强制删除所有属性的文件，包括只读）
- mv：移动或重命名文件或目录
- cp：复制文件或目录
- cat：查看文件内容
- sudo：以超级用户身份执行命令
- su：切换用户身份
- chmod：修改文件或目录权限
- df：查看磁盘使用
- du -sh <directory>：查看目录下总大小

### 进程管理
- ps -ef：查看进程（UID用户标识号，PID进程标识号，PPID父进程标识号，CMD命令）
- ps -ef | grep <PID>:可以查看子父进程之间的关系
- pstree -p PID：查看进程树
- top -d -i -p：实时显示进程信息(-d：更新显示；-i：不显示闲置进程；-p：指定进程ID)
- who：查看登录用户
- kill -s PID：杀死进程
    -9:强制杀死进程
    -15:正常杀死进程
    杀死指定用户进程：kill -9 $(ps -ef | grep user_name)或kill -u user_name

### pip
- pip install <package_name> -i <镜像源> --no-deps(不自动调整其他包版本)：安装包
    镜像源:
    -i https://pypi.tuna.tsinghua.edu.cn/simple/：清华源
    -i https://pypi.doubanio.com/simple/：豆瓣源
    -i https://mirrors.aliyun.com/pypi/simple/：阿里云源

- pip uninstall package_name：卸载包
- pip cache purge：清理缓存

- pip list：查看已安装的包
- pip show package_name：查看包信息
- pip check：检查依赖版本是否一致

- pip list --format=freeze > requirements.txt
    按版本号导出依赖（更建议导出conda环境）
- pip install -r requirements.txt：导入依赖
    对应conda install --file requirements.txt

### 网络管理
- ifconfig：查看网络接口信息
- netstat：查看网络连接信息
- ssh：远程登录
- scp：远程复制文件
- bark通知：
    ```python
    import requests

    r = requests.get('https://api.day.app/yPjmWKqbWBcYB4tWbAVk8/<test title>/<test content>/?group=<test group>/?level=timeSensitive')
    ```

### vim
vim有两种模式：命令模式和编辑模式。

命令模式：按下ESC进入命令模式，可以执行命令，如：
- i：进入编辑模式
- :q!：强制退出并不保存
- :wq：保存并退出
- :set nu：显示行号
- :set nonu：取消显示行号
- :set hlsearch：高亮搜索结果
- :set nohlsearch：取消高亮搜索结果
- :set fileencoding=utf-8：设置文件编码为utf-8

编辑模式：按下i进入编辑模式，按下ESC进入命令模式。


## Conda
### 环境操作
- conda create -n env_name python=xx：创建环境
- conda create -n new_env --clone copied_env：复制环境
    本地复制直接在本地环境文件夹复制并重命名即可
- conda env export > environment.yml：导出环境
- conda env create -f environment.yml：导入环境
- conda env list：查看环境
- conda activate env_name：激活环境
- conda deactivate：退出环境
- conda remove -n env_name --all：删除环境

### 包管理
- conda list：查看已安装的包
- conda install package_name (-c channel_name)：安装包(从指定源安装)
- conda remove package_name：删除包
- conda search package_name：搜索包
- conda update package_name：更新包

#### 验证pytorch安装
1. 在对应虚拟环境中使用conda list查看
2. 输入python / import torch / torch.cuda.is_available()，如果输出True则安装成功，否则失败。

### 源管理
- conda config --add channels url：添加源
- conda config --remove channels url：删除源
