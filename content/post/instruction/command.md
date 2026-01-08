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
- mkdir -p <d1/d2>：创建目录（-p：创建多级目录）
- touch：创建文件
- cat：查看文件内容(不如vim)
- echo "text" > file.txt：将文本**覆盖**写入文件
- echo "text" >> file.txt：将文本追加到文件末尾
- grep <path> "text" filename：在文件中查找指定文本的行
- find . -name "filename"：在当前目录及子目录中查找文件
- cp：复制文件或目录
- mv：移动或重命名文件或目录
- rsync -av --progress source/ destination/：同步文件夹（-a：归档模式，保留权限等信息；-v：显示详细信息；--progress：显示进度）
- rm -rf：删除文件或目录（-r：递归删除文件夹及其下文件；-f：强制删除所有属性的文件，包括只读）
- df -h：查看**磁盘**使用
- du -sh --max-depth=1 <directory>：查看**目录**下总大小

### 用户与权限
- sudo：以超级用户身份执行命令
- su：切换用户身份
- chmod：修改文件或目录权限
- who：查看登录用户

### 进程管理
- ps -ef | grep <PID>:可以查看子父进程之间的关系
- ps -o ppid,cmd -p PID：查看指定进程的父进程和命令
- top [-d -i -p]：实时显示进程信息(-d：更新显示；-i：不显示闲置进程；-p：指定进程ID)
- iotop -oPa：实时显示磁盘I/O使用情况(-o：只显示有I/O的进程；-P：显示累积I/O；-a：显示累计I/O)
- sudo ionice -c 2 -n 0 -p PID：设置进程I/O优先级（-c：调度类，2为最佳努力；-n：优先级，0最高，7最低）
- iostat：显示系统I/O统计信息
- screen：创建会话,保持长期运行的进程  
    screen [-S <session_name>] <command>:创建会话并运行指令（命名可选）  
    screen -ls：SSH重连后查看会话  
    screen -r <session_name>  
- kill -s PID：杀死进程  
    -9:强制杀死进程  
    -15:正常杀死进程  
    杀死指定用户进程：kill -9 $(ps -ef | grep user_name)或kill -u user_name

### 网络管理
- ifconfig：查看网络接口信息
- netstat：查看网络连接信息
- ssh：远程登录
- scp：远程复制文件
- wget：代理下载文件
    wget <link> -e "https_proxy=http://"
- curl：发送HTTP请求
    curl www.google.com --proxy http://
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


## pip
- pip install <package_name> -i <镜像源> --proxy=http:// --no-deps(不自动调整其他包版本)：安装包
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

### 源管理
- conda config --add channels url：添加源
- conda config --remove channels url：删除源

## Docker
镜像是用于创建容器的只读模板。 容器在镜像上添加了一个可写层，是镜像的一个运行实例。
### 镜像操作
- docker pull image_name:tag：拉取镜像
- docker images：查看本地镜像
- docker rmi image_id：删除镜像
- docker build -t image_name:tag .：从Dockerfile构建镜像

### 容器操作
- docker run <-it> --name container_name image_name:tag -e ENV_VAR=value -p <host_port>:<container_port> /bin/bash：运行容器并进入bash
    - -it：交互式终端
    - -d：后台运行
    - -e：设置环境变量
    - -p：端口映射
- docker ps -a：查看所有容器 -a会显示已停止的容器
- docker stop container_id：停止容器
- docker start container_id：启动容器
- docker rm container_id：删除容器
- docker exec -it container_id /bin/bash：进入运行中的容器bash
