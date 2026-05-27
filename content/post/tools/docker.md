+++
title = 'Docker'
date = 2026-04-18T12:00:00+08:00
draft = false
categories = ["tools"]
+++

## Docker 基本工作原理

- 镜像是模板，容器是镜像运行后的实例。
- 同一镜像可以启动多个相互独立的容器。
- Docker 通过进程、文件系统、网络等机制实现隔离。
- 重要数据不要只放在容器可写层里，通常要通过挂载或 volume 持久化。
- 日常最常用的命令集中在镜像管理、容器启停、日志查看、端口映射和目录挂载。

### 镜像与容器
- 镜像（image）可以理解成一个只读模板，里面包含了程序代码、依赖、基础环境和启动命令。
- 容器（container）是镜像运行起来后的实例，可以把它看成“镜像 + 一个可写层 + 运行中的进程”。

### 同一镜像的不同容器之间是什么关系
- 同一个镜像可以创建多个容器。
- 这些容器共享同一个镜像内容，但每个容器都有自己独立的可写层和运行状态。
- 一个容器里新建了文件、改了配置、停掉了进程，通常不会直接影响另一个容器。

### Docker 如何实现数据隔离
Docker 主要通过 Linux 的隔离机制来实现容器之间彼此独立，常见包括：
- 进程隔离：每个容器看到的是相对独立的进程空间。
- 文件系统隔离：每个容器有自己的可写层，看起来像独立文件系统。
- 网络隔离：容器通常有自己的网络环境和端口映射关系。
- 资源限制：可以限制容器使用多少 CPU、内存等资源。

## 数据文件夹与共享文件夹挂载
容器本身适合跑程序，但不适合直接长期保存重要数据，因为容器删掉后，容器可写层里的数据通常也就没了。  
因此实际使用时经常会把宿主机目录或 Docker 数据卷挂载到容器里。

### 1. 挂载宿主机目录
把宿主机上的一个文件夹直接映射到容器里：

```bash
docker run -d --name mynginx -p 8080:80 -v /host/html:/usr/share/nginx/html nginx
```

含义是：
- 宿主机的 `/host/html`
- 挂载到容器内的 `/usr/share/nginx/html`

这样容器读写这个目录时，本质上操作的是宿主机上的文件。  
这很适合做“代码目录共享”“配置文件共享”“日志导出”等场景。

### 2. 挂载 Docker 数据卷
数据卷由 Docker 管理，更适合存数据库数据之类的持久化内容：

```bash
docker volume create mydata
docker run -d --name mydb -v mydata:/var/lib/mysql mysql
```

这样即使容器删除，只要数据卷还在，数据通常还在。

### 数据文件夹和共享文件夹的区别
- 数据文件夹：更强调数据持久化，比如数据库文件、上传文件。
- 共享文件夹：更强调宿主机和容器之间共同访问，比如把本地项目目录挂进去方便开发。

简单理解：
- 想长期保存数据，优先考虑 volume。
- 想让本机目录和容器实时同步，常用 bind mount，也就是 `-v 宿主机路径:容器路径`。

## 常用 Docker 命令
### 镜像相关
```bash
docker pull nginx
docker images
docker rmi nginx
docker build -t myapp:latest .
```

- `docker pull`：拉取镜像
- `docker images`：查看本地镜像
- `docker rmi`：删除镜像
- `docker build`：根据 Dockerfile 构建镜像

### 容器相关
```bash
docker run -d --name myapp -p 8000:8000 myapp:latest
docker ps
docker ps -a
docker stop myapp
docker start myapp
docker restart myapp
docker rm myapp
```

- `docker run`：创建并启动容器
- `docker ps`：查看正在运行的容器
- `docker ps -a`：查看所有容器
- `docker stop`：停止容器
- `docker start`：启动已停止容器
- `docker restart`：重启容器
- `docker rm`：删除容器

### 进入容器与查看日志
```bash
docker exec -it myapp /bin/bash
docker logs myapp
docker logs -f myapp
```

- `docker exec -it`：进入正在运行的容器
- `docker logs`：查看容器日志
- `docker logs -f`：持续跟踪日志输出

### 挂载和端口映射
```bash
docker run -d -p 8080:80 nginx
docker run -d -v /host/data:/data myapp
```

- `-p 主机端口:容器端口`：端口映射
- `-v 宿主机路径:容器路径`：目录挂载

