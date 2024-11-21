---
title = 'Clash'
date = 2024-11-15T14:01:14+08:00
draft = false
categories=['conclusion']
---

## Clash安装及使用

1. 下载Clash for Windows：(<https://github.com/clashdownload/Clash_for_Windows>)

2. 在官网推荐的网站订阅节点（需检查clash core为premium核心/最下面的几个代理开关全部关闭），并复制链接导入配置文件

3. 启用配置，选择节点，打开主页的系统代理开关。测试是否可用

4. 在防火墙开放7890端口的接入规则，打开Allow LAN，设置port为7890，本机就可以当作代理服务器使用

5. 在linux命令行中使用curl --proxy http://本机IPv4地址:7890 www.google.com测试是否成功

6. 在命令行/.sh文件中使用
    export http_proxy=http://本机IPv4地址:7890 
    export https_proxy=https://本机IPv4地址:7890

7. 在.py文件中使用
    os.environ['http_proxy'] = 'http://本机IPv4地址:7890'
    os.environ['https_proxy'] = 'https://本机IPv4地址:7890'

