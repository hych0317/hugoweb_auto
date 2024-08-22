import socket
import threading

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = socket.gethostname()

port = 9999
s.connect((host, port))
name = "cc"
s.send(name.encode("utf-8"))


def receive_handle(sock, addr):
    while True:
        data = sock.recv(1024)
        print(data.decode("utf-8"))

# 开启线程监听接收消息
receive_thread = threading.Thread(target=receive_handle, args=(s, '1'))
receive_thread.start()

while True:
    re_data = input()
    s.send(re_data.encode("utf-8"))