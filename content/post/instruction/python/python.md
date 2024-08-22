+++
title = 'Python'
date = 2024-08-13T09:48:39+08:00
draft = false
categories = ['指令语法']
+++

# Python

## 基础语法

### 数据类型
不可变数据（3 个）：Number（数字）、String（字符串）、Tuple（元组）；  
可变数据（3 个）：List（列表）、Dictionary（字典）、Set（集合）。
>使用type()函数可以查看变量的类型。
>对不可变数据对象重新赋值，实际上是创建了一个新的对象，不是修改了原来的对象。

#### 数据类型转换
对于同大类的数据类型，Python 可以自动进行类型转换。如int+float=float。    
在不同类运算时，Python 中可以使用 int()、float()、str() 函数将其他类型转换为数字、浮点数、字符串。

#### 数字
Python 支持四种不同的数字类型：int、float、bool、complex（复数）。  
>复数由实数部分和虚数部分构成，可以用a + bj,或者complex(a,b)表示， 复数的实部a和虚部b都是浮点型。
>pi,e,tau 都是内置常量。
bool 是 int 的子类，因此布尔值可以被看作整数来使用，其中 True 等价于 1。  
布尔类型可以和其他数据类型进行比较，比如数字、字符串等。在比较时，Python 会将 True 视为 1，False 视为 0。

##### 数字运算
* a**b 表示 a 的 b 次方  
* a//b 表示 a 除以 b 的商  
    > 10//3=3, 10.0//3=3.0  

* a%b 表示 a 除以 b 的余数  
* := 赋值运算符  

        if (n := 10) > 5:   # 赋值表达式
            print(n)
###### 运算函数
abs() 取绝对值  
round(x，n) 四舍五入到 n 位小数  
ceil() 向上取整  
floor() 向下取整  
exp() 计算 e 的 x 次方  
log() 计算自然对数  
log10() 计算以 10 为底的对数  
pow() 计算 x 的 y 次方  
sqrt() 计算平方根  
ord() 获得字符的 ASCII 码  
###### 随机数函数
random() 随机生成 0 到 1 之间的浮点数  
randint(a, b) 随机生成 a 到 b 之间的整数  
choice(seq) 从序列 seq 中随机选择一个元素  
shuffle(seq) 将序列 seq 随机排序  
seed(x) 设置随机数种子  
uniform(a, b) 随机生成 a 到 b 之间的浮点数  
gauss(mu, sigma) 随机生成符合高斯分布的随机数  


##### 成员运算符
in 和 not in 用于判断元素是否存在于列表、元组、字符串、字典等序列中。  

##### 身份运算符
is 和 is not 用于比较两个变量是否指向同一个对象。  


#### 字符串
* Python 中单引号 ' 和双引号 " 使用完全相同  
* 使用三引号 ''' 或 """ 可以指定一个多行字符串(所见即所得) 

        '''This is a 
        multi-line string.
        '''
* 字符串可以用 + 运算符连接在一起，用 * 运算符重复。
* 反斜杠可以用来转义，使用 r 可以让反斜杠不发生转义。 
    
    >如 r"this is a line with \n" 则 \n 会显示，并不是换行。
* Python 中的字符串有两种索引方式，从左往右以 0 开始，从右往左以 -1 开始<br>
 
        str='123456789'

        print(str[0:-1])           # 输出第一个到倒数第二个的所有字符
        print(str[2:])             # 输出从第三个开始后的所有字符
        print(str[1:5:2])          # 输出从第二个开始到第五个且每隔一个的字符（步长为2）
        print(str * 2)             # 输出字符串两次
        print ("我叫%s今年 %d 岁!" % ('小明', 10))
                                   # 格式化输出字符串

##### f-string
    >f'{1+2}'         # 使用表达式
    '3'

    w = {'name': 'Runoob', 'url': 'www.runoob.com'}
    > f'{w["name"]}: {w["url"]}'
    'Runoob: www.runoob.com'
##### 转义字符
|转义字符|含义|
|-------|----|
|\b|退格|
|\t|横向制表符|
|\r|回车|
|\f|换页|
|\ooo|八进制数|
|\xhh|十六进制数|

    print('google runoob taobao\r123456')
    123456 runoob taobao

    八进制      \012 代表换行
    十六进制    \x0a 代表换行

#### 列表
列表是 Python 中最常用的数据结构，用方括号 [] 来表示。  
同一列表可以包含不同类型的数据，可以随时添加或删除元素。 
* 列表可以嵌套列表。   
* 列表可以当作栈、队列、集合使用，特别是pop和append方法。  
![example](post/instruction/python/list1.png)  
    list.append(obj)    # 在列表末尾添加新的对象
    list.extend(seq)     # 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
    list.remove(obj)     # 删除列表中某个值的第一个匹配项
    del list[index]      # 删除指定索引的元素
    list.pop(n)        # 从列表中删除某个元素（默认最后一个元素），并且返回该元素的值
    list.reverse()      # 反转列表



#### 字典
字典是一种映射类型，字典用 { } 标识，它是一个无序的 键(key) : 值(value) 的集合，集合元素间用逗号隔开。  

* 键(key)必须使用不可变类型，比如字符串、数字或元组。  
    键应该是唯一的，重复的键会覆盖前面的键。  

* 值(value)可以是任意类型，包括列表甚至另一个字典？。<br>  

##### 访问字典的值  

    dictexp = {}
    dictexp['one'] = "1 - 菜鸟教程"
    dictexp[2]     = "2 - 菜鸟工具"

    print (dictexp['one'])       # 输出键为 'one' 的值
    print (dictexp[2])           # 输出键为 2 的值
    print (dictexp)              # 输出完整的字典
    print (dictexp.keys())       # 输出所有键
    print (dictexp.values())     # 输出所有值
##### 字典内置函数
    len(dictexp)      #计算字典元素个数  
    str(dictexp)      #输出字典可打印的字符串表示  
##### 字典嵌套
例子中，键'class'的值是子字典。  
操作上实际就是多级索引。

    dictexp = {'name': 'runoob', 'age': 7, 'class': {'name': '101', 'teacher': 'teacher1'}}

    # 访问 'class' 对应子字典中的键 'name' 的值
    print(dictexp['class']['name'])
    # 连续使用get()方法安全访问，避免因为键不存在导致程序崩溃（这当然也适用于普通字典）
    print(f"get:{dictexp.get('class', {}).get('name', 'default')}")  # 输出键为 'class' 的值中的键为 'name' 的值，如果不存在则返回'default'

    # 想要遍历到子字典中的所有键值对，需要循环嵌套
    for key, value in dictexp.items():
        print(f"key:{key}", f"value:{value}")
        if type(value) == dict:
            for k, v in value.items():
                print(f"subdict key:{k}", f"subdict value:{v}")


#### 集合
**注意与字典的区别**  
在 Python 中，集合使用大括号 {} 表示，元素之间用逗号分隔。  
另外，也可以使用 set() 函数创建集合。  
注意：**创建一个空集合**必须用 set() 而不是 { }，因为 { } 是用来**创建一个空字典**。

### 语句规则

#### 多行语句
Python允许使用反斜杠`\`来实现多行语句，例如：  

    total = item_one + \
            item_two + \
            item_three
在 [], {}, 或 () 中的多行语句，不需要使用反斜杠 \

#### print() 函数
* print() 函数可以输出多个值，用逗号隔开。
* 默认输出是换行的，如果要实现不换行需要在变量末尾加上 end=""。
#### 条件语句
Python中的else if关键字为elif.  
Python中没有switch语句，在python3.10中引入了match语句。  

    match subject:
        case <pattern_1>:
            <action_1>
        case <pattern_2>:
            <action_2>
        case <pattern_3>|<pattern_4>|...|<pattern_n>: #多个条件用竖线分隔
            <action_3>
        case _: #'_'是通配符，即C语言的default
            <action_wildcard>

#### 循环语句
Python中有两种循环语句，一种是for...in循环，另一种是while循环。  
**for循环的语法格式如下：**  

    for <变量> in <序列>:
        <语句>
    else:
        <语句> # 当for循环正常结束时，执行该语句。
对于<序列>部分，可以使用range()函数生成整数序列，使用item()函数遍历字典的键值对。  
>item()函数返回一个元组，同时包含字典的键和值。

**while循环的语法格式如下：**  
注意：Python中没有do-while循环。

    while <条件>:
        <语句>
    else:
        <语句> # 当while条件为False时，执行该语句并退出循环。
**break和continue语句**  
使用break语句可以提前退出循环，并不执行else语句；  
使用continue语句可以跳过当前的迭代并继续下一轮循环。  
![break和continue](post/instruction/python/python-while.png)   
这两个语句对于for和while跳出过程一致。

#### 推导式
推导式是一种根据已有列表、字典等创建新数据序列的简洁方式，同时可以对原序列进行过滤、排序等操作。  
语法格式如下：  

    # 生成器表达式
    (<表达式> for <变量> in <序列> if <条件>)
    #eg:
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    even_numbers = (num for num in numbers if num % 2 == 0)
    print(list(even_numbers))  # 输出：[2, 4, 6, 8]//也可以用next方法或for循环
    # 列表推导式
    [<表达式> for <变量> in <序列> if <条件>]
    #eg:
    names = ['Bob','Tom','alice','Jerry','Wendy','Smith']
    new_names = [name.upper()for name in names if len(name)>3]
    print(new_names)
    # 输出结果：['ALICE', 'JERRY', 'WENDY', 'SMITH']
    
    # 字典推导式
    {<键表达式>:<值表达式> for <变量> in <序列> if <条件>}
    #eg:
    d = {'a':1,'b':2,'c':3}
    new_d = {k:v for k,v in d.items() if v>1}
    print(new_d)
    # 输出结果：{'b': 2, 'c': 3}
#### 函数
##### 默认参数
    def function_name(parameter1, parameter2=default_value):
        <表达式>
如果调用该函数时不提供 parameter2 的值，则会使用 default_value。  
**注意：**
1. 默认参数必须放在非默认参数的后面。
2. 默认参数的值在**函数定义**时计算一次，而不是在函数每次调用时计算。  

因此，更推荐使用 None 作为默认参数，在函数内进行判断之后赋值：

    def function_name(parameter1, parameter2=None):
        if parameter2 is None:
            parameter2 = default_value
        <其它表达式>

##### return和yeild
return语句用来提前结束函数  
而yield语句用于生成一个值，并在下一次迭代时返回。  
**注意：**
1. return 语句只能在函数内部使用，而 yield 语句可以在函数内部或外部使用。
2. yield 语句只能用于**生成器**函数，而 return 语句只能用于非生成器函数。

##### lambda（匿名函数）  
lambda 函数通常用于编写简单的、单行的函数，通常在需要函数作为参数传递的情况下使用。  

语法格式：  

    lambda arguments: expression
示例：  

    # 示例1
    x = lambda a, b : a * b
    print(x(5, 6))  # 输出：30

    # 示例2
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    print(even_numbers)  # 输出：[2, 4, 6, 8]


### 功能模块
#### 迭代器(iterator)
迭代器是一个记住**遍历的位置**的对象，不像列表把所有元素一次性加载到内存，而是每次需要数据的时候才实时计算下一个数据，节省内存。  
迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器**只能往前不会后退**。

**基本生成：**
Python的迭代器有两个基本的方法：iter() 和 next()。  

示例：  

    list=[1,2,3,4]
    it = iter(list)    # 创建迭代器对象
    # 使用next()函数获取迭代器中下一个元素
    for x in range(4):
        print (next(it))

    # 使用for循环获取迭代器中下一个元素
    for x in it:
        print (x, end=" ")

    # 直接读取列表
    for x in list:
        print (x, end=" ")
    
    # 三种方法结果都是1 2 3 4
**客制化生成：**
如果要创建自己的迭代器，需要定义一个类，并实现私有的iter()和next()方法。

示例：  

    class MyNumbers:
    def __iter__(self): # 类私有方法，返回一个特殊的迭代器对象
        self.a = 1
        return self
    
    def __next__(self):
        if self.a <= 20: # 可以限定迭代次数
            x = self.a
            self.a += 1 # 可以客制化迭代生成值
            return x
        else:
            raise StopIteration # StopIteration 异常用于标识迭代的完成
    
    myclass = MyNumbers()
    myiter = iter(myclass)
    
    for x in myiter:
    print(x)
#### 生成器(generator)[更简便更常用，当正常函数写]
生成器本质上是一种特殊的迭代器(**使用yeild语句**)，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。  

* 简单地讲，yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，调用 function(var) 不会执行该函数，而是返回一个 iterable 对象。
* 当在生成器函数中使用 yield 语句时，函数的执行将会暂停。yield 对应的值在函数被调用时不会立刻返回，而是调用next方法时才返回。
* 然后，每次调用生成器的 next() 方法或使用 for 循环进行迭代时，函数会从**上次暂停的地方**继续执行，直到再次遇到 yield 语句。

        def countdown(n):
            while n > 0:
                yield n
                n -= 1
        
        # 创建生成器对象
        generator = countdown(5)

        # 使用 for 循环迭代生成器/也可以使用next()方法
        for value in generator:
            print(value)  # 输出:5 4 3 2 1
#### 装饰器

### 面向对象
#### self
self代表类的实例，而非类。  
self 是一个惯用的名称，用于表示类的实例（对象）自身。它是一个指向实例的引用，使得类的方法能够访问和操作实例的属性。

#### 类的继承
Python支持多继承，一个类可以从多个父类继承方法和属性。  
#### 类的方法种类
类的专有方法：

    __init__(self, args)：类的构造函数，在对象创建时调用。
![class_method](post/instruction/python/classmethod.png)
类的私有方法：

    def __private_method(self, args)：
        <表达式> # 私有方法，只能在类的内部调用。
类的公有方法：

    def public_method(self, args)：
        <表达式> # 公有方法，可以在类的外部调用。
类的私有属性： 

    __private_attribute = value #私有属性，只能在类的内部访问。
类的公有属性：

    public_attribute = value #公有属性，可以在类的外部访问。


## 正则表达式

## Python Requests
python的requests模块可以用来发送HTTP请求，它可以自动处理cookie、认证、重定向、超时等问题。

**get()方法**

    requests.get(url, params={key: value}, \
    headers={key: value}, cookies={key: value}, \
    auth=(), timeout=None, allow_redirects=True, \
    proxies=None, hooks=None)

**post()方法**

    requests.post(url, data={key: value}, \
    json={key: value}, args)
data参数用于发送表单数据，json参数用于发送JSON数据。args为其他参数，比如 cookies、headers、verify等。

![请求内容](post/instruction/python/request.png)  
![请求方法](post/instruction/python/request_method.png)  
示例：

    import requests

    # GET请求
    response = requests.get('https://www.google.com')
    print(response.status_code)  # 输出状态码
    print(response.text)  # 输出unicode响应内容
    print(response.content)  # 输出响应内容
    print(response.json())  # 输出JSON格式响应内容

    # POST请求
    response = requests.post('https://httpbin.org/post', data={'key': 'value'})
    print(response.text)  # 输出响应内容

    # 上传文件
    files = {'file': open('report.xls', 'rb')}
    response = requests.post('https://httpbin.org/post', files=files)
    print(response.text)  # 输出响应内容

    # 自定义请求头
    headers = {'User-Agent': 'Mozilla/5.0'}  # 设置请求头
    params = {'key1': 'value1', 'key2': 'value2'}  # 设置查询参数
    data = {'username': 'example', 'password': '123456'}  # 设置请求体
    response = requests.post('https://www.runoob.com', headers=headers, params=params, data=data)  # 输出响应内容

    # 超时设置
    response = requests.get('https://www.google.com', timeout=5)
    print(response.text)  # 输出响应内容

### 进阶用法

#### 身份认证
身份认证是指客户端提供用户名和密码给服务器，服务器验证用户名和密码是否正确，并确认客户端的身份。

requests模块提供了四种身份认证方法：
1. Basic Auth：基本认证，用户名和密码以明文形式发送，容易被窃听。
2. Digest Auth：摘要认证，用户名和密码以加密形式发送，安全性高。
3. OAuth 1.0a：OAuth 1.0a 认证，用于第三方应用授权。
4. OAuth 2.0：OAuth 2.0 认证，用于第三方应用授权。


#### 重定向
requests模块默认会自动处理重定向，如果服务器返回的响应状态码为3xx，客户端自动请求新的URL，进行重定向。

#### 超时设置
超时设置是指如果服务器在指定时间内没有响应，则请求超时。
requests模块默认超时时间为10秒，可以通过timeout参数设置超时时间。
## Python 网络编程
socket使主机间或者一台计算机上的进程间可以通讯。

用一个简单的示例展示服务器与客户端的通信过程中使用的套接字函数。  
#### 示例一：
1. 服务器端： 
server.py

        import socket
        import time
        COD = 'utf-8'
        HOST = socket.gethostname()# 获取本地主机ip
        PORT = 21566 # 软件端口号
        BUFSIZ = 1024
        ADDR = (HOST, PORT)
        SIZE = 10
        tcpS = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 创建socket对象
        tcpS.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) # 加入socket配置，重用ip和端口
        tcpS.bind(ADDR) # 绑定ip和端口号到套接字
        tcpS.listen(SIZE)  # 监听链接，设置最大链接数
        while True:
            print("服务器启动，监听客户端链接")
            conn, addr = tcpS.accept()# 建立客户端链接
            print("链接的客户端", addr)
            while True:
                try:
                    data = conn.recv(BUFSIZ) # 读取已链接客户的发送的消息
                except Exception:
                    print("断开的客户端", addr)
                    break
                print("客户端发送的内容:",data.decode(COD))
                if not data:
                    break
                msg = time.strftime("%Y-%m-%d %X") # 获取结构化事件戳
                msg1 = '已接收到[%s]的内容:%s' % (msg, data.decode(COD))
                conn.send(msg1.encode(COD)) # 发送消息给已链接客户端
            conn.close() # 关闭客户端链接
        tcpS.closel()
2. 客户端：
client.py  

        import socket
        import time
        HOST = socket.gethostname()# 获取本地主机ip
        PORT = 21566 # 服务端端口号
        BUFSIZ = 1024
        ADDR = (HOST, PORT)
        tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 创建socket对象
        tcpCliSock.connect(ADDR) # 连接服务器
        while True:
            data = input('>>').strip()
            if not data:
                break
            tcpCliSock.send(data.encode('utf-8')) # 发送消息
            data = tcpCliSock.recv(BUFSIZ) # 读取消息
            if not data:
                break
            print(data.decode('utf-8'))
        tcpCliSock.close() # 关闭客户端
#### 示例二：
抄送、密送聊天室程序：  
1. 服务器端： 
server.py

        import socket
        import threading

        # 客户端地址 名称
        addr_name = {}
        # 所有客户端
        all_clients = []
        # 名称 客户端
        name_client = {}

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = socket.gethostname()
        port = 9999
        server.bind((host, port))
        server.listen(5)
        lock = threading.Lock()
        print("开启聊天室")

        def handle_sock(sock, addr):
            while True:
                try:
                    data = sock.recv(1024)
                    msg = data.decode("utf-8")
                    print("send msg")
                    from_name = addr_name[str(addr)]
                    if msg.startswith('@'):
                        index = msg.index(' ')
                        # 私聊人
                        to_name = msg[1:index]
                        # 接收者客户端
                        to_sock = name_client[to_name]
                        # 发送的消息
                        to_msg = msg[index:]
                        send_one(to_sock, addr, from_name + ":" + to_msg)
                    else:
                        # 群发消息
                        send_all(all_clients, addr, from_name + ":" + msg)
                except ConnectionResetError:
                    exit_name = addr_name[str(addr)]
                    exit_client = name_client[exit_name]
                    all_clients.remove(exit_client)
                    msg = exit_name + " 退出了群聊"
                    send_all(all_clients, addr, msg)
                    break

        def send_all(socks, addr, msg):
            for sock in socks:
                sock.send(msg.encode("utf-8"))

        def send_one(sock, addr, msg):
            sock.send(msg.encode("utf-8"))

        while True:
            sock, addr = server.accept()
            name = sock.recv(1024).decode("utf-8")
            addr_name[str(addr)] = name
            name_client[name] = sock
            all_clients.append(sock)
            hello = name + "加入了聊天室"
            send_all(all_clients, addr, hello)
            client_thread = threading.Thread(target=handle_sock, args=(sock, addr))
            client_thread.start()
2. 客户端：
client.py  

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

#### 网络编程常用模块：
![Python网络编程常用模块](post/instruction/python/socket_module.png)  
[Python官方 Socket Library and Modules](<https://docs.python.org/3.0/library/socket.html>)
## Python JSON
待补充
## Python MySQL
待补充