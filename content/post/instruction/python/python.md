+++
title = 'Python'
date = 2024-08-13T09:48:39+08:00
draft = false
categories = ['指令语法']
+++

# Python

## 基础语法

*args 允许你将任意数量的非关键字参数传递给一个函数。这些参数会以一个元组的形式传递给函数。

**kwargs 允许你将任意数量的关键字参数传递给一个函数。这些参数会以一个字典的形式传递给函数。
```python
    def my_function(*args, **kwargs):
        print("args:", args)
        print("kwargs:", kwargs)

    my_function(1, 2, 3, name="Alice", age=30)
```
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
```python
    str='123456789'

    print(str[0:-1])           # 输出第一个到倒数第二个的所有字符
    print(str[2:])             # 输出从第三个开始后的所有字符
    print(str[1:5:2])          # 输出从第二个开始到第五个且每隔一个的字符（步长为2）
    print(str * 2)             # 输出字符串两次
    print ("我叫%s今年 %d 岁!" % ('小明', 10))
                            # 格式化输出字符串
```
##### f-string
    >>>f'{1+2}'         # 使用表达式
    '3'

    w = {'name': 'Runoob', 'url': 'www.runoob.com'}
    >>>f'{w["name"]}: {w["url"]}'
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
```python
    list.append(obj)    # 在列表末尾添加新的对象
    list.extend(seq)     # 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
    list.remove(obj)     # 删除列表中某个值的第一个匹配项
    del list[index]      # 删除指定索引的元素
    list.pop(n)        # 从列表中删除某个元素（默认最后一个元素），并且返回该元素的值
    list.reverse()      # 反转列表
```


#### 字典
字典是一种映射类型，字典用 { } 标识，它是一个无序的 键(key) : 值(value) 的集合，集合元素间用逗号隔开。  

* 键(key)必须使用不可变类型，比如字符串、数字或元组。  
    键应该是唯一的，重复的键会覆盖前面的键。  

* 值(value)可以是任意类型，包括列表甚至另一个字典？。<br>  

##### 访问字典的值  
```python
    dictexp = {}
    dictexp['one'] = "1 - 菜鸟教程"
    dictexp[2]     = "2 - 菜鸟工具"

    print (dictexp['one'])       # 输出键为 'one' 的值
    print (dictexp[2])           # 输出键为 2 的值
    print (dictexp)              # 输出完整的字典
    print (dictexp.keys())       # 输出所有键
    print (dictexp.values())     # 输出所有值
```
##### 字典内置函数
```python
    len(dictexp)      #计算字典元素个数  
    str(dictexp)      #输出字典可打印的字符串表示  
```
##### 字典嵌套
例子中，键'class'的值是子字典。  
操作上实际就是多级索引。
```python
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
```

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
```python
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
```
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
```python
    # 示例1
    x = lambda a, b : a * b
    print(x(5, 6))  # 输出：30

    # 示例2
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    print(even_numbers)  # 输出：[2, 4, 6, 8]
```

### 功能模块
#### 迭代器(iterator)
迭代器是一个记住**遍历的位置**的对象，不像列表把所有元素一次性加载到内存，而是每次需要数据的时候才实时计算下一个数据，节省内存。  
迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器**只能往前不会后退**。

**基本生成：**
Python的迭代器有两个基本的方法：iter() 和 next()。  

示例：  
```python
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
```
**客制化生成：**
如果要创建自己的迭代器，需要定义一个类，并实现私有的iter()和next()方法。

示例：  
```python
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
```
#### 生成器(generator)[更简便更常用，当正常函数写]
生成器本质上是一种特殊的迭代器(**使用yeild语句**)，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。  

* 简单地讲，yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，调用 function(var) 不会执行该函数，而是返回一个 iterable 对象。
* 当在生成器函数中使用 yield 语句时，函数的执行将会暂停。yield 对应的值在函数被调用时不会立刻返回，而是调用next方法时才返回。
* 然后，每次调用生成器的 next() 方法或使用 for 循环进行迭代时，函数会从**上次暂停的地方**继续执行，直到再次遇到 yield 语句。
```python
    def countdown(n):
        while n > 0:
            yield n
            n -= 1
    
    # 创建生成器对象
    generator = countdown(5)

    # 使用 for 循环迭代生成器/也可以使用next()方法
    for value in generator:
        print(value)  # 输出:5 4 3 2 1
```
#### 装饰器
装饰器本质上是一个函数，它接收一个函数作为参数并返回一个新的函数。这个新函数是对原函数的一种包装或增强，可以在不改变原函数代码的前提下，增加额外的功能。

Python 还提供了一些内置的装饰器，比如 @staticmethod 和 @classmethod，用于定义静态方法和类方法。

##### 工作流程
装饰器的工作流程可以分为以下几个步骤：

1. 定义装饰器：首先定义一个装饰器函数，该函数**接收一个函数作为参数**。
2. 定义包装函数：在装饰器函数内部，定义一个包装函数（wrapper），这个包装函数会调用原函数，并可以在调用前后添加额外的逻辑。
3. 返回包装函数：装饰器函数返回这个包装函数。
4. 使用@语法：在需要被装饰的函数定义前使用@符号加上装饰器名称，这样使用被装饰函数时，Python解释器会自动**将这个函数作为参数**传递给装饰器，并将**包装函数赋值给原函数名**。

因此调用被装饰函数时，实际上是调用了包装函数。

##### 装饰器语法格式：
```python
    # 这是装饰器函数，参数 func 传入被装饰的函数
    def logger(func):
        def wrapper(*args, **kwargs): # 传入被装饰函数func的各参数
            print('我准备开始执行：{} 函数了:'.format(func.__name__))
            
            func(*args, **kwargs)# 真正执行被装饰函数的是这行。

            print('我执行完了。')
        return wrapper # 注意需要返回包装函数

    @logger
    def add(x, y): # 被装饰的函数，实际上是logger的参数func
        print('{} + {} = {}'.format(x, y, x+y))

    add(2, 3)
```

##### 带参数的装饰器语法格式：需要两层嵌套
因为装饰器函数的参数只有func(被装饰函数)，所以无法直接传入参数，需要额外嵌套一层。

示例一：
```python
    def repeat(n): # 传入装饰器参数
        def decorator(func):
            def wrapper(*args, **kwargs): # 传入被装饰函数func的各参数
                for i in range(n):
                    print('No.',i+1,end=' ')
                    func(*args, **kwargs)
            return wrapper
        return decorator

    @repeat(3) # 装饰器名称应为最外层
    def greet(name):
        print(f"Hello, {name}!")

    greet("Alice")
```
示例二：
```python
    def say_hello(country): # 传入装饰器参数
        def deco(func):
            def wrapper(*args, **kwargs): # 传入被装饰函数func的各参数
                if country == "china":
                    print("你好!")
                elif country == "america":
                    print('hello.')
                else:
                    return

                func(*args, **kwargs)# 真正执行被装饰函数的一行

            return wrapper # 注意需要返回包装函数
        return deco # 注意需要返回次层装饰器函数

    # 小明，中国人
    @say_hello("china")
    def xiaoming():
        pass
    # jack，美国人
    @say_hello("america")
    def jack():
        pass

    xiaoming()
    print("------------")
    jack()
```

### 异常处理
Python的异常处理机制非常灵活，可以处理多种类型的异常。

#### 基本语法

    try:
        <可能发生异常的代码>
    except <异常类型1>:
        <异常处理1>
    except (<异常类型2>, <异常类型3>,...):
        <异常处理2>
    # 可以用except:作为通配符处理所有异常
    else: # 必须放在所有except语句之后
        <没有异常发生时执行>
    finally:
        <无论异常是否发生都执行> # 若try/except/else产生的异常未被处理，将先执行finally语句再被抛出
#### raise语句
raise语句用于手动抛出异常，并通知调用者发生了什么异常。  

语法格式：raise唯一的一个参数Exception，指定了要被抛出的异常。若不提供参数，则重新抛出当前异常而不进行处理。

    raise [Exception [, args [, traceback]]]
示例：  
```python
    x=10
    try:
        if x>5:
            raise NameError('HiThere')  # 抛出一个NameError异常。
    except NameError:
        print('An exception flew by!')
        raise # 不提供Exception参数，则重新抛出当前异常，程序中断于此。若删去该行，仍会执行下面语句输出y的值。
    y=6
    print(y)
```
#### 用户自定义异常
用户自定义异常类需要继承自Exception类。

示例：  
```python
    class MyError(Exception):
        def __init__(self, value): # 覆盖类Exception的__init__方法
            self.value = value
        def __str__(self):
            return repr(self.value)
   
    try:
        raise MyError(2*2)
    except MyError as e:
        print('My exception occurred, value:', e.value)
```
#### with语句进行预定义的清理
一些对象定义了标准的清理行为，无论系统是否成功的使用了它，一旦不需要它了，那么这个标准的清理行为就会执行。
关键词 with 语句就可以保证诸如文件之类的对象在使用完之后一定会正确的执行它的清理方法:
```python
    with open("myfile.txt") as f:
        for line in f:
            print(line, end="")
```
以上这段代码执行完毕后，就算在处理过程中出问题了，文件 f 总是会关闭。        

补充：with 语句的语法格式如下：

    with expression [as variable]:
        with-block

expression 是一个**上下文管理器对象**，它定义了该对象的上下文，with-block 是在该上下文中要执行的语句。

当执行到 with 语句时，会首先执行 expression，该表达式应该返回一个上下文管理器对象。然后，with 语句将该上下文管理器对象压入一个栈，并将该对象的变量（如果有）赋值给 variable（如果有）。

当执行完 with-block 后，会弹出该上下文管理器对象，并调用其清理方法。

### 面向对象
#### self参数
self表示类的实例（对象）自身，通过self参数将类的实例传入类的方法中，使得类的方法能够访问和操作**所创建的各个实例**的属性。
##### 示例一：
```python
    class MyClass:
        def __init__(self, value):
            self.value = value

        def display_value(self):
            print(self.value)

    # 创建一个类的实例
    obj = MyClass(42) # self传入obj实例
    obj2 = MyClass(25) # self传入obj2实例
    # 调用实例的方法
    obj.display_value() # 输出 42
    obj2.display_value() # 输出 25
```
##### 示例二：查看self实例位置
```python
    class Desc:
        def __get__(self, ins, cls):
            print(self, ins, cls) # self参数为Desc类的实例，ins参数为Test类的实例，cls参数为Test类本身。
    class Test:
        x = Desc()
        def prt(self):
            print('self in Test: %s' % self)

    t0 = Test()
    t1 = Test()

    t0.prt()
    t1.prt()
    t0.x
    t1.x
    # 输出：
    # self in Test: <__main__.Test object at 0x000001>
    # self in Test: <__main__.Test object at 0x000002>
    # <__main__.Desc object at 0x000003> <__main__.Test object at 0x000001> <class '__main__.Test'>
    # <__main__.Desc object at 0x000003> <__main__.Test object at 0x000002> <class '__main__.Test'>
    # 说明：
    # 1. 实例化Test类时，t0和t1传入的self参数分别对应Test类的两个实例。
    # 2. Test类中的x = Desc()语句与self参数无关，因此不论是t0.x还是t1.x或Test.x，x都指向同一个Desc类的实例，即x是类Test的类属性。
```

#### 类的继承

    class DerivedClassName(Base1, Base2, Base3):
        <statement-1>
        .
        .
        .
        <statement-N>
子类会继承父类的属性和方法，并可以对其进行覆写，也可以添加新的属性和方法。   

Python支持多继承(并行继承、多重继承都可以)，一个类可以从多个父类继承方法和属性。
当调用父类方法，且几个父类中有相同的方法名时，python将按从左到右的顺序调用父类中的方法。
父类可以从别的文件import。

##### 示例：
```python
    class people:
        # 定义构造方法
        def __init__(self, n, a):
            self.name = n
            self.age = a
        def speak(self):
            print("父类%s 说: 我 %d 岁。" % (self.name, self.age))

    # 单继承示例
    class student(people):
        grade = ''
        def __init__(self, n, a, g):
            # 调用父类的构函
            people.__init__(self, n, a)
            self.grade = g
        # 覆写父类的方法
        def speak(self):
            print("子类%s 说: 我 %d 岁了，我在读 %d 年级" % (self.name, self.age, self.grade))

    s0 = people('s0',18)
    s1 = student('s1',10,3)
    s0.speak()
    s1.speak()
    # 输出：父类s0 说: 我 18 岁。 子类s1 说: 我 10 岁了，我在读 3 年级
```
#### 类的方法种类
静态方法: 用 @staticmethod 装饰的不带 self 参数的方法叫做静态方法，类的静态方法可以没有参数，可以直接使用类名调用。

普通方法: 默认有个self参数，且只能被对象调用。

类方法: 默认有个 cls 参数，可以被类和对象调用，需要加上 @classmethod 装饰器。定义和调用时不需要传入实例对象。

    @staticmethod
    def func0():
        print('静态方法')

    @classmethod
    def func1(cls):
        print('类方法')
        print(cls)
        
类的专有方法：

    __init__(self, args)：类的构造函数，在对象创建时调用。
![class_method](post/instruction/python/classmethod.png)
类的私有方法：

    def __private_method(self, args)：
        <表达式> # 私有方法，只能在类的内部调用。
类的公有方法：

    def public_method(self, args)：
        <表达式> # 公有方法，可以在类的外部调用。
类的私有属性(封装)： 

    __private_attribute = value #私有属性，只能在类的内部访问。
这确保了外部代码不能随意修改类的内部状态。  
若需要从外部读取或修改私有属性时，需要通过公有方法来实现。

类的公有属性：

    public_attribute = value #公有属性，可以在类的外部访问。


## 正则表达式
### 基本函数
1. re.search(pattern, string, flags=0)：在字符串中搜索匹配项。  
2. re.match(pattern, string, flags=0)：从字符串的开头匹配，只匹配开头。  
3. re.findall(pattern, string, flags=0)：找到所有匹配的子串，并以列表形式返回。  
    或pattern.findall(string[, pos[, endpos]])，与findall()相同，可指定搜索区间。  
4. re.sub(pattern, repl, string, count=0, flags=0)：替换匹配到的子串。  
5. re.split(pattern, string[, maxsplit=0, flags=0])：根据正则表达式分割字符串。  
6. re.compile(pattern[, flags])：编译正则表达式，以便重复使用。 

参数解释：

- pattern：正则表达式。
- string：要匹配的字符串。
- flags：可选参数，表示匹配模式，比如忽略大小写，多行匹配等。
- repl：要替换的字符串，repl参数也可以是一个函数，用于 re.sub() 方法。
- count：用于 re.sub() 方法，表示最多替换的次数。
- maxsplit：用于 re.split() 方法，表示最多分割次数。

示例1：
```python
    import re
    pattern = re.compile(r'([a-z]+) ([a-z]+) ([a-z]+)', re.I)   # re.I 表示忽略大小写
    # 匹配三个单词的组合
    m = pattern.search('Hello World Wide Web')
    print(m.groups())
```
示例2：re.sub()方法，repl参数是一个函数
```python
    import re
    
    # 将匹配的数字乘以 2
    def double(matched):
        valueout = int(matched.group('value0'))
        return str(valueout * 2)
    
    s = 'A23G4HFD567'
    print(re.sub('(?P<value0>\d+)', double, s))  
    # (?P<name>...)：这是命名捕获组的语法。name 是指定的组名，... 是要捕获的正则表达式模式。
    # 此处即捕获由数字组成的字符串组value0，传递至double()函数
```
### 匹配规则
#### 字符和分组
    .：匹配除换行符外任意字符。
    []：匹配括号中的任意字符。如[abc]：匹配a、b、c中的任意一个字符。
    [^]：匹配不在括号中的任意字符。如[^abc]：匹配除 a、b、c 以外的任意字符。
    [a-z]：匹配指定范围内的字符。例如，[a-z] 匹配任意小写字母。
    ()：用于分组。例如，(abc)+ 匹配一个或多个连续的 abc。
    |：表示或。例如，abc|def 匹配 abc 或 def。
#### 量词
    *：匹配前面的元素零次或多次。例如，a* 匹配 ""、a、aa、aaa 等。
    +：匹配前面的元素一次或多次。例如，a+ 匹配 a、aa、aaa 等。
    ?：匹配前面的元素零次或一次。例如，a? 匹配 "" 或 a。
    {n}：匹配前面的元素恰好 n 次。例如，a{3} 匹配 aaa。
    {n,}：匹配前面的元素至少 n 次。例如，a{3,} 匹配 aaa、aaaa 等。
    {n,m}：匹配前面的元素至少 n 次，但不超过 m 次。例如，a{2,3} 匹配 aa 或 aaa。
#### 位置匹配
    ^：匹配字符串的开始。例如，^abc 匹配以 abc 开头的字符串。
    $：匹配字符串的结束。例如，abc$ 匹配以 abc 结尾的字符串。
    \b：匹配单词边界。例如，\bcat\b 匹配完整的单词 cat。
    \B：匹配非单词边界。例如，\Bcat\B 匹配字符串 scatters 中的 cat。
#### 特殊字符
    \：转义字符，用于匹配特殊字符。如'\. '匹配点号，而不是匹配任意字符。
    \d：匹配任意数字。相当于 [0-9]
    \D：匹配任意非数字。相当于 [^0-9]
    \w：匹配任意字母、数字、下划线。相当于 [a-zA-Z0-9_]
    \W：匹配任意非字母、数字、下划线。相当于 [^a-zA-Z0-9_]
    \s：匹配任意空白字符，包括空格、制表符、换行符。相当于[ \t\n\r\f\v]。
    \S：匹配任意非空白字符。相当于 [^ \t\n\r\f\v]。
    \n：匹配换行符。
    \t：匹配制表符。
    \r：匹配回车符。
#### 零宽断言
    (?=...)：正向肯定断言，要求后面必须能匹配某个模式。例如，a(?=b) 匹配 a，但要求 a 后面必须有 b。
    (?!...)：正向否定断言，要求后面不能匹配某个模式。例如，a(?!b) 匹配 a，但要求 a 后面不能有 b。
    (?<=...)：反向肯定断言，要求前面必须能匹配某个模式。例如，(?<=b)a 匹配 a，但要求 a 前面必须有 b。
    (?<!...)：反向否定断言，要求前面不能匹配某个模式。例如，(?<!b)a 匹配 a，但要求 a 前面不能有 b。

#### 常见表达式
匹配电子邮件：

    [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+
    # [a-zA-Z0-9_.+-]+：匹配至少一个字母、数字、下划线、点、加号或减号
    # [a-zA-Z0-9-]+：匹配@后面至少一个字母、数字或减号
    # [a-zA-Z0-9-.]+：匹配.后面至少一个字母、数字、减号或点
    
匹配手机号码：

    (\+?\d{1,3})?[-.\s]?(\d{3})[-.\s]?(\d{4})[-.\s]?(\d{4})
    # (\+?\d{1,3})?：加号可选，后跟一到三位数字。该非捕获组可选
    # [-.\s]?：连字符、句点、空格可选
    # (\d{3})：三位数字


匹配IP地址：

    ((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
    # 匹配((0-255).)三次，再匹配(0-255)一次

匹配URL：

    https?://[a-zA-Z0-9./?=&_-]+
    # s?：匹配s可选，即http或https
    # (?:www\.)是非捕获组，只用于分组而不捕获，配合其后的问号，(?:www\.)?表示www.可选。
    # [a-zA-Z0-9./?=&_-]+匹配方括号中任意字符一次或多次
示例：
```python
    import re

    pattern_mail=re.compile('[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    pattern_num=re.compile('(\+?\d{1,3})?[-.\s]?(\d{3})[-.\s]?(\d{4})[-.\s]?(\d{4})')
    pattern_ip=re.compile('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)')
    pattern_url= re.compile('https?://(?:www\.)?[a-zA-Z0-9./?=&_-]+')
    
    s = 'self表示类的实例（对象）自身，通过self参数ychuang317@163.com将类的实例传入+86-13706811848类的方法中，使得类的方法能255.255.255.0够访问和操作http://hych0317.github.io/hugoweb_auto所创建的各个实例的属性。'
    print(re.search(pattern_num, s).group(0))
```
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
```python
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
```
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
```python
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
```
2. 客户端：
client.py  
```python
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
```
#### 示例二：
抄送、密送聊天室程序：  
1. 服务器端： 
server.py
```python
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
```
2. 客户端：
client.py  
```python
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
```
#### 网络编程常用模块：
![Python网络编程常用模块](post/instruction/python/socket_module.png)  
[Python官方 Socket Library and Modules](<https://docs.python.org/3.0/library/socket.html>)

## Python JSON
待补充
## Python MySQL
待补充