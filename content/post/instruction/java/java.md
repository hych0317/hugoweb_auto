+++
title = 'Java'
date = 2025-09-13T09:48:39+08:00
draft = false
categories = ['指令语法']
+++



## 语法基础
### 权限修饰符
    private：仅在同一个类中可见。
    default（无修饰符）：同一个包中可见。
    protected：同一个包 和 不同包的子孙类 中可见。
    public：所有地方都可见。

### 变量类型
实例（成员）变量 (instance variable)：属于某个对象（实例）的变量。
    可以由public和private修饰。见封装部分。

静态（类）变量 (static variable)：由static修饰，属于类本身的变量，不依赖某个具体对象。
```java
public class VarTest{
    private int instVar;
    private static int staVar;
    ...
}

VarTest v1 = new VarTest();
VarTest v2 = new VarTest();

v1.instanceVar = 10;
v2.instanceVar = 20;//不同
// v1.staVar与v2.staVar相同
```
静态变量推荐使用类名访问VarTest.staVar

局部变量：方法内部的变量，**必须初始化**

### 实例、静态方法
静态方法：常用于main方法、工具类中的方法
由static修饰，是属于类的方法，用于只需完成功能，而**不需要访问具体对象的数据**时。  
可以用类名(推荐)和对象名访问  

    为什么工具类的方法要用静态方法？
    1. 可以直接用工具类类名访问，方便；
    2. **不用创建对象，节省内存**。建议私有化工具类的构造器，使得无法创建对象。

实例方法：由对象名访问

### 方法重载/重写
方法重载：在同一个类中，方法名相同，但参数列表不同（参数个数、类型或顺序不同），返回值类型可以不同。

方法重写：子类继承父类后，对父类中已有方法进行重新实现，**方法名、参数列表必须完全相同**，返回值类型相同或范围更小，且修饰符不能比父类严格。  
父类的私有方法、静态方法不能被重写。  
使用 **@Override** 校验注解检查重写的方法是否符合规范。

重载（Overload）：编译时多态（方法名相同，参数不同）。
重写（Override）：运行时多态（子类重写父类的方法）。 


### 数据类型
整型的默认类型是int
```java
sum(100,200);//会报错，因为字面量100是默认类型int，需要(byte)100

public static int sum(byte a, byte b)
{
    return a+b;//返回类型为int
}
```

浮点数的默认类型是double

### 表达式类型转换
最终结果由表达式的最高类型决定；
byte、short、char在表达式中运算时直接提升为int参与运算。

### 运算符
+号在字符串运算中起连接作用，"abc"+5得"abc5","abc"+5+'a'得"abc5a", 'a'+5+"abc"得"102abc"

b = a++;//先赋值再自加
b = ++a;//先自加再赋值

扩展赋值运算符：+=等五种
a += b ：a = **(a的类型)** a + b

短路与、或(&&、||)：若左边为false/true，则不执行右边(如++b>1不会导致自加)提前返回结果。

### 流程控制语句
#### switch
表达式支持数据类型：byte、short、char、int、String、枚举类型；**不支持**double、float、long
(因为储存小数靠二进制拟合，0.3实际上是0.300..004之类，导致无法正确匹配)  

case的值必须是确定的字面量，且不能重复
break关键字是可选的，如果没有则**执行下一个case**(穿透性，当几种case处理代码一致时可以复用)；如果有，则跳出switch语句。
default也是可选的。

## 面向对象
三大特征：封装、继承、多态
### 构造器
创建类的对象时，会自动调用构造器（不指定即是默认的无参）。  
是一种特殊的**方法**(因此可以重载)，无返回值类型，且**名称必须和类名相同**。  
类中默认自带无参构造器，但一旦定义了有参构造器，就不能使用默认无参构造器（需要自行手动构建的无参）。
```java
//对于Student类创建对象s
    Student   s   = new   Student()
//    类名  对象名          方法名
```

### this关键字
this是方法中的一个变量，用于**指代调用方法的对象**。  
解决变量名称冲突问题：如方法的局部变量名称和类成员变量名称冲突，用this访问类的变量可以避免冲突，使得可以使用相同的名称。

静态方法不可能出现this关键字。

### 封装
设计要求：合理隐藏、合理暴露
#### 隐藏
使用private关键字修饰成员变量，使得只能在本类中被直接访问。

#### 暴露
使用public关键字修饰方法，通过方法操作成员变量。  
在方法中可以添加限制，限定成员变量的范围等。

好处：如果需要修改变量类型，只需修改类中对应的方法，不用修改每一个调用的地方。

#### 实体类
成员变量全部私有，并提供公开的getter、setter方法；需要提供无参构造器（有参可选）。
只用于保存事物的数据而不进行处理。

### 继承
public class B extends A{}//B是子类，A是父类

子类能继承父类的非私有成员（变量、方法）  

子类的对象由父类和子类共同构建：  
父类的private成员也在子类对象里，只是子类代码不能直接访问，要靠父类提供的public/protected方法间接访问。

提高代码复用性：可以使用父类的public属性和方法（父类方法可以重写），也可以有自己新的属性和方法满足拓展。

#### 继承的特点
1.java不支持多继承（因为多个父类的方法可能冲突），但可以使用多层继承
2.java的祖宗类：object
3.子类访问成员遵循就近原则
```java
class Fu {
    String name = "父类的name";
}

class Zi extends Fu {
    String name = "子类的name";

    public void show() {
        String localName = "方法的name"; 
        // 就近原则：先找局部变量
        System.out.println(localName);       // 方法的name
        // this 访问当前类成员变量
        System.out.println(this.name);       // 子类的name
        // super 访问父类成员变量
        System.out.println(super.name);      // 父类的name
    }
}
```

#### 子类构造器
子类的构造器必须先调用父类的构造器，再调用自己的。  
1.默认调用父类的无参super()，可以用super(...)调用父类的有参。
2.可以使用this(...)调用子类的构造器

```java 
private static final String DEFAULT_SCHOOL = "school";

    // 构造器1：没有传 schoolName 时，使用默认值
    public Student(final String name, final char sex, final int age) {
        this(name, sex, age, DEFAULT_SCHOOL);
        //注意 this(...)不能与super()同时出现，因为两者都要求出现在构造器的第一行
    }

    // 构造器2：完整参数
    public Student(final String name, final char sex, final int age, final String schoolName) {
        this.name = name;
        this.sex = sex;
        this.age = age;
        this.schoolName = schoolName;
    }
```

#### final关键字
可以修饰类、方法、变量。
    修饰类：称为最终类，不能再被继承；
    修饰方法：称为最终方法，不能被重写；
    修饰变量：该变量仅在初始化的时候能被赋值。


### 多态
继承/实现情况下同一个行为可以具有不同的表现形式，表现为对象多态、行为多态

多态的前提条件有三个：
    子类继承父类
    子类重写父类的方法
    父类引用指向子类的对象

java的成员变量没有多态一说。  
多态下不能调用子类的独有行为，只能调用重写方法。

```java
class Animal {
    String name = "Animal"; // 父类成员变量

    public void sound() {
        System.out.println("Animal makes a sound");
    }
}

class Dog extends Animal {
    String name = "Dog"; // 子类成员变量

    @Override
    public void sound() {
        System.out.println("Dog barks");
    }
    // 子类独有方法
    public void guardHouse() {
        System.out.println("Dog is guarding the house");
    }
}

public class PolymorphismDemo {
    public static void main(String[] args) {
        Animal a = new Dog();// 父类引用指向子类对象 -> 多态

        // 1. 成员变量没有多态（编译看左边声明类型）
        System.out.println(a.name); // 输出 Animal，而不是 Dog
        // 2. 成员方法有多态（运行时看右边动态绑定）
        a.sound(); // 输出 Dog barks
        // 3. 无法直接调用子类独有方法（编译报错）
        // a.guardHouse();
        // 如果要调用子类独有方法，需要向下强制转换
        if (a instanceof Dog) {
            ((Dog) a).guardHouse();
        }
    }
}
```

