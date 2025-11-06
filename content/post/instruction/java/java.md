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
1.（实例/成员）变量 (instance variable)：属于某个对象（实例）的变量。
    可以由public和private修饰。见封装部分。

2.（静态/类）变量 (static variable)：由**static**修饰，属于类本身的变量，不依赖某个具体对象。
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

3.局部变量：**方法内部**的变量，**必须初始化**

4.常量：由**final**修饰

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
- 整型的默认类型是int
```java
sum(100,200);//会报错，因为字面量100是默认类型int，需要(byte)100

public static int sum(byte a, byte b)
{
    return a+b;//返回类型为int
}
```
- 浮点数的默认类型是double

BigDecimal类可以表示任意精度的浮点数，避免精度丢失问题。
```java
BigDecimal a = new BigDecimal("0.1");// 通过字符串构造
BigDecimal b = BigDecimal.valueOf(0.2);// 通过静态方法构造
BigDecimal c = a.add(b);
BigDecimal d = a.divide(b, 2, RoundingMode.HALF_UP);// 除法需要指定精度和舍入模式，避免无限循环报错
System.out.println(c); // 输出 0.3，不使用BigDecimal时会输出0.30000000000000004
```

#### StringBuilder
可变字符串，效率高于String。

| 方法 | 说明 |
|---|---|
| `StringBuilder append(任意类型)` | 在字符串末尾追加内容 |
| `StringBuilder reverse()` | 反转字符串 |
| `StringBuilder insert(int offset, String str)` | 在指定位置插入内容 |
| `StringBuilder delete(int start, int end)` | 删除指定范围的内容 |

返回值均为StringBuilder对象本身，因此可以链式调用。


示例：
```java
StringBuilder sb = new StringBuilder("Hello");
sb.append(" World");
System.out.println(sb.toString()); // 转换回String输出
```

#### 表达式类型转换
最终结果由表达式的最高类型决定；
byte、short、char在表达式中运算时直接提升为int参与运算。

#### 运算符
+号在字符串运算中起连接作用，"abc"+5得"abc5","abc"+5+'a'得"abc5a", 'a'+5+"abc"得"102abc"

b = a++;//先赋值再自加
b = ++a;//先自加再赋值

扩展赋值运算符：+=等五种
a += b ：a = **(a的类型)** a + b

短路与、或(&&、||)：若左边为false/true，则不执行右边(如++b>1不会导致自加)提前返回结果。

#### 方法中可变参数
方法签名中定义的特殊的形参，在**类型**后加上...，表示该形参可以接受任意数量的实参。  
一个方法中的形参列表中，**只能有一个**可变参数，并且需要放在最后。

```java
public static int sum(int... nums) {
    int total = 0;
    for (int n : nums) {total += n;}// 可变参数在方法中视作数组
    return total;
}
// 调用
sum(); // 不传
sum(1,2,3,4,5); // 多个
sum(new int[]{1,2,3,4,5})// 数组
```


### 流程控制语句
#### switch
表达式支持数据类型：byte、short、char、int、String、枚举类型；**不支持**double、float、long
(因为储存小数靠二进制拟合，0.3实际上是0.300..004之类，导致无法正确匹配)  

case的值必须是确定的字面量，且不能重复
break关键字是可选的，如果没有则**执行下一个case**(穿透性，当几种case处理代码一致时可以复用)；如果有，则跳出switch语句。
default也是可选的。

### 注解
注解是代码中的一种元数据，用于为程序元素（类、方法、变量等）提供额外的信息。本质是一种特殊的接口。

#### 元注解
元注解是用于注解的注解，主要有以下几种：
- @Retention：定义生命周期-源码/类文件/运行时
- @Target：定义适用范围-类/方法/变量等
- @Documented：将注解包含在Javadoc中
- @Inherited：允许子类继承父类的注解

#### 自定义注解
使用 @interface 关键字定义注解。  
```java
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)// 元注解
public @interface MyAnnotation {// 自定义注解
    String value();// 当注解只有一个属性时且命名为value，使用时可以省略属性名
}

// 使用注解示例
@MyAnnotation("example value")
public class MyClass {
    System.out.println(MyClass.class.getAnnotation(MyAnnotation.class).value());
}
```
#### 注解的解析
使用反射机制获取注解信息，并根据注解执行相应的逻辑。  
getDeclaredAnnotations()：获取当前对象上所有注解
getAnnotation(Class<T> annotationClass)：获取指定类型的注解对象


### 导第三方包
使用import语句导入第三方包中的类或接口，以便在代码中使用。  
语法：
```java
import 包名.类名;
import 包名.*;
```
在项目中创建lib文件夹，放入jar包，然后在jar包上右键选择“Add as Library”。


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
             对于引用变量（数组）：其地址不可改变，所指向对象的内容值可以改变

常量名采用全大写，单词间以下划线分割。

### 多态
父类引用指向子类对象，通过调用父类中定义的方法，实际执行相应子类重写后的方法。

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

### 反射
作用：
- 1.在运行时动态获取类的信息（类名、方法、属性等），并可以动态创建对象、调用方法和访问属性（可以通过setAccessible(true)访问私有属性）。  
- 2.可以绕过泛型（因为类型擦除）进行操作。
- 3.适合用于通用框架开发，如Spring、Hibernate等。

```java
// 1从对象获取
Class<?> clazz = obj.getClass();
// 2类名获取类对象
Class<?> clazz = Class.forName("com.example.MyClass");
// 3从类获取
Class<?> clazz = MyClass.class;

Object obj = clazz.getDeclaredConstructor().newInstance();// 获取构造器，再创建对象
Method method = clazz.getMethod("myMethod", String.class);// 获取方法
method.invoke(obj, "Hello");// 调用方法
Field field = clazz.getDeclaredField("myField");// 获取属性
field.setAccessible(true);// 取消访问检查，可以访问私有属性
field.set(obj, 42);// 设置属性值
```

### 特殊类
#### 单例类（单例设计模式）
**构造器私有化，以在类内创建唯一的对象**，确保某个类只能创建一个对象（应用如任务管理器）。  

```java
// 单例类 single instance
public class A{
    // 在类内创建唯一的对象
    // 为保护该对象不在外部被置成null：
    // 若使用public修饰，则加上final；若使用private修饰，则使用方法返回对象
    private static A a = new A();//懒汉式：private static A a;

    // 构造器私有，避免在类外被调用，只能在类内创建对象
    private A(){
    }

    public static A getObject(){
        // 懒汉式
        // if (a == null){
        //     a = new A();
        // }
        return a;
    }
}
// 饿汉式单例：在获取类的对象前就已经创建好对象
// 懒汉式单例：在获取类的对象时才创建对象，即在返回方法内创建对象
```

#### 枚举类enum
用于信息分类和标识，常应用于switch的case处
```java
public enum PlayerType {
    TENNIS,
    FOOTBALL,
    BASKETBALL
}

// 反编译结果
public final class PlayerType extends Enum
{

    public static PlayerType[] values(){
        return (PlayerType[])$VALUES.clone();
    }
    public static PlayerType valueOf(String name){
        return (PlayerType)Enum.valueOf(com/cmower/baeldung/enum1/PlayerType, name);
    }
    private PlayerType(String s, int i){
        super(s, i);
    }

    public static final PlayerType TENNIS;
    public static final PlayerType FOOTBALL;
    public static final PlayerType BASKETBALL;
    private static final PlayerType $VALUES[];

    static 
    {
        TENNIS = new PlayerType("TENNIS", 0);
        FOOTBALL = new PlayerType("FOOTBALL", 1);
        BASKETBALL = new PlayerType("BASKETBALL", 2);
        $VALUES = (new PlayerType[] {
            TENNIS, FOOTBALL, BASKETBALL
        });
    }
}
```

    1.枚举都是继承自Enum的最终类，不可被继承；
    2.枚举类的第一行只能罗列常量的名称，每个常量都是枚举类的一个对象；
    3.枚举类的构造器私有，因此不会对外创建对象


#### 抽象类abstract--模板
使用abstract关键字修饰类和方法。  
抽象方法**没有方法体，只有方法声明**。
如public abstract void m();

1.抽象类中可以没有抽象方法，但**抽象方法必须在抽象类中**。
2.抽象类**不能创建对象**，仅作为父类以供继承。
3.抽象类的子类必须**重写抽象类的全部抽象方法**。

因此抽象类强迫子类重新实现抽象方法，避免方法不适用。常用于多态

### 接口interface--功能
定义规范，分化出不同实现类，使得可以在不同实现类中灵活切换。  

使用 **implements**(实现)关键字 通过接口实现“多继承”，注意要重写所有接口的抽象方法。  
// class C implements A , B {//重写全部抽象方法}

#### 规则
    接口中只允许定义常量（默认加public static final，因此实际为常量）
    接口中允许定义抽象方法
    接口中允许定义特定实例方法（Java 8 之后）
    接口不能创建对象（无构造器、只有抽象方法）
    接口也可以用于多态（相当于义父，地位低于父类）

#### JDK8后新增三种实例方法
增强了接口的功能，且添加功能时避免已有的实现类需要重写。

1.默认方法  
使用default修饰，默认会被加上public。  
只能由**接口的实现类对象**调用。

2.私有方法  
使用private修饰
只能被**接口中其它实例方法**调用。

3.静态方法  
使用static修饰，默认会被加上public。  
只能由**接口名**调用。

#### *接口与继承的注意事项
1、接口可以多继承，且同名方法会合并；  
2、若多个接口出现方法签名冲突，则不支持多继承，也无法被实现；
```java
interface A {void show();}
interface B {
    void show();
    // String show();
}

interface C extends A , B{
    // 此时C内相当于是A B的结合
    // 若B中show()返回值类型是String，则签名冲突报错
}
```
3、一个类继承了父类，同时又实现了接口，会优先调用父类中的同名方法；  
4、可以通过重写冲突方法来规避2中的报错。  

如果一定要调用3 4中接口的方法，使用 接口名.super.方法名 进行调用。

### 代码块
分为两种：
静态代码块： static{...}  
类加载时自动执行，只会执行一次（类只加载一次），常用于静态变量初始化。  

实例代码块： {...}
创建对象时自动执行，常用于对实例变量初始化。

### 内部类
定义在另一个类内部的类。  

#### 1.成员内部类
成员内部类寄生于 *外部类的**对象***。  
成员内部类可以直接访问外部类的所有成员属性。

```java
public class Outer {
    int num = 1;

    public Outer () {
        new Inner().print();// 外部类要通过创建成员内部类的对象，才可以访问内部类的成员
    }

    class Inner {
        public void print() {
            System.out.println(num);// 直接访问外部类
        }
    }
}

// 因此访问内部类时，先得创建一个外部类的对象
Outer.Inner a = new Outer().new Inner();
a.print();
```

#### 2.静态内部类
由static关键字修饰，只在创建时加载一次，不允许访问外部类中 *非static* 的变量和方法（即外部类对象）。  

Outer.Inner a = new Outer.Inner();

#### 3.局部内部类
局部内部类是定义在一个方法或者一个作用域里面的类，其生命周期仅限于作用域内。
不能被权限修饰符(public、private、protected、static)修饰


#### 4.匿名内部类（子类对象）
在创建对象的同时，当场声明并实例化的一个类。用于创建只需要使用一次的、临时的实现类。  
 主要是用来继承其他类或者实现接口，并不需要增加额外的方法，方便对继承的方法进行实现或者重写。

**唯一一种没有构造方法的类**，是直接通过new关键字创建的一个子类对象。既是类也是对象。

格式：
```java
new 类/接口(参数值){
    // 匿名类的类体 (可以包含字段、方法等)
    // 实现接口的方法 或 重写父类的方法
}
```
```java
Comparator<String> lengthComparator1 = new Comparator<String>() {
    @Override
    public int compare(String s1, String s2) {
        return Integer.compare(s1.length(), s2.length());
    }
};
// 2. Lambda 表达式写法 (简洁)
Comparator<String> lengthComparator2 = (s1, s2) -> Integer.compare(s1.length(), s2.length());
// 3. 使用方法引用 (更简洁)
Comparator<String> lengthComparator3 = Comparator.comparingInt(String::length);
```

### 函数式编程
#### Lambda表达式
用于替代**函数式接口(有且仅有一个抽象方法)**的匿名内部类对象。  
用注解 @FunctionalInterface 来声明函数式接口。

格式：  
(被重写方法的形参列表) -> {被重写方法的方法体}

Lambda语句简化规则：  
参数类型可以省略不写；单个参数可以省略()；只有单行代码时，可以省略{}，同时需要去掉分号和return（如果是return语句的话）

#### 方法引用
1.静态方法引用  
用法：当Lambda表达式调用了一个静态方法时，且->前后参数形式一致。  
格式：**类名**::静态方法名

```java
Class Arr {
    private int num;

    public static int minus(Arr a1, Arr a2){
        return a1.getNum() - a2.getNum();
    }
}

Arr[] arr1 = new Arr[2];
Array.sort(arr1, (a1,a2)->a1.getNum() - a2.getNum())
Array.sort(arr1, Arr::minus)
```

2.实例方法引用  
用法：当Lambda表达式调用了一个实例方法时，且->前后参数形式一致。  
格式：**对象名**::实例方法名

3.特定类的方法引用  
用法：当Lambda表达式调用了一个实例方法时，且第一个参数是方法的主调，后续参数均为该方法的参数。  
格式：特定类名::方法名

```java
Arrays.sort(names, (n1,n2) -> n1.compareToIgnoreCase(n2))
Arrays.sort(names, String::compareToIgnoreCase)
```

4.构造器引用  
用法：当Lambda表达式只是在创建对象，且->前后参数形式一致。  
格式：类名::new

```java
Class Car{
    private String name;
}

@FunctionalInterface
public interface CarFactory {
    Car getCar(String name);// 函数式接口的抽象方法
}

// -----------简化示例-----------
CarFactory cf = new CarFactory() {
    @Override
    public Car getCar(String name) {
        return new Car(name);
    }
};
// Lambda表达式：简化为参数 -> 方法体
CarFactory cf = name -> new Car(name);
// 构造器引用：L式只是在创建对象，实际需要提供的只有类名
CarFactory cf = Car::new;

// -----------实际使用-----------
Car c1 = cf.getCar("奔驰");
// 本例重写的方法相当于实现了
Car c1 = new Car("奔驰");
```

### API
#### String
1.通过new创建新对象 

提供了四种构造器API：可创建空字符串，根据字符串、字符数组、byte数组创建。  
```java
char[] chars = ['h','e','l','l','o']
String s = new String(chars)
```

与常规初始化（String s = 'hello'）不同的地方：  
常规方法的字符串对象存储在*字符串常量池*，相同内容的对象s1 s2实际是同一个对象；  
而new方式会创建新的对象（**即使对象内容相同**）放在堆内存中

2.部分常用API
获取index处的字符: charAt(int index); 将字符串转换为字符数组char[]: toCharArray();  
忽略大小写比较: equalsIgnoreCase(); 截取:  substring(bIndex, eIndex); 替换: replace(target, replacement);  


## 泛型
在**编译阶段**约束数据类型，保证数据类型一致性，避免强制转换异常。  
泛型只在编译时起作用，**运行时并不会保留**泛型类型信息。

```java
// 不指定泛型的ArrayList可以存放任何类型的数据，因为所有类型都继承自Object类。
ArrayList list = new ArrayList();// 实际是ArrayList<Object>
list.add("text");
list.add(new Date());
String str = (String)list.get(0);// 取出数据的时候需要强制类型转换。因为存放的是Object类型的元素，编译器不知道元素本身的类型。
```
可以用各字母指代泛型，常用<E/T/K/V>，意义分别是Element/Type/Key/Value，

### 包装类
泛型只支持对象类型，**不支持基本数据类型**。  
因为类型擦除后泛型容器在底层实际上按照一个 Object[] 数组来存储元素，而基本数据类型不属于object的子类。  
于是使用包装类Integer、Character(其余均是首字母大写)，它们是基本数据类型的对象版本，并且可以**直接当基本类型使用**（自动装拆箱）。  

```java
// 手动包装
Integer in1 = Integer.valueOf(100); // 创建100的对象
// 实际上会自动包拆装，直接使用即可
Integer in2 = 100;
```

可以使用包装类的方法实现与字符串之间的相互转换。
```java
String s = Integer.toString(10);
int i = Integer.parseInt("99")
```

### 类型擦除
JVM编译时会把泛型的 类型变量 擦除，并替换为限定类型（没有限定则是Object），这会导致一些问题

下例类型变量String和Date在擦除后会自动消失，method方法的实际参数是ArrayList list，因此导致编译失败
```java
public static void method(ArrayList<String> list) {
        System.out.println("ArrayList<String> list");
    }

public static void method(ArrayList<Date> list) {
        System.out.println("ArrayList<Date> list");
    }
```

### 泛型类/接口/方法
泛型类
```java
class MyArrayList<E> {
    private Object[] elementData;
    private int size = 0;

    public MyArrayList(int initialCapacity) {
        this.elementData = new Object[initialCapacity];
    }
    
    public boolean add(E e) {
        elementData[size++] = e;
        return true;
    }
    
    E elementData(int index) {
        return (E) elementData[index];
    }
}
```
泛型接口/方法：可以在实现时才指定具体类型，使得接口/方法更通用。
```java
public interface Printer<T> {// 泛型接口
    void print(T data);
    }

public <T> T[] toArray(T[] a) {// <T> 声明T作为泛型；T[]是返回类型及参数类型，可无
        return (T[]) Arrays.copyOf(elementData, size, a.getClass());
    }

public E get(int index) {// 不是泛型方法，只是使用类的泛型
        return (E) elementData[index];
    }
```
### 泛型限定符extends
限定符 extends 可以缩小泛型的类型范围
```java
class GrandFather {
    void show() {System.out.println("I am GrandFather");}
}
class Father extends GrandFather {
    @Override
    void show() {System.out.println("I am Father");}
}
class Child extends Father {
    @Override
    void show() {System.out.println("I am Child");}
}
class MyList<E extends Father> {}

MyList<GrandFather> list = new MyList<>();// 错误，E只能是Father或其子类
MyList<Father> list = new MyList<>();// 正确
MyList<Child> list = new MyList<>();// 正确
list.add(new GrandFather());// 错误，GrandFather不是Father的子类
list.add(new Father());
list.add(new Child());// 正确，Child是Father的子类
``` 

### 通配符<?>
**通配符<?>**用来解决类型不确定的情况，例如在方法参数或返回值中使用。

**上限通配符<? extends T>**表示通配符只能接受 T 或 T的子类。

**下限通配符<? super T>**表示通配符必须是 T 或 T的超类。
```java 
public static void printList(List<?> list) {...}// 可以接受任意类型的List，如List<Integer>、List<String>等
public static void printNumberList(List<? extends Number> list) {...}// 可以接受List<Integer>、List<Double>等
```
假设有一个类Animal及其子类 Dog和Cat。则对于List<? super Dog>集合，类型参数必须是Dog或其父类类型。  
可以向该集合中添加Dog类型的元素，也可以添加它的子类**元素**。但是不能向其中添加Cat类型的元素。

**PECS (Producer Extends, Consumer Super)原则**
? extends T：可以安全地读取数据，但限制了写入。  
? super T：可以安全地写入数据，但限制了读取。


## 集合框架（容器）
![group](/content/post/instruction/java/group.png)
### Collection

**常用方法**
| 方法 | 说明 |
|---|---|
public boolean add(E e)	| 把给定的对象添加到当前集合中
public void clear()	| 清空集合中所有的元素
public boolean remove(E e)	|把给定的对象在当前集合中删除
public boolean contains(Object obj)	|判断当前集合中是否包含给定的对象
public boolean isEmpty()	|判断当前集合是否为空
public int size()	|返回集合中元素的个数。
public Object[] toArray()	|把集合中的元素，存储到数组中

#### 遍历方法
1.普通for循环  
只适用于有索引集合。

2.迭代器
```java
Iterator it = list.iterator();// 初始位于第一个元素处
while (it.hasNext()) {// 判断是否还有下一个元素，返回true/false
    System.out.print(it.next());// 读取当前元素再移位
}
```
3.增强for循环（for-each） 
本质是迭代器遍历集合的简化写法。  

格式：for(元素数据类型  变量名 : 要遍历的数组/集合)
```java
for (String s : list) {
    System.out.print(s);
}
```
4.forEach方法  
源码：
```java
default void forEach(Consumer<? super T> action) {
    Objects.requireNonNull(action);// 判断传入的action对象是否为null
    for (T t : this) {
        action.accept(t);// 遍历action对象
    }
}
```
调用：
```java
list.forEach(new Consumer<Integer>(){
    @Override
    public void accept(Integer int) {
        System.out.println(int);
    }})
// 简化
list.forEach(int -> System.out.println(int);)
list.forEach(System.out::println)// 已经指明对象为list，forEach方法不需要其他参数
```
##### 并发修改异常
对于有索引的集合，在一次遍历中使用list.remove()，后续的元素位置会前移，而迭代器的游标位置后移，导致跳过了一个元素。  
1 在for循环中删除元素时进行i--，或者从后往前进行循环  
2 迭代器会检测是否出现并发修改异常并报错，使用迭代器的**it.remove()**可以避免，对于**无索引的集合**只能使用迭代器  
3 增强for和forEach方法无法解决并发修改异常，只适合用于遍历

#### List
均有序、可重复、有索引，因此多了与索引相关的方法。

有ArrayList、LinkedList两种实现类。其底层数据结构分别是动态数组和双向链表，应用场景不同（查询/增删）。

· ArrayList第一次添加元素时，默认容量为10，后续每次扩容为原来的1.5倍。  
· LinkedList可以从头尾两端添加元素，多了与**头尾相关**的方法。常用于构建队列、栈等。

#### Set
所有set都不重复、无索引；set本身无序。

**特点**：
有HashSet、LinkedHashSet、TreeSet三种实现类，其增删查改的速度均快。  
    HashSet底层是哈希表，无序。  
    LinkedHashSet底层是哈希表+链表，有序（默认插入顺序）。  
    TreeSet底层是红黑树，有序（默认升序排序）。

##### HashSet
基于哈希表实现，依赖于元素的 `hashCode()` 和 `equals()` 方法。  
如果希望hashSet认为内容相同的对象是一样的，则需要重写这两个方法（直接使用自动重写）。
```java
@Override
public int hashCode() {
    return Objects.hashCode(this.name, this.age);
}
```
流程:  
1.创建默认长度为16的数组，默认加载因子为0.75。
2.添加元素时，首先计算元素的哈希值（`hashCode`），以确定元素的存储位置。
3.判断该位置是否有其他元素，若无元素则创建新节点；若有元素则调用 `equals()` 方法比较，丢弃重复元素，将非重元素添加到链表末尾。

扩容：当链表长度超过16*0.75=12的时候，会扩容为原来的2倍。另外当链表长度超过8且数组长度大于64时，会转换为红黑树。

**哈希值**
java中每个对象都有哈希值，哈希值是int类整数值。  
同一个对象多次调用hashCode()方法，返回的哈希值相同，即使对象内容不同。  
不同的对象的哈希值大概率不同，但也可能相同（哈希冲突）。

**哈希表**
基于数组、链表、红黑树实现。

**红黑树**
可以自平衡的二叉树，查询、插入、删除的时间复杂度都为O(log n)。  

##### LinkedHashSet
也是基于哈希表实现，但是每个元素都增加了双链表机制记录前后元素，因此**插入有序**。  
元素实际为实现类Entry的对象，Entry对象包含：key-value对（底层的map），前后节点的指针，下节点的指针，哈希值。

##### TreeSet
基于红黑树实现，元素自动升序排序。  
对于自定义类型，默认无法直接排序，需要重写compareTo()方法。重写方法可以保留相同值的元素。

### Map
也叫做键值对集合，格式为：{key1:value1, key2:value2, ...}。   
其中key不可重复，value可以重复。key与value一一对应。

有HashMap、LinkedHashMap、TreeMap三种实现类。  
其特点见上面set的三种实现类，因为实际上set系列集合底层是基于map实现的（只使用key丢弃了value）。
```java
Map<String, Integer> map = new HashMap<>();
```

**常用方法**
| 方法 | 说明 |
|---|---|
public V put(K key, V value)	|添加元素，如果key已存在，则用新的value覆盖旧的value，并返回旧的value
public int size()	|获取集合的大小
public void clear()	|清空集合
public boolean isEmpty()	|判断集合是否为空, 为空返回true
public V get(Object key)	|根据键获取对应值
public V remove(Object key)	|根据键删除整个元素
public boolean containsKey(Object key)	|判断是否包含某个键
public boolean containsValue(Object value)	|判断是否包含某个值
public Set<K> keySet()	|获取全部键的集合
public Collection<V> values()	|获取Map集合的全部值

**遍历方式**
1.键找值  
先获取键的集合（keySet()方法），再通过键获取值（get(Object key)方法）

2.键值对  
entrySet()方法获取键值对的集合，再通过getKey()、getValue()方法获取键和值。

```java
Set Map.Entry<String, Integer> entrySet = map.entrySet();
for (Map.Entry<String, Integer> entry : entrySet) {
    String key = entry.getKey();
    Integer value = entry.getValue();
    System.out.println(key + ":" + value);
}
```
3.forEach方法  
forEach方法可以遍历键值对，并对键值对进行操作。
```java
map.forEach(new BiConsumer<String, Integer>() {
    @Override
    public void accept(String key, Integer value) {
        System.out.println(key + ":" + value);
    }
});
// 简化
map.forEach((key,value) -> System.out.println(key + ":" + value));
```

### Stream流
用于对集合、数组进行操作的API，可以进行过滤、排序、映射、聚合等操作。

示例：
```java
List<String> newList = list.stream().filter(s->s.startWith("a")).filter(s->s.length()==2).collect(Collectors.toList());  // 过滤以a开头，长度为2的字符串，并收集到新List中
```

#### 获取stream流
1. 集合的stream方法
```java
Stream<String> stream1 = list.stream();
Stream<String> stream11 = map.keySet().stream();// map先获取键或者值 map.values().stream() 或键值对map.entrySet().stream()
```
2. 数组的stream方法
```java
Stream<String> stream2 = Arrays.stream(arr);
```
3. stream类的方法
```java
Stream<String> stream3 = Stream.of("a", "ab", "ca");
```

#### 中间处理方法
调用完成后返回新的Stream流，可以链式调用。

| 方法 | 说明 |
|---|---|
Stream<T> filter(Predicate<? super T> predicate)	| 过滤元素，返回符合条件的元素
Stream<T> sorted()	|对元素进行升序排序
Stream<T> sorted(Comparator<? super T> comparator)	|按照指定规则排序
Stream<T> limit(long maxSize)	|获取前几个元素
Stream<T> skip(long n)	|跳过前几个元素
Stream<T> distinct()	|去除流中重复的元素（自定义类的对象需要重写hashCode和equals方法）
<R> Stream<R> map(Function<? super T, ? extends R> mapper)	|对元素进行加工，并返回对应的新流
static <T> Stream<T> concat(Stream a, Stream b)	|合并a和b两个流为一个流

#### 终止方法
调用完成后返回结果，终止stream流。

| 方法 | 说明 |
|---|---|
void forEach(Consumer action)	|对此流运算后的元素执行遍历
long count()	|统计此流运算后的元素个数
Optional<T> max(Comparator<? super T> comparator)	|获取此流运算后的最大值元素放入optional容器，避免空指针异常
Optional<T> min(Comparator<? super T> comparator)   |获取此流运算后的最小值元素
Object[] toArray()	|将流元素收集到数组
R collect(Collector<? super T> collector)	|将流元素收集到容器中，collector类可以进一步收集到集合中


collect方法可以收集到集合中，如toList()、toSet()、toMap(keyMapper, valueMapper)等。
例：Map<String,Integer> map = stream.collect(Collectors.toMap(Students::getName, Students::getAge));// 收集到map集合

## IO流
用于读取文件、网络中的数据。实际开发中常用框架如Apache Commons IO、Google Guava等来简化IO操作。  

按内容分为：字节流（处理所有类型数据）和字符流（处理文本文件）。  
按方向分为：输入流（从源读取数据）和输出流（向目的写入数据）。

![IO流分类](/content/post/instruction/java/IOStream.png)

### 文件类操作
| 方法 | 说明 |
|---|---|
file.exists()	|判断文件是否存在
file.isFile()	|判断是否为文件
file.isDirectory()	|判断是否为目录
file.getName()	|获取文件名
file.length()	|获取文件大小
file.lastModified()	|获取文件最后修改时间
file.createNewFile()	|创建文件
file.delete()	|删除文件或空文件夹
file.mkdir()	|创建目录
file.mkdirs()	|创建多级目录
File[] listFiles()	|获取目录下所有一级文件对象到文件对象数组中

### 输入输出流

#### 字节输入输出流
**建立**
|类|说明|
|---|---|
InputStream in = new FileInputStream(file/path);	|建立字节输入流
BufferedInputStream bin = new BufferedInputStream(in);	|缓冲字节输入流，提高读取效率
OutputStream out = new FileOutputStream(file/path, boolean append);	|建立字节输出流，append为true时表示追加写入
BufferedOutputStream bout = new BufferedOutputStream(out);	|缓冲字节输出流，提高写入效率

**读写方法**
|读写方法|说明|
|---|---|
int read(int b)	|读取一个字节
int read(byte[] b [, int offset, int len])	|从offset开始读取len个字节到byte数组b中
byte[] ReadAllBytes |读取整个文件内容到字节数组中
write(int b)	|写出一个字节
write(byte[] b [, int pos, int len])	|写出字节数组
close() throws IOException	|关闭流

**资源释放方案**
1. try-catch-finally
```java
try{
    InputStream fin = new FileInputStream(file/path);	|建立字节输入流
    OutputStream fout = new FileOutputStream(file/path, boolean append);
    // 读写操作
}catch(IOException e){
    // 异常处理
}finally{// 资源释放
    fin.close();
    fout.close();
}
```
2. try-with-resources(推荐)
在try后面的小括号内创建流对象，JVM会自动释放资源，避免忘记关闭流。  
资源对象必须实现AutoCloseable接口（所有流类均已实现）。  
```java
try(InputStream fin = new FileInputStream(file/path);
    OutputStream fout = new FileOutputStream(file/path, boolean append);){
    // 读写操作
}catch(IOException e){
    // 异常处理
}
```

#### 字符输入输出流
**建立**
|类|说明|
|---|---|
FileReader fr = new FileReader(file/path);	|建立字符输入流
BufferedReader br = new BufferedReader(fr);	|缓冲字符输入流，提高读取效率
FileWriter fw = new FileWriter(file/path, boolean append);	|建立字符输出流
BufferedWriter bw = new BufferedWriter(fw);	|缓冲字符输出流，提高写入效率

|读写方法|说明|
|---|---|
int read()	|读取一个字符
int read(char[] cbuf [, int offset, int len])	|从offset开始读取len个字符到char数组cbuf中
char[] ReadAllChars |读取整个文件内容到字符数组中
String readLine()	|缓冲流独有的功能：读取一行文本
void write(int c)	|写出一个字符
void write(String str [, int pos, int len])	|写出字符串
void write(char[] cbuf [, int pos, int len])	|写出字符数组
void newLine()	|缓冲流独有的功能：写出一个换行符

字符输出流写出数据后，需要**调用 flush() 或 close() 方法**将缓冲区数据强制写出，否则数据可能在内存中未写入硬盘文件。

#### 字符输入/输出转换流
用于解决字节流与字符流之间的转换问题，指定编码格式。
|类|说明|
|---|---|
InputStreamReader isr = new InputStreamReader(InputStream in, String charsetName);	|字节输入流转换为字符输入流，可指定编码格式
OutputStreamWriter osw = new OutputStreamWriter(OutputStream out, String charsetName);    |字节输出流转换为字符输出流，可指定编码格式

### 打印流
更高效的输出流，提供了多种重载的print()和println()方法，可以**直接输出**各种数据类型。

|类|说明|
|---|---|
PrintStream ps = new PrintStream(OutputStream/File/Path out, boolean autoFlush, String charsetName);	|字节打印流
PrintWriter pw = new PrintWriter(Writer out, boolean autoFlush);	|字符打印

### 特殊流
数据流(Data Stream)：用于读写基本数据类型和字符串，提供了readInt()、writeInt()等方法。
对象流(Object Stream)：用于读写Java对象，提供了readObject()、writeObject()方法。对象必须实现Serializable接口。

|类|说明|
|---|---|
DataInputStream dis = new DataInputStream(InputStream in);	|字节数据输入流
DataOutputStream dos = new DataOutputStream(OutputStream out);	|字节数据输出流
ObjectInputStream ois = new ObjectInputStream(InputStream in);	|字节对象输入流
ObjectOutputStream oos = new ObjectOutputStream(OutputStream out);	|字节对象输出流


## 异常
有错误则抛出异常，但不会终止程序。
### 异常分类
![异常分类](/content/post/instruction/java/exception.png)
Exception和Error都继承了Throwable类。只有Throwable类（或者子类）的对象才能使用throw关键字抛出，或者作为catch的参数类型。

Checked Exception（受检异常）：在编译期被检查的异常，必须显式处理（通过try-catch 或 throws）。正逐步淘汰。
Unchecked Exception（非受检异常 / 运行时异常）：运行时异常，不需要在编译期显式处理。继承自：RuntimeException

·NoClassDefFoundError 和 ClassNotFoundException的区别：  
都由于系统运行时找不到要加载的类导致，但触发原因不同。
    NoClassDefFoundError：程序在**编译时可以**找到所依赖的类，但是在**运行时找不到**指定的类文件；原因可能是编译的类文件被删除。
    ClassNotFoundException：当动态加载 Class 对象的时候找不到对应的类时抛出该异常；原因可能是要加载的类不存在或者类名写错了。

### 异常处理
#### try-catch(-finally)
```java
try{
// 可能发生异常的代码
}catch (exception(type) e(object)){
// 异常处理代码
}catch (exception2 e2){
// 可以有多个catch
}finally{
// 无论是否发生异常，都会执行的代码
}
```
多个catch块捕获异常时，**子类异常必须在前，父类异常在后**，否则会报错。如ArithmeticException是Exception的子类，其范围更具体，因此要放在前面。

即便是try块中执行了return、break、continue这些跳转语句，finally块也会被执行。  
不执行的情况：死循环 、JVM退出（System.exit(0)）等。

#### throws和throw
throws关键字用于声明异常，它的作用和try-catch相似；而throw关键字用于显式的抛出异常。  
throws关键字后面跟的是异常的名字；而throw关键字后面跟的是异常的对象。
```java
public void myMethod1() throws ArithmeticException, NullPointerException{...};// 方法体中有这些异常则抛出
if(b==0){throw new ArithmeticException("算术异常");}// 显式抛出异常
```

### 自定义异常
1.继承Exception或RuntimeException类
2.提供两个构造器：无参和带String参数（异常信息）
```java
public class MyException extends Exception {
    public MyException() {
        super();// 调用父类的无参构造器
    }

    public MyException(String message) {
        super(message);// 调用父类的带String参数的构造器
    }
}
```
### JUnit
测试方法必须使用@Test注解标记，且方法必须是public无参无返回值。  
```java
import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testMyMethod() {
        // 测试代码
    }
}
```


## 多线程

### 创建线程
1. 继承Thread类，重写run()方法
```java
public class MyThread extends Thread {
    @Override
    public void run() {
        // 线程执行的代码
    }
}
MyThread t1 = new MyThread();
t1.start();// 启动线程，调用run()方法
```
编码简单，但Java单继承的缺点也限制了其使用。  
*执行结果不能直接返回*（run方法返回值是void）。

2. 实现Runnable接口，重写run()方法
```java
public class MyRunnable implements Runnable {
    @Override
    public void run() {
        // 线程执行的代码
    }
}
MyRunnable r1 = new MyRunnable();// 创建Runnable对象
new Thread(r1).start();// 将Runnable对象传递给Thread构造器，启动线程

// 匿名内部类写法
new Thread(new Runnable() {
    @Override
    public void run() {
        // 线程执行的代码
    }
}).start();
```
扩展性强，但需要多创建一个Runnable对象。  
*执行结果不能直接返回*（run方法返回值是void）。

3. Callable接口，重写call()方法
Callable接口可以有返回值，并且可以抛出异常。
```java
public class MyCallable implements Callable<Integer> {
    @Override
    public Integer call() throws Exception {
        // 线程执行的代码
        return 123;// 返回值
    }
}
MyCallable c1 = new MyCallable();
FutureTask<Integer> task = new FutureTask<>(c1);// FutureTask实现了Runnable接口
new Thread(task).start();// 启动线程，调用call()方法
Integer result = task.get();// 获取返回值，可能会阻塞等待线程执行完毕
```

### 线程的常用方法
| 方法 | 说明 |
|---|---|
| start() | 启动线程 |
| run() | 线程执行的代码 |
| sleep(long millis) | 让当前线程睡眠指定毫秒数 |
| join() | 调用该方法的线程优先执行 |
| isAlive() | 判断线程是否存活 |
| setName(String name) | 设置线程名称 |
| getName() | 获取线程名称 |   
| setPriority(int newPriority) | 设置线程优先级，范围1-10，默认5 |
| getPriority() | 获取线程优先级 |
| currentThread() | 获取当前线程对象 |

### 线程同步
多线程并发访问*共享资源*时，可能会出现数据不一致的问题。  
为了解决这个问题，可以使用以下几种方式进行线程同步：

1. **同步代码块**：在访问共享资源的代码块上加上`synchronized`关键字，表示该代码块是同步的，只有一个线程可以访问。
```java
public void myMethod() {
    synchronized (this) {// 锁对象:实例方法使用共享资源；静态方法使用字节码（类名.class）
        // 线程安全的代码
    }
}
```

2. **同步方法**：在方法上加上`synchronized`关键字，表示该方法是同步的，只有一个线程可以访问。
```java
public synchronized void myMethod() {
    // 线程安全的代码
}
```

3. **Lock锁**：使用`java.util.concurrent.locks.Lock`接口及其实现类（如`ReentrantLock`）来实现更灵活的线程同步。
```java
final Lock lock = new ReentrantLock();// 创建Lock对象
lock.lock();// 获得锁
try {
    // 线程安全的代码
} finally {
    lock.unlock();// 释放锁，之中的代码会被锁保护
}
```

### 线程池
线程池可以复用线程，避免频繁创建和销毁线程带来的开销，提高系统性能。

**常用方法：**  
| 方法 | 说明 |
|---|---|
| execut(Runnable task) | 提交**Runnable任务**到线程池，无返回值 |
| Future<T> submit(Runnable/Callable task) | 提交**Callable任务**到线程池，返回Future对象 |
| shutdown() | 任务**执行完毕后**，关闭线程池 |
| List<Runnable> shutdownNow() | **立即关闭**线程池，尝试停止所有正在执行的任务，返回队列中未执行的任务 |

**1.ThreadPoolExecutor类**
是线程池ExecutorService的核心类，可以通过构造器创建线程池。

构造器有七个参数，含义为：
- corePoolSize：核心线程数，线程池中始终保持的线程数。
- maximumPoolSize：最大线程数，线程池中允许的最大线程数。
- keepAliveTime：线程空闲时间，超过这个时间的空闲线程会被终止。
- unit：时间单位，keepAliveTime的时间单位。
- workQueue：任务队列，用于存放待执行的任务。
- threadFactory：线程工厂，用于创建新线程。
- handler：拒绝策略，当任务队列满且线程数达到最大值时，如何处理新任务。

创建临时线程的时机：当**核心线程全被占用**且**任务队列满**，同时线程数未达到maximumPoolSize时。

拒绝任务的时机：当**核心线程全被占用**且**任务队列满**，同时线程数已达到maximumPoolSize时。
```java
ThreadPoolExecutor executor = new ThreadPoolExecutor(
        5, 10, 60, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
for (int i = 0; i < 10; i++) {
    final int taskId = i;
    executor.submit(() -> {
        // 线程执行的代码
        System.out.println("Task " + taskId + " is running");
    });
}
executor.shutdown();// 关闭线程池
```

**2.Executors工具类**
本质是对ThreadPoolExecutor的封装，简化线程池的创建。

**工具类静态方法：**
| 方法 | 说明 |
|---|---|
| newSingleThreadExecutor() | 创建单线程的线程池，出现异常会自动补充 |
| newFixedThreadPool(int nThreads) | 创建固定大小的线程池，若有线程出现异常结束，会自动补充新的线程 |
| newCachedThreadPool() | 创建可缓存的线程池，线程数随任务数增加，空闲一分钟回收 |
| newScheduledThreadPool(int corePoolSize) | 创建固定大小的线程池，可以延迟或定时执行任务 |

**弊端：**
1. FixedThreadPool和SingleThreadExecutor允许请求的队列长度无限制，可能导致内存耗尽。  
2. CachedThreadPool和ScheduledThreadPool允许创建的线程数无限制，可能导致系统资源耗尽。  
因此**阿里禁用Executors**创建线程池。

示例：
```java
ExecutorService executor = Executors.newFixedThreadPool(5);// 创建固定大小的线程池
for (int i = 0; i < 10; i++) {
    final int taskId = i;
    executor.submit(() -> {
        // 线程执行的代码
        System.out.println("Task " + taskId + " is running");
    });
}
executor.shutdown();// 关闭线程池
```

## 网络编程
Java提供了丰富的网络编程API，主要包括以下几个方面：

1. **Socket编程**：用于实现TCP/IP协议的网络通信，包括客户端和服务器端的Socket编程。
2. **UDP编程**：用于实现基于UDP协议的网络通信，适用于对实时性要求较高的场景。
3. **HTTP编程**：用于实现基于HTTP协议的网络通信，包括发送HTTP请求和处理HTTP响应。
4. **WebSocket编程**：用于实现基于WebSocket协议的双向通信，适用于实时应用场景。

### Socket编程(TCP)
Socket编程是Java网络编程的基础，主要包括以下几个类：
| 类 | 说明 |
|---|---|
| Socket | 客户端Socket类，用于连接服务器 |
| ServerSocket | 服务器Socket类，用于监听客户端连接 |
| InetAddress | 表示IP地址的类 |
| SocketException | Socket操作异常类 |
#### 创建客户端Socket
```java
Socket socket = new Socket("localhost", 8080);// 连接服务器
OutputStream out = socket.getOutputStream();// 获取输出流
InputStream in = socket.getInputStream();// 获取输入流
// 读写数据等，输出后记得flush()
in.close();
out.close();
socket.close();// 关闭Socket
```
#### 创建服务器端Socket
```java
ServerSocket serverSocket = new ServerSocket(8080);// 监听端口
while (true) {
    Socket clientSocket = serverSocket.accept();// 接受客户端连接
    clientSocket.getInputStream();// 获取输入流
    clientSocket.getOutputStream();// 获取输出流
    // 读写数据等
    clientSocket.close();// 关闭客户端Socket
}
serverSocket.close();// 关闭服务器Socket
```
如果要处理多个客户端连接，可以为每个连接的socket创建一个新的线程。
```java
new Thread(() -> {
    clientSocket.getInputStream();// 获取输入流
    clientSocket.getOutputStream();// 获取输出流
    // 读写数据等
    clientSocket.close();// 关闭客户端Socket
}).start();
```

#### B/S架构
B/S（Browser/Server）架构是基于浏览器和服务器的网络架构，常用于Web应用开发。  
HTTP协议是B/S架构中最常用的协议，Java提供了HttpURLConnection类用于发送HTTP请求和处理HTTP响应。


### UDP编程
UDP编程主要包括以下几个类：
| 类 | 说明 |
|---|---|
| DatagramSocket | UDP Socket类，用于发送和接收数据包 |
| DatagramPacket | 数据包类，用于封装发送和接收的数据 |
#### 创建UDP客户端
```java
DatagramSocket socket = new DatagramSocket();// 创建UDP Socket
String message = "Hello, UDP!";
byte[] buffer = message.getBytes();
DatagramPacket packet = new DatagramPacket(buffer, buffer.length, InetAddress.getByName("localhost"), 8080);// 创建数据包
socket.send(packet);// 发送数据包
socket.close();// 关闭Socket
```
#### 创建UDP服务器端
```java
DatagramSocket socket = new DatagramSocket(8080);// 监听端口
byte[] buffer = new byte[1024];
DatagramPacket packet = new DatagramPacket(buffer, buffer.length);// 创建数据包
socket.receive(packet);// 接收数据包
String message = new String(packet.getData(), 0, packet.getLength());
System.out.println("Received message: " + message);
socket.close();// 关闭Socket
```