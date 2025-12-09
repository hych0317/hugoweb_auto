+++
title = 'Redis'
date = 2025-09-13T09:48:39+08:00
draft = false
categories = ['指令语法']
+++


## Redis
基于内存的键值型NoSQL数据库，单线程（因此每个命令具有原子性）
- NoSql(Not Only Sql)，不仅仅是SQL，泛指非关系型数据库。

Redis应用场景：缓存、消息队列、任务队列、分布式锁


## 通用命令
KEYs pattern	查找所有符合给定模式(pattern)的key
EXISTs key	检查给定key是否存在
TYPE key	返回key所储存的值的类型
TTL key	返回给定key的剩余生存时间(TTL, time to live)，以秒为单位
DEL key	该命令用于在key存在是删除key

## 数据类型
![Redis数据类型](./redis_types.jpg)
### String
又细分为普通字符串、整数和浮点数。  
字符串类型的最大空间不超过512MB。

**常用命令：**
SET key value	设置指定key的值为value
GET key	获取指定key的值
MSET key value [key value ...]	同时设置一个或多个key-value对
MGET key [key ...]	获取所有给定key的值

INCR key	将key中储存的数字值增一
DECR key	将key中储存的数字值减一
INCRBY key increment	将key中储存的数字值增加指定的增量increment
DECRBY key decrement	将key中储存的数字值减少指定的减量decrement

SETNX key value	只有在key不存在时，设置key的值为value
SETEX key seconds value	将key的值设为value，并将key的生存时间设为seconds秒
APPEND key value	如果key已经存在并且是一个字符串，APPEND命令将value追加到key原来的值的末尾
STRLEN key	返回key所储存的字符串值的长度

**KEY结构：**
Redis没有类似MySQL中Table的概念，使用多个单词形成层级结构以区分不同类型的数据。  

prj:user:1	{“id”:1, “name”: “Jack”, “age”: 21}
prj:dish:1	{“id”:1, “name”: “鲟鱼火锅”, “price”: 4999}

### Hash
Hash类型是一个键值对集合，适合用于存储对象。
其value本身是一个键值对集合。

**常用命令：**
HSET key field value	为key中的hash类型添加一个field-value对
HGET key field	获取key中的hash类型指定field的值
HMSET key field value [field value ...]	同时为key中的hash类型设置多个field-value对
HMGET key field [field ...]	获取key中的hash类型指定的多个field的值
HGETALL key	获取key中的hash类型的所有field-value对
HDEL key field [field ...]	删除key中的hash类型指定的一个或多个field
HLEN key	获取key中的hash类型的field数量
HKEYS key	获取key中的hash类型的所有field

### List
List类型是一个简单的字符串列表，按照插入顺序排序。