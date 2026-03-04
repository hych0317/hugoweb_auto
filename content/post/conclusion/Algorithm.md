+++
title = 'Algorithm'
date = 2024-08-24T21:01:10+08:00
draft = false
categories = ['conclusion']
description = ''
+++

## 数组
1. 双指针
- 有序数组：左右指针
- 无序数组：快慢指针
2. 滑动窗口
- 固定窗口：双指针确定窗口大小
- 动态窗口：根据条件动态调整窗口
3. 前缀和
- 计算区间和：prefixSum[j] - prefixSum[i-1]

### 题目
***27.*** 移除元素
双指针：fast遍历数组，slow记录不等于 val 的个数（位置）。   
if (nums[fast] != val) { nums[slow++] = nums[fast]; }

***209.*** 长度最小的子数组
滑动窗口：right指针扩展窗口，left指针收缩窗口。  
注意收缩使用while循环。

***15.*** 【三数之和】
排序+双指针：固定一个数，使用左右指针寻找另外两个数。

## 链表
***19.*** 删除链表的倒数第N个节点
双指针：first先走n步，然后second和first一起走，直到first到达末尾。

***160.*** 链表相交
双指针：pA，pB以不同顺序遍历两个链表，两指针相等时即为相交节点。
```java
ListNode pA = headA, pB = headB;
while (pA != pB) {
    pA = pA == null ? headB : pA.next;
    pB = pB == null ? headA : pB.next;
}
return pA;
```

***142.*** 环形链表 II
双指针：快慢指针，fast每次走两步，slow每次走一步。  
快慢指针相遇后，令另一个指针ptr从head开始每次走一步，其与slow相遇的节点即为环入口。

原理：设head到环入口距离为a，环入口到相遇点距离为b，相遇点到环入口距离为c。  
则有：2(a+b) = a+b+n(b+c) => a = c + (n-1)(b+c)，即head到环入口距离 = 相遇点到环入口距离加上若干个环的周长。  
  两倍慢指针路程 = 快指针路程  
因此ptr与slow相遇的节点即为环入口。

***206.*** 反转链表
使用三个指针：pre，cur，**next**。next保存cur的下一个节点，防止断链。


## 哈希表
哈希表：**通过哈希函数将键映射到值**。  
常用操作：插入、删除、查找，平均时间复杂度为O(1)。

面向问题：元素是否出现过，或是否在集合里

### 题目
***202. 快乐数***
对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，如果这个数最终变为1，那么这个数就是快乐数。也可能进入无限循环。  

使用哈希表记录出现过的平方和，避免进入循环。

***454.*** 四数相加 II
哈希表key为前两数组合的和，value为该和出现的次数。  
对于后两数组合的和，查找哈希表中是否存在相反数。
```java
HashMap<Integer, Integer> map = new HashMap<>();  
for (int i : A) for (int j : B) map.put(i + j, map.getOrDefault(i + j, 0) + 1);  
int count = 0;  
for (int i : C) for (int j : D) count += map.getOrDefault(-(i + j), 0);  
return count;
```


## 字符串
Java中字符串不可变，修改字符串会创建新对象。  
多使用StringBuilder进行字符串拼接。

### 题目
***541.*** 反转字符串 II
每隔2k个字符反转前k个字符。
```java
for (int i = 0; i < ch.length; i += 2*k) {
    int left = i, right = Math.min(i+k-1, ch.length-1);
    while (left < right) {
        char temp = ch[left];
        ch[left++] = ch[right];
        ch[right--] = temp;
    }
}
```

***151.*** 翻转字符串里的单词
去除多余空格，反转整个字符串，再反转每个单词。
```java
while (left <= right) {// left为整个串的开头位置，不移动
    int end = right;
    while (left <= right && s.charAt(right) != ' ') { right--; }
    sb.append(s.substring(right+1, end+1)).append(" ");
    while (left <= right && s.charAt(right) == ' ') { right--; }// 去空格
}
```

1047. 删除字符串中的所有相邻重复项
双指针，fast遍历输入串，slow维护结果串。  
双指针其实可以理解为分别操作两个串，由于fast一定比slow快，因此可以复用相同空间。
```java
public String removeDuplicates(String s) {
    char[] ch = s.toCharArray();
    int fast = 0;
    int slow = 0;
    while(fast < s.length()){
        if(slow > 0 && ch[fast] == ch[slow - 1]) { slow--; }// 当前slow位置未赋值，slow-1为结果串最后一个位置
        else { ch[slow++] = ch[fast]; }
        fast++;
    }
    return new String(ch,0,slow);
}
```

#### KMP算法
KMP算法：利用已经匹配过的信息，避免重复匹配。时间复杂度O(n+m)。
 
构建部分匹配表（next数组）：记录模式串中**每个子串**前后缀的最长公共部分长度。  
i++相当于给当前的子串增加了一个“尾巴”，我们只需要看新加的这个尾巴（s[i]）能不能和前缀新加的那个成员（s[j]）对上。
例如：
    ababab  next数组为[0,0,1,2,3,4]；
    aabcaa  next数组为[0,1,0,0,1,2]；
    aabcaab  next数组为[0,1,0,0,1,2,3]；

```java
int[] next = new int[n];
int j = 0; // 前缀末尾，也是最长公共部分长度
for (int i = 1; i < n; i++) { // 后缀末尾
    while (j > 0 && pattern.charAt(i) != pattern.charAt(j)) {
        j = next[j - 1];// 回溯到上个匹配位置
    }
    if (pattern.charAt(i) == pattern.charAt(j)) {
        j++;
    }
    next[i] = j;
}
```


***28.*** 实现 strStr()
返回子串在主串中第一次出现的位置。  
1. 使用双指针遍历主串和子串。
```java
int left = 0, right = 0;
while (left < s.length()) {
    if (s.charAt(left) == sub.charAt(right)) {
        right++;
        if (right == sub.length()) { return left-right+1; }
    } else {
        left -= right;// 下面有left++，不用加1
        right = 0;
    }
    left++;
}
return -1;
```
2. 使用KMP算法。
先构建模式串的next数组，然后使用双指针遍历主串和模式串。  
此时回溯时不需要回溯主串指针，只需根据next数组**回溯模式串指针**。


***459.*** 重复的子字符串
判断字符串是否由重复的子字符串构成。
1. 将字符串与自身拼接，去掉首尾字符后，判断原字符串是否在新字符串中出现。
2. 使用KMP算法，构建next数组。
```java
int n = s.length(); 
int[] next = new int[n];
for (int i = 1, j = 0; i < n; i++) {// i为后缀末尾，j为前缀末尾
    while(j > 0 && s.charAt(i) != s.charAt(j)) { j = next[j-1]; }
    if (s.charAt(i) == s.charAt(j)) { j++; }
    next[i] = j;
}
if (next[n-1] > 0 && n % (n - next[n-1]) == 0) { return true; }// 注意判断条件，next[n-1]>0
```

## 栈和队列

232. 用栈实现队列
使用两个栈实现，一个只用于输入，一个只用于输出。
```java
// 核心函数
// 输出栈为空时才将输入栈元素弹出压入输出栈
// 因此中途入队的元素不会影响出队顺序，每个元素都只翻转一次
private void dump() {
    if (stackout.isEmpty()) {
        while (!stackin.isEmpty()) {
            stackout.push(stackin.pop());
        }
    }
}
```

225. 用队列实现栈
将除了队列尾部的元素依次出队再入队，最后一个元素即为栈顶元素。

150. 逆波兰表达式求值
遇到数字入栈，遇到运算符出栈两个数字进行运算，结果入栈。

239. 滑动窗口最大值
```java
public int[] maxSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    if (n == 0 || k == 0) { return new int[0]; }
    int[] res = new int[n - k + 1];
    Deque<Integer> queue = new ArrayDeque<>();

    for (int i = 0; i < nums.length; i++) {
        while (!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]) {
            queue.pollLast();
        }// 出队所有比当前位置小的，因此队列中必为降序

        queue.offerLast(i);// 入队
        if (queue.peekFirst() <= i - k) {// 判断队首是否出窗口范围（用<=）
            queue.pollFirst();
        }

        if (i >= k - 1) {
            res[i - k + 1] = nums[queue.peekFirst()];
        }
    }
    return res;
}
```

347. 前K个高频元素
使用小顶堆维护前k个高频元素。
```java
public int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int num : nums) {
        map.put(num, map.getOrDefault(num, 0) + 1);
    }

    // 最小堆：按出现次数排序
    Queue<Map.Entry<Integer, Integer>> pq =
            new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());

    for (Map.Entry<Integer, Integer> entry : map.entrySet()) {// entrySet() 遍历键值对
        pq.offer(entry);
        if (pq.size() > k) {// 个数超过k，弹出优先级最高（最小）元素
            pq.poll();
        }
    }

    int[] result = new int[k];
    for (int i = 0; i < k; i++) {
        result[i] = pq.poll().getKey();
    }
    return result;
}
```

## 二叉树
### 基础知识
1. 特殊二叉树：满二叉树、完全二叉树、平衡二叉树、二叉搜索树、堆。
- 满二叉树：每层节点数都达到最大值，深度为k时有2^k-1个节点。
- 完全二叉树：除最后一层外都是满的，最后一层节点从左到右连续排列。
- 平衡二叉树：任意节点的左右子树高度差不超过1。
- 二叉搜索树：左子树所有节点值<根节点值<右子树所有节点值。
- 堆：完全二叉树，父节点值总是大于/小于子节点值（大顶堆/小顶堆）。

2. **遍历方式**：前序、中序、后序、层序。  
*其实就是处理左右子节点的顺序不同：*  
前序：根->左->右  
中序：左->根->右  
后序：左->右->根  
层序：按层从上到下、从左到右遍历。
```java
// 递归前序遍历，中、后序则result.add分别在第二、三行
void dfs(TreeNode root, List<Integer> list) {
    if (root == null) {
        return;
    }
    result.add(root.val);
    preorder(root.left, result);
    preorder(root.right, result);
}
// 层序遍历
// // 也可以是Queue<TreeNode> queue = new LinkedList<>(); Deque支持更多操作，性能更好
Deque<TreeNode> queue = new ArrayDeque<>();
queue.offer(root);
while (!queue.isEmpty()) {
    int size = queue.size();// size为当前层节点数，避免处理下一层节点
    List<Integer> level = new ArrayList<>();
    for (int i = 0; i < size; i++) {
        TreeNode node = queue.poll();
        level.add(node.val);
        if (node.left != null) { queue.offer(node.left); }
        if (node.right != null) { queue.offer(node.right); }
    }
    result.add(level);
}
return result;
```
（处理节点时会按序访问其左右子节点。由于代码会自动拦截空节点，因此适用于任意形态的二叉树，无需局限于满二叉树）

**通用遍历标记法：**
压入null表示已处理该节点的子节点，可以直接读取它的值。
```java
Stack<TreeNode> stack = new Stack<>();
stack.push(root);

while (!stack.isEmpty()) {
    TreeNode node = stack.pop();
    if (node != null) {
        // 示例：中序
        if (node.right != null) stack.push(node.right); // 右
        stack.push(node); // 中
        stack.push(null); // 标记位
        if (node.left != null) stack.push(node.left);   // 左
    } else {
        // 遇到null，说明下一个节点是需要被处理的根节点
        result.add(stack.pop().val);
    }
}
```

3. 存储方式：顺序存储、链式存储。
顺序存储即使用数组，适合完全二叉树。节点i的左子节点为2i+1，右子节点为2i+2。  
链式存储即使用节点类，适合各种二叉树。

### 题目
101. 对称二叉树
```java
public boolean isSymmetric(TreeNode root) {
    if (root == null) return true;
    return dfs(root.left, root.right);
}
private boolean dfs(TreeNode left, TreeNode right) {
    if (left == null && right == null) return true;
    if (left == null || right == null || left.val != right.val) return false;
    return dfs(left.left, right.right) && dfs(left.right, right.left);
}
```

110. 平衡二叉树
比较高度，则递归使用后序遍历。

111. 二叉树的最小深度
```java
// 层序遍历：当遇到第一个叶子节点时，当前层数即为最小深度。
while (!queue.isEmpty()){
    int size = queue.size();
    depth++;
    TreeNode cur = null;
    for (int i = 0; i < size; i++) {
        cur = queue.poll();
        if (cur.left == null && cur.right == null){ //直接返回最小深度
            return depth;
        }
        if (cur.left != null) queue.offer(cur.left);
        if (cur.right != null) queue.offer(cur.right);
    }
}
// 递归
public int minDepth(TreeNode root) {
    if (root == null) return 0;
    int m1 = minDepth(root.left);
    int m2 = minDepth(root.right);
    // 注意：如果有一个子树为空，判断语句返回0，会导致结果错误，需要特殊处理。
    // 如果有一个子树为空，返回 m1 + m2 + 1 （m1或m2有一个为0）
    // 如果都不为空，返回 min(m1, m2) + 1
    return (root.left == null || root.right == null) ? (m1 + m2 + 1) : Math.min(m1, m2) + 1;
}
```


***116.*** 填充每个节点的下一个右侧节点指针
给定一个完美二叉树，填充每个节点的 next 指针，使其指向下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
```java
// 1. 递归，不适用于任意形态的二叉树
public Node connect(Node root) {
    if (root == null) { return null; }
    Node leftmost = root;
    while (leftmost.left != null) {
        Node head = leftmost;
        while (head != null) {
            head.left.next = head.right;// 同一父节点的左右子节点连接
            if (head.next != null) {
                head.right.next = head.next.left;// 不同父节点的子节点连接
            }
            head = head.next;// 同一层的下一个父节点
        }
        leftmost = leftmost.left;// 下一层的最左节点
    }
    return root;
}
// 2. 层序遍历，都适用
public Node connect(Node root) {
    if (root == null) return null;
    Queue<Node> queue = new LinkedList<>();
    queue.offer(root);
    
    while (!queue.isEmpty()) {
        int size = queue.size();
        Node prev = null;
        
        for (int i = 0; i < size; i++) {
            Node node = queue.poll();
            if (prev != null) prev.next = node;
            prev = node;
            
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
    }
    return root;
}
```

## 动态规划
动态规划：**将复杂问题分解为更小子问题**。  
初始状态->子问题->最终状态 遵循相同的原则，因此可以递推。

核心要素：发现复杂问题可以由子问题递推。
1. 将什么状态定义为dp数组（注意初始化，一般求什么就定义什么）
2. 状态转移方程
3. 遍历顺序

Debug：打印dp数组

### 题目
***70.*** 爬楼梯
dp[i]： 爬到第i层楼梯，有dp[i]种方法
```java
dp[0] = 1;
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {// 一次能爬 1~m 步
        if (i - j >= 0) { dp[i] += dp[i - j]; }
    }
}
return dp[n];
```
状态转移方程为： dp[i] = sum(dp[i-j]) (1<=j<=m)
由于遍历时只用到前m项，可以用m个变量代替dp数组，降低空间复杂度。

***746.*** 使用最小花费爬楼梯
```java
int dpi = Math.min(dp0+cost[i-2], dp1+cost[i-1]);
dp0 = dp1;
dp1 = dpi;
```
为什么dp0，dp1的更新只跨一步？只和遍历顺序有关。
遍历顺序为i++，遍历顺序不是最终路径。因此跨越一步或者两步与遍历顺序无关。

***62.*** 不同路径
dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

***96.*** 不同的二叉搜索树
令 dp[i] 表示 i 个节点能构成的 BST 数量，dp[0]=1。
对于节点数为i（1<=i<=n），选择j(0 <= j <= i-1)为左子树节点数（根节点占一个），此时左子树由j个节点构成，右子树由i-j-1个节点构成。
因此子问题求解为 dp[i]=sum(dp[j] * dp[i-j-1])，即状态转移方程。

***198.*** 打家劫舍
```java
int n = nums.length;
int[] dp = new int[n + 1];
dp[1] = nums[0];
for (int i = 3; i <= n; i++) {
    dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
}
return dp[n];
```

#### 背包问题
状态定义：
- dp[j]：容量为 j 的背包，所能装下的最大价值
- dp[j - weight[i]]：容量为 j - weight[i] 的背包，所能装下的最大价值

状态转移：
- 01背包：每个物品选择一次或不选。  
- 完全背包：每个物品可选择多次。

具体类型：
- 最大价值：dp[j] = Math.max(dp[j], dp[j - cost] + value)
- 共有几种划分方法：dp[j] += dp[j - cost]
- 能否划分成功：dp[j] = dp[j] || dp[j - cost]
- 最多/最少用几个数字：dp[j] = Math.[max/min](dp[j], dp[j - cost] + 1)

**最大价值问题**
```java
// M 种研究材料，costs[i]，values[i];背包总空间为 N
int[] dp = new int[N + 1];
for (int i = 0; i < M; i++) {
    // 内层循环j--。同746题，遍历顺序!=最终路径 
    for (int j = N; j >= costs[i]; j--) {// 01背包倒序，完全背包正序
        dp[j] = Math.max(dp[j], dp[j - costs[i]] + values[i]);
        // 不选择or选择
    }
}
return dp[N];
```

**划分成功问题**
***416.*** 分割等和子集
[1,5,11,5] -> true  
背包容量 sum/2，物品为 nums 数组。
```java
if (sum % 2 != 0) { return false; }
int target = sum/2;
boolean[] dp = new boolean[target+1];
dp[0] = true;
for (int num : nums) {
    for (int j = target; j >= num; j--) {
        dp[j] = dp[j] || dp[j-num];
    }
}
return dp[target];
```

**多维背包：**
 ***474.*** 一和零
