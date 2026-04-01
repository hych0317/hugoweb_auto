+++
title = 'Algorithm'
date = 2025-08-24T21:01:10+08:00
draft = false
categories = ["notes"]
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

***581.*** 最短无序连续子数组
双指针：从两个方向遍历，找出中间无序数组的左右边界。
```java
// 示例1 2 6 5 3 4 7 8
int max = Integer.MIN_VALUE;
int min = Integer.MAX_VALUE;
for (int i = 0; i < n; i++) {
    
    // 从左向右，找中间子数组的右边界
    if (nums[i] >= max) {// 对于升序数组，当前元素应大于之前的max
        max = nums[i];// 正常更新max
    } else {// 比当前max小的（说明非升序）都在中子数组范围内
        end = i;
    }
    // 从右向左，找左边界
    if (nums[n - 1 - i] <= min) {// 正常更新min
        min = nums[n - 1 - i];
    } else {
        start = n - 1 - i;
    }
}
```

***76.*** 最小覆盖子串
滑动窗口：先统计t中各字符的频次，根据频次特征移动s的窗口
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
```java
public String minWindow(String s, String t) {
    if (s == null || t == null || s.length() < t.length()) {
        return "";
    }

    int[] need = new int[128];
    for (char c : t.toCharArray()) {need[c]++;}
    // 统计t中各字符的频次

    int left = 0, right = 0;
    int count = t.length(); // t总字符数
    int start = 0, minLen = Integer.MAX_VALUE;

    while (right < s.length()) {
        char c = s.charAt(right);

        if (need[c] > 0) {count--;}// 是t中的字符，计数--
        need[c]--;// 对每个字符都执行，
        // 因此即使下面d是无关的字符时，其need的值也不会大于0
        right++;

        while (count == 0) {// 满足条件，收缩窗口
            if (right - left < minLen) {// 先记录开始位置和长度
                minLen = right-left;
                start = left;
            }
            // 收缩窗口
            char d = s.charAt(left);
            if (need[d] == 0) {count++;}// 刚好用到的字符，移出窗口则count需要++
            need[d]++;
            left++;
        }
    }

    return minLen == Integer.MAX_VALUE ? "" : s.substring(start, start+minLen);
}
```

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

234. 回文链表
使用快慢指针找到链表中点，原地反转后半部分链表（保证空间O(1)），然后比较前半部分和反转后的后半部分。

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
前序：根->左->右  **约等于对应自顶向下**
中序：左->根->右  
后序：左->右->根  **约等于对应自底向上**
层序：按层从上到下、从左到右遍历。

![二叉树遍历](post/programming/notes/assets/btTraversal.png)

从整体视角看，**dfs函数返回当前节点的左右子树**（子树内部也按顺序排列）
可以看到前序遍历的分布为根，左子树，右子树
```java
递归中序遍历，前、后序即改变result.add位置
void dfs(TreeNode root, List<Integer> list) {
    if (root == null) {
        return;
    }
    
    dfs(root.left, result);// 左子树
    result.add(root.val);// 代码处理部分
    dfs(root.right, result);// 右子树
}
层序遍历
// 也可以是Queue<TreeNode> queue = new LinkedList<>(); Deque支持更多操作，性能更好
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

### 递归问题
从局部到全局，**每个节点的处理都遵循相同的原则**，因此可以递归。  

构建递归函数：  
1. **信息传递**：需要谁的？给谁传递？
    自底向上（110、236题）：需要子节点信息，给父节点返回信息。虽然节点是从上到下访问的，但信息传递方向从下到上。
    自顶向下（98、257题）：需要父节点信息，给子节点传递信息。
2. **递归出口**：
节点为 null 或 叶子节点 时，做特殊处理。


### 题目

#### 深度与路径类

***引子：513. 找树左下角的值***
方法1：递归
构建函数思路：根据信息判断

- 二叉树节点的信息：a.左右子节点，b.节点值。  
需要的信息：c.当前节点所在深度。  
- 函数需要完成的：
    提供递归出口
    进行处理（b,c）
    访问左右子节点（a）
```java
private int maxDepth = 0;
private int leftmostValue;
public int findBottomLeftValue(TreeNode root) {
    dfs(root, 1);
    return leftmostValue;
}
private void dfs(TreeNode node, int depth) {
    if (node == null) { return; }
    if (depth > maxDepth) {
        maxDepth = depth;
        leftmostValue = node.val;
    }
    dfs(node.left, depth + 1);// 先访问左子节点，该函数会更新maxDepth，因此访问同一层右子节点时不会覆盖结果。
    dfs(node.right, depth + 1);
}


方法2：层序遍历
调换左右节点入队顺序，每层最后一个被弹出的节点为最左节点，就无需额外的判断了。

public int findBottomLeftValue(TreeNode root) {
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
        root = queue.poll();
        if (root.right != null) queue.offer(root.right);
        if (root.left != null) queue.offer(root.left);
    }
    return root.val; // 最后一个被弹出的就是最左下角的
}
```

111. 二叉树的最小深度
最大深度只需要看层序遍历的总层数，而最小深度需要在遍历过程中判断是否为叶子节点。  
因为层序遍历自上而下、从左到右，因此当遇到第一个叶子节点时，当前层数即为最小深度。  

第112题.路径总和 的层序遍历解法采用了两个队列同步出入队列，一个存储节点，一个存储当前路径和。不具体展开了。
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

257. 二叉树的所有路径
子节点需要父节点信息（路径），因此自顶向下递归。
```java
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        constructPaths(root,"",res);
        return res;
    }

    private void constructPaths(TreeNode node, String path, List<String> res) {
        if (node!=null) {
            path += Integer.toString(node.val);
            if (node.left == null && node.right == null) {
                res.add(path);
            } else {
                path+="->";
                constructPaths(node.left,path,res);
                constructPaths(node.right,path,res);
            }
        }
    }
}
```



#### 树的属性判断类

***98. 验证二叉搜索树***
自顶向下传递父节点范围，判断当前节点是否在区间内。
```java
public boolean isValidBST(TreeNode root) {
    return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);// 注意这里Long大写L
}

private boolean isValidBST(TreeNode node, long min, long max) {
    if (node == null) return true;
    if (node.val <= min || node.val >= max) return false;
    return isValidBST(node.left, min, node.val) && isValidBST(node.right, node.val, max);
}
```

***101. 对称二叉树***
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
父节点需要子节点信息（高度），因此自底向上递归。
```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return height(root)!=-1;
    }

    private int height(TreeNode node) {
        if (node == null) {return 0;}// 递归出口
        
        // 获取左右子树高度并剪枝
        int leftHeight = height(node.left);
        if (leftHeight==-1) {return -1;}// 剪枝
        
        int rightHeight = height(node.right);
        if (rightHeight==-1) {return -1;}// 剪枝
        
        // 对信息进行处理
        if (Math.abs(leftHeight-rightHeight) > 1) {return -1;}
        
        // 返回当前节点高度
        return Math.max(leftHeight,rightHeight)+1;
    }
}
```

#### 节点处理类

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

236. 二叉树的最近公共祖先
传递的信息：当前节点及其子树**是否包含p或q**，自底向上  
代码的设计：直接用null表示不包含；直接将节点作为返回值，不需要额外的变量记录答案。  
对于四种情况的讨论：
1.均为空：不包含，返回null
2.一个空一个不空：两种情况，非空的那个节点要么表示包含了p或q；要么此时非空的那个已经在公共祖先节点上方，其指向的已经是公共祖先节点，回传即可
3.两个都非空：说明当前root即是公共祖先节点，回传
```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) {return root;}

    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    
    // 对于left/right的四种情况判断（有有、有无、无有、无无）
    if(left == null) return right;
    if(right == null) return left;
    return root;// 两个都非空
}
```
968. 监控二叉树
题目描述：给定一个二叉树，我们在树的节点上安装摄像头，摄像头可以监视其父节点、自身和直接子节点。计算监控整棵树所需的最少摄像头数量。  
解法：**后序遍历（自底向上）**，根据左右子节点的状态判断自身状态，贪心决定是否放置摄像头。
```java
int ans=0;

public int minCameraCover(TreeNode root) {
    if (dfs(root) == 0) {ans++;}
    return ans;
}

// 定义三个状态：0：该节点未被覆盖
// 1：安装了摄像头；2：已被覆盖，但没摄像头
private int dfs(TreeNode node) {
    if (node == null) {return 2;}

    int left =dfs(node.left);
    int right = dfs(node.right);

    if (left == 0 || right == 0) {// 有一个没有就需要安装
        ans++;
        return 1;
    }
    if (left == 1 || right == 1) {return 2;}// 有子节点装了，被覆盖到

    return 0;
}
```


404. 左叶子之和
注意到叶子节点本身并不能判断其是否为左叶子节点。
```java
// 父级预判：通过root.left判断是左叶子。
public int sumOfLeftLeaves(TreeNode root) {
    if (root == null) return 0;
    int sum = 0;
    if (root.left != null && root.left.left == null && root.left.right == null) {
        sum += root.left.val;
    }
    return sum + sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
}

// 叶子节点作为递归出口：添加标记
public int sumOfLeftLeaves(TreeNode root) {
    return dfs(root, false); // 根节点不是任何人的左孩子
}

private int dfs(TreeNode node, boolean isLeft) {
    if (node == null) return 0;
    
    if (node.left == null && node.right == null) {
        return isLeft ? node.val : 0;
    }
    // 左true 右false
    return dfs(node.left, true) + dfs(node.right, false);
}
```



#### 区间切分构造类
654. 最大二叉树
```java
public TreeNode constructMaximumBinaryTree(int[] nums) {
    return build(nums,0,nums.length-1);
}

private TreeNode build(int[] nums, int left, int right) {
    if (left>right) {return null;}// 注意等于号

    int maxIndex = left;
    for (int i = left+1; i <= right; i++) {
        if (nums[i] > nums[maxIndex]) {
            maxIndex = i;
        }
    }

    TreeNode root = new TreeNode(nums[maxIndex]);
    // 注意向下传递的左右边界
    root.left = build(nums,left,maxIndex-1);
    root.right = build(nums,maxIndex+1,right);

    return root;
}
```

***⭐106. 从中序与后序遍历序列构造二叉树***
后序遍历顺序是 左 → 右 → 根，倒着读就是 根 → 右 → 左  
从后序序列获取第一个根节点->在中序序列中找到，切分左右子树->从后序序列找到左右子树的根节点  

前序序列也是同理

注意postIndex的全局/局部问题：  
- 如果为全局指针，，则每次递归--；
- 如果为局部变量，则需要记录子树长度，根据区间长度计算。
```java
private Map<Integer, Integer> indexMap;
private int[] postorder;
private int postIndex;

public TreeNode buildTree(int[] inorder, int[] postorder) {
    this.postorder = postorder;
    this.postIndex = postorder.length - 1;// 从最后开始读取
    this.indexMap = new HashMap<>();
    for (int i = 0; i < inorder.length; i++) {
        indexMap.put(inorder[i], i);
    }
    return build(0, inorder.length - 1);
}
private TreeNode build(int inLeft, int inRight) {
    if (inLeft > inRight) {return null;}

    int rootVal = postorder[postIndex--];
    TreeNode root = new TreeNode(rootVal);
    int rootIndex = indexMap.get(rootVal);
    // 全局指针，根右左遍历的顺序不可以换
    root.right = build(rootIndex + 1, inRight);
    root.left = build(inLeft, rootIndex - 1);

    return root;
}

// 前序数组的局部index写法
private TreeNode build(int preIndex, int begin, int end) {
    if (begin>end) return null;

    int rootVal = preorder[preIndex];
    int inIndex = indexMap.get(rootVal);

    TreeNode root = new TreeNode(rootVal);
    int leftLen = inIndex - begin;

    root.left  = build(preIndex+1, begin, inIndex-1);
    root.right = build(preIndex+leftLen+1, inIndex+1, end);

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

## 贪心算法
134. 加油站
如果从加油站A出发最远只能到达加油站B，那么A B之间的任何一个加油站作为起点，都不可能越过B点。  
所以下一个出发点直接选为(B+1)，使得只需要遍历一次数组。  
（题目规定存在解是唯一的，因此只要totalOil证明存在后，新的index无需循环遍历数组证明存在）
```java
public int canCompleteCircuit(int[] gas, int[] cost) {
    int totalOil = 0;
    int index = 0;
    int currOil = 0;
    for (int i = 0; i < gas.length; i++) {
        int diff = gas[i] - cost[i];
        totalOil += diff;
        currOil += diff;
        if (currOil < 0) {
            index = i+1;
            currOil = 0;
        }
    }
    return totalOil < 0 ? -1 : index;
}
```

45. 跳跃游戏 II
返回到达 n - 1 的最小跳跃次数。
注意，nums数组的值表示在该位置可以跳跃的最大距离，可以跳到该距离内任意一个点。  
```java
public int jump(int[] nums) {
    int cnt = 0;
    int max = 0;
    int end = 0;
    for (int i = 0; i < nums.length-1; i++) {
        max = Math.max(max, i+nums[i]);
        // 当前一跳覆盖范围是i~end之间，范围内的每个点都可以作为下一跳的出发点
        // 不关心具体在哪个点起跳，只关心最远范围，更新max
        if (i==end) {
            cnt++;
            end = max;
        }
    }
    return cnt;
}
```

738. 单调递增的数字
如果不递增，则将该位开始的低位设为9并借位减一。
从后往前遍历。
```java
public int monotoneIncreasingDigits(int n) {
    String s = Integer.toString(n);
    char[] chars = s.toCharArray();
    
    // start 记录从哪一位开始全部变为 9
    int start = chars.length; 
    
    for (int i = chars.length-1; i > 0; i--) {
        if (chars[i-1] > chars[i]) {
            chars[i-1]--;
            start = i;
        }
    }
    
    for (int i = start; i < chars.length; i++) {
        chars[i] = '9';
    }
    
    return Integer.valueOf(String.valueOf(chars));
}
```

### 两个维度
***⭐135. 分发糖果***
问题有两个维度的约束，既要和左边比又要和右边比。  
分别从左右两个方向遍历。
```java
public int candy(int[] ratings) {
    int n = ratings.length;
    int[] candyVec = new int[n];
    Arrays.fill(candyVec, 1);

    // 从前向后遍历
    for (int i = 1; i < ratings.length; i++) {
        if (ratings[i] > ratings[i-1]) {
            candyVec[i] = candyVec[i-1] + 1;
        }
    }

    // 从后向前遍历
    for (int i = ratings.length-1; i >= 1; i--) {
        if (ratings[i-1] > ratings[i]) {
            candyVec[i-1] = Math.max(candyVec[i-1], candyVec[i]+1);
        }
        
    }

    int result = 0;
    for (int s : candyVec) {
        result += s;
    }
    return result;
}

// 如果candy数组不赋初始值1，则如下表示
for (int i = 0; i < n; i++) {// 前向后遍历
    if (i > 0 && ratings[i] > ratings[i - 1]) {
        candyVec[i] = candyVec[i - 1] + 1;
    } else {candyVec[i] = 1;}
}
```

406. 根据身高重建队列
先按身高降序排序，再按k值直接插入（小个子插入不影响高个子）
注意：比较器返回负数则升序，返回正数则降序。通过改变比较逻辑实现不同排序。
```java
public int[][] reconstructQueue(int[][] people) {
    // 身高从大到小排（身高相同，k小的站前面）
    Arrays.sort(people, (i,j) -> {
        if (i[0] == j[0]) return i[1] - j[1];// k小在前
        return j[0] - i[0];// h小在后
    });

    ArrayList<int[]> que = new ArrayList<>();

    for (int[] p : people) {que.add(p[1], p);}
    // 因为降序排序，因此k值就是p应该在的索引位置

    return que.toArray(new int[que.size()][]);
}
```

### 区间调度
***435. 无重叠区间***
返回需要移除的重叠区间的最小数量
```java
public int eraseOverlapIntervals(int[][] intervals) {
    if (intervals.length == 0) return 0;
    // 按区间的结束位置升序排序
    Arrays.sort(intervals, (a,b) -> {return a[1] - b[1];});

    int count = 1; // 记录非重叠区间的数量，初始为1
    int edge = intervals[0][1]; // 记录当前非重叠区间的右边界

    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] >= edge) {// 注意此处，大于右边界
            count++;
            edge = intervals[i][1];
        }
    }

    return intervals.length - count;
}
```

56. 合并区间
合并所有重叠的区间
```java
public int[][] merge(int[][] intervals) {
    if (intervals.length <= 1) {return intervals;}

    Arrays.sort(intervals, (a, b) -> a[0] - b[0]);// 起点升序排序

    List<int[]> res = new ArrayList<>();
    int[] curr = intervals[0];
    res.add(curr);

    for (int[] interval : intervals) {
        int currEnd = curr[1];
        int nextStart = interval[0];
        int nextEnd = interval[1];

        if (currEnd >= nextStart) {
            curr[1] = Math.max(curr[1], nextEnd);// res中存的是curr地址，会同步更新
        } else {
            curr = interval;
            res.add(curr);
        }
    }

    return res.toArray(new int[res.size()][]);
}
```

## 单调栈/队列
适用于需要**找到第一个满足条件的元素**的题目。 

可以理解为排队，每次遇到新元素，while比较栈顶元素与新元素的关系？满足则出队：不满足继续排队。
**新元素第一次肯定入队**。因为比新元素小/大的元素都会被弹出。

单调栈的本质是空间换时间，在一次遍历中维护一个单调递增/递减的栈。

### 题目
***42. 接雨水***
当前元素作为*坑底*，找到左右第一个比它更高的元素，计算雨水量。
```java
// 双指针法：从两端向中间遍历，更新左右最大高度，计算雨水量。
public int trap(int[] height) {
    int left = 0, right = height.length - 1;// 双指针
    int leftMax = 0, rightMax = 0;
    int ans = 0;

    while (left<right) {
        if (height[left]<height[right]) {// 短板效应，更新较矮的指针
            if (height[left] >= leftMax) {leftMax=height[left];} 
            else {ans += leftMax-height[left];}
            left++;
        } else {
            if (height[right] >= rightMax) {rightMax = height[right];} 
            else {ans += rightMax-height[right];}
            right--;
        }
    }
    return ans;
}
// 单调栈法：遍历数组时，单调栈会找到当前元素作为*坑底*，左右第一个比它更高的元素，计算雨水量。
// 完备性讨论：算法遍历了每个元素，找到其对应的最大左右边界，因此算法是完备的。
public int trap(int[] height) {
    int ans = 0;
    Deque<Integer> stack = new ArrayDeque<>();
    for (int i = 0; i < height.length; i++) {
        while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
            int top = stack.pop();
            if (stack.isEmpty()) { break; }
            int left = stack.peek();
            int width = i - left - 1;
            int boundedHeight = Math.min(height[i], height[left]) - height[top];
            ans += width * boundedHeight;
        }
        stack.push(i);
    }
    return ans;
}
```

***⭐84. 柱状图中最大的矩形***
当前元素作为*矩形高度*。

接雨水找到两侧第一个比当前柱子更高的柱子，计算面积；本题找到两侧第一个比当前柱子更矮的柱子，计算面积。
```java
public int largestRectangleArea(int[] heights) {
    Stack<Integer> stack = new Stack<>();
    int maxArea = 0;
    for (int i = 0; i <= heights.length; i++) {
        int currentHeight = (i == heights.length) ? 0 : heights[i];// 最后多加一个0，保证所有柱子都能出栈
        while (!stack.isEmpty() && currentHeight < heights[stack.peek()]) {
            int height = heights[stack.pop()];
            int width = stack.isEmpty() ? i : i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }
        stack.push(i);
    }
    return maxArea;
}
// 双指针法：预处理每个柱子左右第一个比它更矮的柱子索引，计算面积(minRightIndex[i]-minLeftIndex[i]-1)*heights[i]。
```
完备性讨论：
- 怀疑：对于1 4 2 3中的4来说，其左右均边没有更高的柱子，面积为4。但其组成的最大矩形面积应为4 2 3组成的6。 
4 2 3组成的6实际是以2为高求得的矩形面积，而不是以4为高。  

这并不代表算法不完备，因为算法的目标是遍历每个柱子为高的最大矩形面积，之后比较出全局最大面积。而不是直接找到全局最大矩形面积。
- 证明：算法**遍历了以每根柱子i**为高的矩形面积，并通过单调栈找到每根柱子对应的最大宽度。

***85. 最大矩形***
通过遍历每一行，将矩阵问题转化为上题的直方图问题。
```java
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        if (matrix[i][j] == '1') {
            heights[j] += 1;
        } else {
            heights[j] = 0;
        }
    }
    maxArea = Math.max(maxArea, maxRecArea(heights));
}
// 处理函数
private int maxRecArea(int[] heights) {
    int n = heights.length;
    Stack<Integer> stack = new Stack<>();
    int maxArea = 0;
    int[] h = new int[n + 2];
    System.arraycopy(heights, 0, h, 1, n);
    
    for (int i = 0; i < h.length; i++) {
        while (!stack.isEmpty() && h[i] < h[stack.peek()]) {
            int height = h[stack.pop()];
            int width = i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }
        stack.push(i);
    }
    return maxArea;
}

```


***739. 每日温度***
ans[i]求第一个比第i天温度高的日期距离i几天。
解题需要获取某天温度和更高温度的索引差，因此单调栈存放索引。
```java
for (int i = 0; i < n; i++) {
    int currentTemp = temperatures[i];      
    // 当栈不为空，且当前温度大于栈顶索引对应的温度时
    while (!stack.isEmpty() && currentTemp>temperatures[stack.peek()]) {
        ans[stack.pop()] = i-prevIndex;
    }
    stack.push(i);
}
```

***496. 下一个更大元素 I***
找出 nums1 中每个元素在 nums2 中的下一个更大元素，没有则返回-1。
结果要求元素的值，因此单调栈存放元素值。
```java
public int[] nextGreaterElement(int[] nums1, int[] nums2) {
    // Map 用于存储 nums2 中每个元素及其对应的下一个更大元素
    Map<Integer, Integer> map = new HashMap<>();
    Deque<Integer> stack = new ArrayDeque<>();

    // 遍历 nums2 构建单调栈
    for (int num : nums2) {
        while (!stack.isEmpty() && num>stack.peek()) {
            map.put(stack.pop(),num);
        }
        stack.push(num);
    }

    int[] res = new int[nums1.length];
    for (int i = 0; i < nums1.length; i++) {
        res[i] = map.getOrDefault(nums1[i],-1);// 没有更大的元素，返回-1
    }

    return res;
}
```

***503. 下一个更大元素 II***
注意要循环数组，注意三点：  
1.需要遍历两遍数组for (int i = 0; i < 2*n; i++)
2.栈内存放索引时需要对n取模stack.push(i%n)
3.只有i < n时才入栈，保证栈内元素索引不重复
```java
for (int i = 0; i < 2*n; i++) {
    int num = nums[i%n];
    // 单调栈逻辑：如果当前元素大于栈顶索引对应的元素
    while (!stack.isEmpty() && num>nums[stack.peek()]) {
        res[stack.pop()] = num;
    }
    // 只需要在第一轮遍历时将索引入栈
    if (i<n) stack.push(i);
}
```