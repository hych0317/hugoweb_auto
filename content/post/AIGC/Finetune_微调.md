
## 概述
### 序言
用好大模型的第一个层次，是掌握提示词工程(Prompt Engineering)，
用好大模型的第二个层次，是大模型的微调(Fine Tuning)

Prompt Engineering 的方式会把Prompt搞得很长
微调通过自有数据，优化模型在特定任务上的性能，减少幻觉。  

### 技术路线
从参数规模的角度，大模型的微调分成两条技术路线：
* 全量微调FFT(Full Fine Tuning)：对全量的参数，进行全量的训练。
* 参数高效微调PEFT(Parameter-Efficient Fine Tuning)：只对部分的参数进行训练，如Lora。

从训练的方法的角度
* 监督式微调SFT(Supervised Fine Tuning)，主要是用人工标注的数据，用传统机器学习中监督学习的方法，对大模型进行微调；
* 基于人类反馈的强化学习微调RLHF(Reinforcement Learning with Human Feedback)，这个方案的主要特点是把人类的反馈，通过强化学习的方式，引入到对大模型的微调中去，让大模型生成的结果，更加符合人类的一些期望；
* 基于AI反馈的强化学习微调RLAIF(Reinforcement Learning with AI Feedback)，这个原理大致跟RLHF类似，但是反馈的来源是AI。

## 微调流程
[Lora参考流程](<https://www.bilibili.com/video/BV1dr421w7J5>)
### 数据集
instruction字段通常用于描述任务类型或给出指令  
input字段包含模型需要处理的文本数据  
output字段则包含对应输入的正确答案或期望输出  

>常用中文微调数据集可能包括：
    中文问答数据集（如CMRC 2018、DRCD等），用于训练问答系统。
    中文情感分析数据集（如ChnSentiCorp、Fudan News等），用于训练情感分类模型。
    中文文本相似度数据集（如LCQMC、BQ Corpus等），用于训练句子对匹配和相似度判断任务。
    中文摘要生成数据集（如LCSTS、NLPCC等），用于训练文本摘要生成模型。
    中文对话数据集（如LCCC、ECDT等），用于训练聊天机器人或对话系统。

### 训练过程

### 评估与迭代

### Lora训练示例
```python
import torch ​
import torch.nn as nn​
import torch.nn.functional as F​
import math​
​
class LoRALinear(nn.Module):​
    def __init__(self, in_features, out_features, merge, rank=16, lora_alpha=16, dropout=0.5):​
        super(LoRALinear, self).__init__()​
        self.in_features = in_features​
        self.out_features = out_features​
        self.merge = merge​
        self.rank = rank​
        self.dropout_rate = dropout​
        self.lora_alpha = lora_alpha​
        ​
        self.linear = nn.Linear(in_features, out_features)​
        if rank > 0:​
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))​
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))​
            self.scale = self.lora_alpha / self.rank​
            self.linear.weight.requires_grad = False​
        ​
        if self.dropout_rate > 0:​
            self.dropout = nn.Dropout(self.dropout_rate)​
        else:​
            self.dropout = nn.Identity()​
        ​
        self.initial_weights()​
    ​
    def initial_weights(self):​
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))​
        nn.init.zeros_(self.lora_b)​
        ​
    def forward(self, x):​
        if self.rank > 0 and self.merge:​
            output = F.linear(x, self.linear.weight + self.lora_b @ self.lora_a * self.scale, self.linear.bias)​
            output = self.dropout(output)​
            return output​
        else:​
            return self.dropout(self.linear(x))​
​
```