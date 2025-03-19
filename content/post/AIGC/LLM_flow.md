+++
title = 'LLM_flow'
date = 2024-09-25T22:49:25+08:00
draft = false
description = 'LLM训练、部署全流程及问题解决、经验总结'
categories = ['AIGC']
+++

## 前置
### 服务器使用指南
选择GPU的几种方式
```python
# 设置可见GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"
#n为GPU编号，从0开始。需要在import torch前设置
import torch

device = torch.device("cuda:0")# 只能指定单卡

# 指定使用GPU 0, 1和2（不设置device_ids或令其=None，则默认使用所有GPU）
model = nn.DataParallel(model, device_ids=[0, 1, 2])
```
### pytorch安装
使用pip安装比conda更靠谱，指令见(<https://pytorch.org/get-started/locally/>)
cuda版本等细节见土堆教程。
(pip临时代理： --proxy=http://)

### 下载预训练模型
**注意：需要使用git lfs clone以下载模型参数文件。**
* git clone  
    git clone url --depth=1:只下载最近一次commit  
    (-c http.proxy=\"http://127.0.0.1:7890\" # 临时代理)  
    lfs下载大文件时，其文件大小会增加，但进度条百分比和网速会卡住，当下完完整的一个大文件后才更新进度，耐心等待即可。  

* huggingface下载：  

    1.(推荐)
    ```bash
    sudo apt install aria2
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod +x hfd.sh
    export HF_ENDPOINT=https://hf-mirror.com
    # 下载模型(下载速度慢可以取消下载后断点重连)
    ./hfd.sh <model_name> --tool aria2c -x 4 [可选]--hf_username <username> --hf_token <apikey>
    # 代理：--all-proxy=http://
    # 下载数据集
    ./hfd.sh <dataset_name> --dataset --tool aria2c -x 4 [可选]--hf_username <username> --hf_token <apikey>
    ```
    2.  
    ```bash
    pip install -U huggingface_hub
    # 设置环境变量为镜像源(建议写入/.bashrc)
    export HF_ENDPOINT=https://hf-mirror.com
    # 下载模型
    huggingface-cli download --resume-download <model_name> --local-dir <local_dir_name>
    # 下载数据集
    huggingface-cli download --repo-type dataset --resume-download <dataset_name> --local-dir <local_dir_name>
    ## 可以添加--local-dir-use-symlinks False禁用软链接，下载路径下所见即所得
    ```  
    3. 选择clone repository（三个点展开），使用提供的git clone命令下载到本地。  

* 魔搭社区下载：  
    参考见上。

## 推理(inference)
### 概述    
推理(inference)是指模型对输入数据进行预测，得到模型输出结果。  
推理过程可以分为三个步骤：  
1. 加载模型参数：加载模型参数，包括模型结构和模型参数。
2. 输入数据预处理：对输入数据进行预处理，如tokenizing、padding等。
3. 模型推理：使用模型进行推理，得到模型输出结果。

## 微调(finetune)
### 介绍
#### 序言
用好大模型的第一个层次，是掌握提示词工程(Prompt Engineering)，
用好大模型的第二个层次，是大模型的微调(Fine Tuning)

Prompt Engineering 的方式会把Prompt搞得很长
微调通过自有数据，优化模型在特定任务上的性能，减少幻觉。  

#### 技术路线
从参数规模的角度，大模型的微调分成两条技术路线：
* 全量微调FFT(Full Fine Tuning)：对全量的参数，进行全量的训练。
* 参数高效微调PEFT(Parameter-Efficient Fine Tuning)：只对部分的参数进行训练，如Lora。

从训练的方法的角度
* 监督式微调SFT(Supervised Fine Tuning)，主要是用人工标注的数据，用传统机器学习中监督学习的方法，对大模型进行微调；
* 基于人类反馈的强化学习微调RLHF(Reinforcement Learning with Human Feedback)，这个方案的主要特点是把人类的反馈，通过强化学习的方式，引入到对大模型的微调中去，让大模型生成的结果，更加符合人类的一些期望；
* 基于AI反馈的强化学习微调RLAIF(Reinforcement Learning with AI Feedback)，这个原理大致跟RLHF类似，但是反馈的来源是AI。

### 微调流程
[Lora参考流程](<https://www.bilibili.com/video/BV1dr421w7J5>)
#### 数据集
instruction字段通常用于描述任务类型或给出指令  
input字段包含模型需要处理的文本数据  
output字段则包含对应输入的正确答案或期望输出  

>常用中文微调数据集可能包括：
    中文问答数据集（如CMRC 2018、DRCD等），用于训练问答系统。
    中文情感分析数据集（如ChnSentiCorp、Fudan News等），用于训练情感分类模型。
    中文文本相似度数据集（如LCQMC、BQ Corpus等），用于训练句子对匹配和相似度判断任务。
    中文摘要生成数据集（如LCSTS、NLPCC等），用于训练文本摘要生成模型。
    中文对话数据集（如LCCC、ECDT等），用于训练聊天机器人或对话系统。

#### 训练过程

#### 评估与迭代

#### Lora训练示例
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

## 量化(quantization)
降低精度，减少模型大小，提升推理速度。

## 蒸馏(distillation)
让小模型学习大模型的知识，提升小模型的性能。


## 部署(deployment)
### vllm部署

## 评估(evaluation)