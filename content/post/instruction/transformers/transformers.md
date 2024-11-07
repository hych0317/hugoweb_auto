+++
title = 'Transformers'
date = 2024-10-26T15:34:07+08:00
draft = false
categories = ['instruction']
+++


## 基础部件
基本流程：
![steps](post/transformers/step.png)
### tokenizer
```python
from transformers import AutoTokenizer

# 加载
tokenizer = Autotokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese",trust_remote_code=True)# 从hf加载
tokenizer.save_pretrained("./my_tokenizer")# 保存到本地
tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")# 从本地加载

# 分词
tokens = tokenizer.tokenize("你好，欢迎使用！")

tokenizer.vocab# 查看词表
tokernizer.vocab_size# 查看词表大小

ids0 = tokenizer.convert_tokens_to_ids(tokens)# 转换成id
# 可通过tokenizer.convert_ids_to_tokens(ids)转换回来，convert_tokens_to_string(tokens)转换回句子

ids1 = tokenizer.encode("你好，欢迎使用！",add_special_tokens=False)# 一步到位
str1 = tokenizer.decode(ids1)# 解码

# 填充和截断，以适应batch长度
input_ids = tokenizer.encode("你好，欢迎使用！",max_length=10,padding="max_length",truncation=True)

```

### model
```python
from transformers import AutoModel

# 加载
model = AutoModel.from_pretrained("uer/roberta-base-finetuned-dianping-chinese",trust_remote_code=True)# 从hf加载
model = AutoModel.from_pretrained("./model_name",output_attentions=True)# 从本地加载

# 输出
inputs = tokenizer("你好，欢迎使用！",return_tensors="pt")
outputs = model(**inputs)
outputs.last_hidden_state()# 输出最后一层隐藏层

# 指定model head
from transformers import AutoModelForSequenceClassification
cls_model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese",num_labels=3,output_attentions=True)# 指定三分类
# 对基本模型的输出进行任务处理
output_cls = cls_model(**inputs)
```
### dataset
```python
# 加载数据集
dataset = load_dataset("dataset_name","subtask_name", split="train[10:100]")# (数据集名,[可选]子任务名(有的话),[可选]切片)
# 查看
dataset["train"][:2]
# 划分
dataset.train_test_split(test_size=0.2,stratify_by_column="label")# 按比例划分,label分布均衡
# 数据选取与过滤
datasets["train"].select([1, 5])# 选取第2和第6条数据,返回的类型仍是dataset
filter_dataset = datasets["train"].filter(lambda example: "中国" in example["title"])
# 数据映射
processed_datasets = datasets.map(preprocess_function,batched=True,remove_columns=["text"])# 映射函数,<可选>使用batch处理,去除text列
# 本地保存与加载
processed_datasets.save_to_disk("./processed_data")
processed_datasets = load_from_disk("./processed_data")
```

### 实例
```python
import torch
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)

# 加载数据集
dataset = MyDataset()

# 划分数据集
from torch.utils.data import random_split
trainset, validset = random_split(dataset, lengths=[0.9, 0.1])

# 定义dataloader
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)# shuffle=True表示每个epoch打乱顺序
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)

# optimizer
from torch.optim import Adam

model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
if torch.cuda.is_available():
    model = model.cuda()

optimizer = Adam(model.parameters(), lr=2e-5)

# 训练/评估
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)

def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:# 训练集
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
                # 把batch的key和value都转到cuda上
            optimizer.zero_grad()# 清空梯度
            output = model(**batch)
            output.loss.backward()# 反向传播
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}")

train()

sen = "我觉得这家酒店不错，饭很好吃！"
id2_label = {0: "差评！", 1: "好评！"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sen, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    print(f"输入：{sen}\n模型预测结果:{id2_label.get(pred.item())}")
```

### 使用trainer优化实例
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
# 加载/划分数据集
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets = dataset.train_test_split(test_size=0.1)

# 处理数据集
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)

# 模型\评估
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

import evaluate

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

# 创建training_arguments
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=64,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=10,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型

# 创建trainer
from transformers import DataCollatorWithPadding
trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

# 训练/评估
trainer.train()
trainer.evaluate(tokenized_datasets["test"])
trainer.predict(tokenized_datasets["test"])

```

## NLP任务

### 命名实体识别(NER)
NER是指识别文本中的实体，如人名、地名、机构名等。  
通常，NER任务包括两部分:
- 实体识别: 识别出文本中的实体，并给予其相应的标签。
- 实体分类: 将识别出的实体进行分类，如人名、地名、机构名等。

##  微调


## 低精度训练

## 分布式训练