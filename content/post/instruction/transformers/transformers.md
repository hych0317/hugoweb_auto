+++
title = 'Transformers'
date = 2024-10-26T15:34:07+08:00
draft = false
categories = ['指令语法']
+++


## 基础部件
基本流程：
![steps](post/transformers/steps.png)
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

## NLP任务实操

### 命名实体识别(NER)
NER是指识别文本中的实体，如人名、地名、机构名等。  
通常，NER任务包括两部分:
- 实体识别: 识别出文本中的实体，并给予其相应的标签。
- 实体分类: 将识别出的实体进行分类，如人名、地名、机构名等。

##  PEFT微调
在创建模型后设置tuning_config,随后model = get_peft_model(model, config)  

>常见高效微调方法综述见arXiv:2303.15647

### Prompt tuning
```python
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
# Hard Prompt
config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                            prompt_tuning_init=PromptTuningInit.TEXT,
                            prompt_tuning_init_text="下面是一段人与机器人的对话。",
                            num_virtual_tokens=len(tokenizer("下面是一段人与机器人的对话。")["input_ids"]),
                            tokenizer_name_or_path="Langboat/bloom-1b4-zh")
model = get_peft_model(model, config)

# 进行训练...

# 加载训练完的模型
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")# 原模型
peft_model = PeftModel.from_pretrained(model=model, model_id="./output/checkpoint-500/")
```

### P-tuning/Prefix tuning
P-tuning把prompt加在输入embedding层的前缀，而Prefix tuning将kv值作为前缀加在模型的每一层前，而不仅仅是输入层。  
![prefix tuning](post/instruction/transformers/prefix.png)

原理(类似kv缓存的思想):
![prefix tuning](post/instruction/transformers/prefix_kvcache.png)
因为对于扩展后的KV矩阵，Qm\*n,K(m+x)\*n,V(m+x)\*n而言,Q·KT得m\*(m+k)维矩阵，再乘V得m\*n维矩阵，和原矩阵相乘维度一样。  
```python
from peft import PrefixTuningConfig, get_peft_model, TaskType
config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10, prefix_projection=True)
# prefix_projection默认值为false，表示使用P-Tuning v2， 如果为true，则表示使用 Prefix Tuning
# 其余流程一致
```

### Lora
通过矩阵分解的方式，将原始权重分解为低秩矩阵，计算时仅优化低秩矩阵，最后把低秩矩阵相乘加到原始权重上作为微调结果。 

```python
from peft import LoraConfig, TaskType, get_peft_model

# 查看target_modules参数要分解的权重层,该参数课传入列表如:
# ["word_embeddings", "encoder.layer.0.attention.self.query", "encoder.layer.0.attention.self.key", "encoder.layer.0.attention.self.value"]
# 也可以传入正则表达式如下
for name, parameter in model.named_parameters():
    print(name)

config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*\.1.*query_key_value", modules_to_save=["word_embeddings"])# modules_to_save表示其它要参与训练的权重层

model = get_peft_model(model, config)

# 进行训练...

# 加载训练完的模型
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")

peft_model = PeftModel.from_pretrained(model=model, model_id="./output/checkpoint-500/")
# 合并模型
# peft_model和merge_model的权重相同，p的预训练模型和LoRA微调权重是分开的,LoRA权重在推理时动态加载;而m是成为一个新的完全体模型
merge_model = peft_model.merge_and_unload()
merge_model.save_pretrained("./output/merge_model")# 保存模型
```

### IA3
```python
# 仅记录调用方法
from peft import IA3Config, TaskType, get_peft_model
config = IA3Config(task_type=TaskType.CAUSAL_LM)
```

### 使用不同适配器
```python
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, PeftModel

net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
# 对层0进行Lora微调
config1 = LoraConfig(target_modules=["0"])
model1 = get_peft_model(net1, config1)
model1.save_pretrained("./loraA")
print(model1)
# 对层2进行Lora微调
config2 = LoraConfig(target_modules=["2"])
model2 = get_peft_model(net1, config2)
model2.save_pretrained("./loraB")
print(model2)
# 此时model2会显示层0,层2都被lora,因为net1会记录被A调整的部分
# 但是!!!经验证,实际上loraB只记录了层2的权重调整,因为model2的输入是net1+loraA,输出是net1+loraA+loraB,所以loraB只记录了层2的权重

net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)# 上面的net1被使用后调整了,重新定义原网络

# 使用原网络和保存的loraA参数得到PeftModel
model3 = PeftModel.from_pretrained(net1, model_id="./loraA/", adapter_name="loraA")# 此时的模型是net1+loraA(层0的适配器参数)
model3.active_adapter# 显示当前激活的适配器A

# 改用loraB参数
model3.load_adapter("./loraB/", adapter_name="loraB")# 加载loraB,实际模型结构是net1+loraA+loraB,激活的结构是net1+loraA(还没切换)
model3.set_adapter("loraB")# 切换到loraB,loraA被禁用,模型激活结构变为net1+loraB
model3.active_adapter# 显示当前激活的适配器B

with model3.disable_adapter():
    <code># 需要使用with语句关闭适配器
```


## 低精度训练
默认单精度fp32,每个参数占4Byte.半精度即fp16(更推荐bf16),每个参数占2Byte.  
### 半精度训练实例
```python
model = AutoModelForCausalLM.from_pretrained("<model name>", low_cpu_mem_usage=True, 
                                             torch_dtype=torch.bfloat16, device_map="auto")# 半精度训练
# 建议加载时用

model = model.half()
# 在fine tuning后把调整的参数也转成半精度
```
### 量化
显存占用变少,但是训练推理速度变慢.

INT8 量化即将浮点数$x_f$通过缩放因子scale映射到范围在[-128, 127] 内,用8bit表示即
\[x_q = Clip(Round(x_f*scale))\]
其中scale=127/浮点数绝对值最大值;Round是四舍五入;  
数据中离群值(与其它数值相差很大)的存在会导致丢失很多信息,使用Clip将离群值限制在[-128, 127]范围内.   

反量化的过程为:
\[x_f = x_q/scale\]

因此可以采取混合精度量化:  
将包含了Emergent Features的几个维度从矩阵中分离出来，对其做高精度的矩阵乘法；其余数值接近的部分进行量化


### 8bit,4bit量化与QLoRA模型训练
```python
model = AutoModelForCausalLM.from_pretrained("D:/Pretrained_models/modelscope/Llama-2-7b-ms", low_cpu_mem_usage=True, 
                                             torch_dtype=torch.bfloat16, device_map="auto", load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("D:/Pretrained_models/modelscope/Llama-2-13b-ms", low_cpu_mem_usage=True, 
                                             torch_dtype=torch.bfloat16, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)# 启用nf4量化,启用双重量化
```

## 分布式训练
### 各类并行
data parallel: 每个GPU加载完整的模型,训练的数据不同  
pipeline parallel: 每个GPU加载模型不同的层  
tensor parallel: 把同一层的各部分参数拆分到各个GPU上

3D并行:
![3Dpara](post/instruction/transformers/3Dpara.png)
图中:2(数据并行)\*4(流水并行|横向箭头,代表不同层)\*4(张量并行|竖向箭头,同层的不同参数)=32GPUs  
解释:模型32层,每8层分成一个流水并行块;每个流水并行块分成4个张量并行块,每个张量并行块有4个GPU,共16个GPU;再乘以2行数据并行=32GPUs    

### Distributed DataParallel
![datapara](post/instruction/transformers/datapara.png)

```python
# 指定使用GPU 0, 1和2（不设置device_ids或令其=None，则默认使用所有GPU）
model = nn.DataParallel(model, device_ids=[0, 1, 2])

# 在训练时，需要对loss进行mean()，因为loss需要是标量才可以进行反向传播
```

### Accelerater