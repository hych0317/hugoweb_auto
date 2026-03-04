+++
title = '报错总结'
date = 2024-11-21T14:21:14+08:00
draft = false
categories=['conclusion']
description = '总结刁钻/麻烦的报错及解决方法，容易搜到的一般不记载'
+++

## 环境错误

### LLaVA
解决LLaVA error：ImportError: cannot import name 'LlavaLlamaForCausalLM' from 'llava.model'
```bash
pip uninstall  flash-attn
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
```

### GIT
git error:fatal: the remote end hung up unexpectedly
fatal: early EOF
网络问题，重新下载

## 机器学习库错误
### torch
error:module 'torch.library' has no attribute 'register_fake'
torch和torchvision不匹配

### NVIDIA CUDA
error：libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12
一般是重装torch（pip uninstall、pip install）解决，还有人提到软链接方法，优先考虑torch安装顺序的原因

### transformers


## 程序写法错误