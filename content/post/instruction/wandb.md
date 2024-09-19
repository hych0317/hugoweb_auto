---
title: 'wandb'
date: 2024-08-12T14:36:30+08:00
draft: false
categories:
    - 指令语法

---


## 安装库
```python
pip install wandb
# 创建账户
wandb login
# 初始化
# Inside my model training code
import wandb
wandb.init(project="my-project")
# 声明超参数
wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128
# 记录日志
def my_train_loop():
    for epoch in range(10):
        loss = 0 # change as appropriate :)
        wandb.log({'epoch': epoch, 'loss': loss})
# 保存文件
# by default, this will save to a new subfolder for files associated
# with your run, created in wandb.run.dir (which is ./wandb by default)
wandb.save("mymodel.h5")

# you can pass the full path to the Keras model API
model.save(os.path.join(wandb.run.dir, "mymodel.h5"))
```

使用wandb以后，模型输出，log和要保存的文件将会同步到cloud。

## PyTorch应用wandb
我们以一个最简单的神经网络为例展示wandb的用法：

### 首先导入必要的库
```python
from __future__ import print_function
import argparse
import random  # to set the python random seed
import numpy  # to set the numpy random seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb
```

### 使用前的准备
```python
# 登陆你的wandb账户：
# WandB – Login to your wandb account so you can log all your metrics
!wandb login

# 定义Convolutional Neural Network：

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # In our constructor, we define our neural network architecture that we'll use in the forward pass.
        # Conv2d() adds a convolution layer that generates 2 dimensional feature maps
        # to learn different aspects of our image.
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Linear(x,y) creates dense, fully connected layers with x inputs and y outputs.
        # Linear layers simply output the dot product of our inputs and weights.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Here we feed the feature maps from the convolutional layers into a max_pool2d layer.
        # The max_pool2d layer reduces the size of the image representation our convolutional layers learnt,
        # and in doing so it reduces the number of parameters and computations the network needs to perform.
        # Finally we apply the relu activation function which gives us max(0, max_pool2d_output)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Reshapes x into size (-1, 16 * 5 * 5)
        # so we can feed the convolution layer outputs into our fully connected layer.
        x = x.view(-1, 16 * 5 * 5)

        # We apply the relu activation function and dropout to the output of our fully connected layers.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Finally we apply the softmax function to squash the probabilities of each class (0-9)
        # and ensure they add to 1.
        return F.log_softmax(x, dim=1)

# 定义训练函数
def train(config, model, device, train_loader, optimizer, epoch):
    # switch model to training mode. This is necessary for layers like dropout, batchNorm etc.
    # which behave differently in training and evaluation mode.
    model.train()

    # we loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_id, (data, target) in enumerate(train_loader):
        if batch_id > 20:
            break
        # Loop the input features and labels from the training dataset.
        data, target = data.to(device), target.to(device)

        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()

        # Forward pass: Pass image data from training dataset, make predictions
        # about class image belongs to (0-9 in this case).
        output = model(data)

        # Define our loss function, and compute the loss
        loss = F.nll_loss(output, target)

        # Backward pass:compute the gradients of loss,the model's parameters
        loss.backward()

        # update the neural network weights
        optimizer.step()

# 定义测试函数
# wandb.log用来记录一些日志(accuracy,loss and epoch), 便于随时查看网路的性能
def test(args, model, device, test_loader, classes):
    model.eval()
    # switch model to evaluation mode.
    # This is necessary for layers like dropout, batchNorm etc. which behave differently in training and evaluation mode
    test_loss = 0
    correct = 0
    example_images = []

    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)

            # Make predictions: Pass image data from test dataset,
            # make predictions about class image belongs to(0-9 in this case)
            output = model(data)

            # Compute the loss sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Log images in your test dataset automatically,
            # along with predicted and true labels by passing pytorch tensors with image data into wandb.
            example_images.append(wandb.Image(
                data[0], caption="Pred:{} Truth:{}".format(classes[pred[0].item()], classes[target[0]])))

   # wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
   # You can log anything by passing it to wandb.log(),
   # including histograms, custom matplotlib objects, images, video, text, tables, html, pointclounds and other 3D objects.
   # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss
    })

# 初始化一个wandb run，并设置超参数：
# Initialize a new run
wandb.init(project="pytorch-intro")
wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

# config is a variable that holds and saves hyper parameters and inputs
config = wandb.config  # Initialize config
config.batch_size = 4  # input batch size for training (default:64)
config.test_batch_size = 10  # input batch size for testing(default:1000)
config.epochs = 50  # number of epochs to train(default:10)
config.lr = 0.1  # learning rate(default:0.01)
config.momentum = 0.1  # SGD momentum(default:0.5)
config.no_cuda = False  # disables CUDA training
config.seed = 42  # random seed(default:42)
config.log_interval = 10  # how many batches to wait before logging training status
```

### 主函数
```python
def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)      # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    # numpy.random.seed(config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load the dataset: We're training our CNN on CIFAR10.
    # First we define the transformations to apply to our images.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Now we load our training and test datasets and apply the transformations defined above
    train_loader = DataLoader(datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    ), batch_size=config.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    ), batch_size=config.batch_size, shuffle=False, **kwargs)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Initialize our model, recursively go over all modules and convert their parameters
    # and buffers to CUDA tensors (if device is set to cuda)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    # wandb.watch() automatically fetches all layer dimensions, gradients, model parameters
    # and logs them automatically to your dashboard.
    # using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all")
    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, epoch)
        test(config, model, device, test_loader, classes)

    # Save the model checkpoint. This automatically saves a file to the cloud
    torch.save(model.state_dict(), 'model.h5')
    wandb.save('model.h5')


if __name__ == '__main__':
    main()
```