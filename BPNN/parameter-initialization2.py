from torch.nn import init
import numpy as np
import torch
from torch import nn
# 定义一个sequential模型
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
print(net1[0].weight)
# 这个方法已经被弃用
init.xavier_uniform(net1[0].weight) # Xavier初始化方法，Pytorch直接内置了其实现
print(net1[0].weight)