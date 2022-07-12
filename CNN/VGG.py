import sys

sys.path.append('..')
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from utils import train


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]  # 定义第一层
    for i in range(num_convs - 1):  # 定义后面的很多层
        # print("i", i)
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
    net.append(nn.MaxPool2d(2, 2))  # 定义池化层
    return nn.Sequential(*net)


# block_demo = vgg_block(3, 64, 128)
# print('block_demo', block_demo)
# # 首先定义输入为 (1, 64, 300, 300)
# input_demo = Variable(torch.zeros(1, 64, 300, 300))
# output_demo = block_demo(input_demo)
# print(output_demo.shape)


# 定义一个函数对这个 vgg block 进行堆叠
def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


# 1、定义Vgg结构**********************************************
vgg_net = vgg_stack(
    (1, 1, 2, 2, 2),
    ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512))
)
# print(vgg_net)
# 我们可以看到网络结构中有个 5 个 最大池化，说明图片的大小会减少 5 倍，
test_x = Variable(torch.zeros(1, 3, 256, 256))
test_y = vgg_net(test_x)
print(test_y.shape)  # 可以看到图片减小了32倍


# 2、添加全连接层**********************************************
class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


net = vgg()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()


# 3、训练模型********************************************
# ①获取数据集
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x


train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
# ②开始训练
train(net, train_data, test_data, 20, optimizer, criterion)
# ③保存模型
torch.save(net, './saved model/save_vgg_net.pth')
