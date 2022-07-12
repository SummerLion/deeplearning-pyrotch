import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import time
# 与上一份代码相比：
# 1、使用了自带的loss,
# 2、使用 torch.optim 更新参数
# 设定随机种子
torch.manual_seed(2017)
# 从 data.txt 中读入点
with open('data/data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
# 标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]

# 用过滤器对数据进行分类 filter()方法的第一个参数是一个函数，此处一般用匿名函数的方式表示
points0 = list(filter(lambda x: x[-1] == 0.0, data))  # 存放分类为0的数据
points1 = list(filter(lambda x: x[-1] == 1.0, data))  # 存放分类为1的数据
plot_x0 = [i[0] for i in points0]
plot_y0 = [i[1] for i in points0]
plot_x1 = [i[0] for i in points1]
plot_y1 = [i[1] for i in points1]
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')
plt.show()
# 将数据转换成numpy类型，接着再转换到Tensor
np_data = np.array(data, dtype='float32')
x_data = torch.from_numpy(np_data[:, 0:2])
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1)

x_data = Variable(x_data)
y_data = Variable(y_data)
print('x_data--->', x_data)
print('y_data--->', y_data)


def binary_loss(y_pred, y):
    loss = -(y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return loss


# 使用自带的loss
criterion = nn.BCEWithLogitsLoss()  # 将 sigmoid 和 loss 写在一层，有更快的速度、更好的稳定性
# 使用 torch.optim 更新参数
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))


def logistic_reg(x,w,b):
    return torch.mm(x, w) + b


optimizer = torch.optim.SGD([w, b], 1.)

# 进行1000次更新
start = time.time()
for e in range(1000):
    # 前向传播
    y_pred = logistic_reg(x_data, w, b)
    # loss = binary_loss(y_pred, y_data)  # 计算 loss
    loss = criterion(y_pred, y_data)
    # 反向传播
    optimizer.zero_grad()  # 使用优化器将梯度归 0
    loss.backward()
    optimizer.step()  # 使用优化器来更新参数
    # 计算正确率
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().item() / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e + 1, loss.item(), acc))
during = time.time() - start
print()
print('During Time: {:.3f} s'.format(during))
# 画出更新完成之后的图
w0 = w[0].data[0]
w1 = w[1].data[0]
b0 = b.data[0]
plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1
plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')
plt.show()
