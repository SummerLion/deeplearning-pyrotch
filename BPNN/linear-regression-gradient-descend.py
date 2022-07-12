import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.manual_seed(2017)
# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# 画出图像
plt.plot(x_train, y_train, 'bo')
plt.show()
# 转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
# 定义参数 w 和 b
w = Variable(torch.randn(1), requires_grad=True)  # 随机初始化
b = Variable(torch.zeros(1), requires_grad=True)  # 使用 0 进行初始化
# 构建线性回归模型
x_train = Variable(x_train)
y_train = Variable(y_train)


def linear_model(x):
    return x * w + b


# 计算误差
def get_loss(y_, y):
    return torch.mean((y_ - y) ** 2)


# y_ = linear_model(x_train)
# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
# plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
# plt.legend()
# plt.show()
# loss = get_loss(y_, y_train)
# # 打印一下看看 loss 的大小
# print('loss--->', loss)
# # 自动求导
# loss.backward()
# # 查看 w 和 b 的梯度
# print(w.grad)
# print(b.grad)
# # 更新一次参数
# w.data = w.data - 1e-2 * w.grad.data
# b.data = b.data - 1e-2 * b.grad.data
# # 更新完成参数之后，我们再一次看看模型输出的结果
# y_ = linear_model(x_train)
# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
# plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
# plt.legend()
# plt.show()


# 没有特别好的拟合蓝色的真实值，所以我们需要在进行几次更新
for e in range(10):  # 进行 10 次更新
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)

    loss.backward()

    w.data = w.data - 1e-2 * w.grad.data  # 更新 w
    b.data = b.data - 1e-2 * b.grad.data  # 更新 b
    print('epoch: %s, loss: %s' % (e, loss.item()))
    w.grad.zero_()  # 记得归零梯度
    b.grad.zero_()  # 记得归零梯度

y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()


