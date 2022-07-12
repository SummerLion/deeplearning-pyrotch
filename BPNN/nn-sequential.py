# 用Sequential和Module来定义初文件中的神经网络
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 在构建模型的时候，需要定义需要的参数，对于比较小的模型，可以手动去定义参数，如nn-sequential-module-test中那样
# 但是，对于大的模型，比如100层的神经网络，这个时候再去手动定义就显得非常麻烦。
def plot_decision_boundary(model, x, y):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)

# （一）准备数据
################################################################################
# 这次我们仍然处理一个二分类问题，但是比前面的 logistic 回归更加复杂
np.random.seed(1)
m = 400  # 样本数量
N = int(m / 2)  # 每一类的点的个数
D = 2  # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8')  # label 向量，0 表示红色，1 表示蓝色

a = 4
for j in range(2):
    ix = range(N * j, N * (j + 1))
    print(ix)
    t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
    r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
    x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j
plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
plt.show()

# __________________________________________________________________
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# （二）定义并训练模型
################################################################################
# 1、定义模型：用Sequential模块定义模型
seq_net = nn.Sequential(
    nn.Linear(2, 4),  # PyTorch 中的线性层，wx + b
    nn.Tanh(),
    nn.Linear(4, 1)
)
'''
# 序列模块可以通过索引访问每一层
seq_net[0]  # 访问第一层
# 打印出第一层的权重
w0 = seq_net[0].weight
print(w0)
'''
# 2、定义优化器
param = seq_net.parameters()  # 通过 parameters 可以取得模型的参数
optim = torch.optim.SGD(param, 1.)   # 学习率1
criterion = nn.BCEWithLogitsLoss()
# 3、执行训练
for e in range(10000):
    out = seq_net(Variable(x))
    loss = criterion(out, Variable(y))
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e + 1, loss.item()))


# 4、绘图
def plot_seq(x):
    out = torch.sigmoid(seq_net(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5) * 1
    return out


plot_decision_boundary(lambda x: plot_seq(x), x.numpy(), y.numpy())
plt.title('sequential')
plt.show()

# （三）保存模型
################################################################################
# 方式一：将模型和参数保存在一起
torch.save(seq_net, 'saved model/save_seq_net.pth')
seq_net1 = torch.load('saved model/save_seq_net.pth')  # 读取保存的模型
print('seq_net[0].weight：0000======>', seq_net[0].weight)
print('seq_net1[0].weight：1111======>', seq_net1[0].weight)

# 方式二：仅保存模型参数，如果要重新读入模型的参数，首先需要重新定义一次模型，接着重新读取参数
torch.save(seq_net.state_dict(), 'saved model/save_seq_net_params.pth')
seq_net2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)
seq_net2.load_state_dict(torch.load('saved model/save_seq_net_params.pth'))
print('seq_net2[0].weight：2222======>', seq_net2[0].weight)

