import numpy as np
import torch
from torchvision.datasets import mnist  # 导入 pytorch 内置的 mnist 数据

from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# 使用内置函数下载 mnist 数据集
# train_set = mnist.MNIST('./data', train=True, download=True)
# test_set = mnist.MNIST('./data', train=False, download=True)


# a_data, a_label = train_set[0]
# a_data
# a_label
# a_data = np.array(a_data, dtype='float32')
# print(a_data.shape)
# print(a_data)

# 对于神经网络，我们第一层的输入就是 28 x 28 = 784，所以必须将得到的数据我们做一个变换，
# 使用 reshape 将他们拉平成一个一维向量
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x


train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)  # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
# a, a_label = train_set[0]
# print(a.shape)
# print(a_label)

# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
# a, a_label = next(iter(train_data))
# 打印出一个批次的数据大小
# print(a.shape)
# print(a_label.shape)

# 使用 Sequential 定义 4 层神经网络
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    # nn.Linear(200, 100),
    # nn.ReLU(),
    nn.Linear(200, 10)
)
if torch.cuda.is_available():
    net = net.cuda()
# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    flag = 0
    for im, label in train_data:
        flag += 1
        if torch.cuda.is_available():
            im = Variable(im).cuda()
            label = Variable(label).cuda()
        else:
            im = Variable(im)
            label = Variable(label)


        # im = Variable(im)
        # label = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    print('----flag', flag)
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        if torch.cuda.is_available():
            im = Variable(im).cuda()
            label = Variable(label).cuda()
        else:
            im = Variable(im)
            label = Variable(label)
        # im = Variable(im)
        # label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))

# 画出 loss 曲线和 准确率曲线
plt.plot(np.arange(len(losses)), losses)
plt.title('train loss')
plt.show()

plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.show()

plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.show()

plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()
