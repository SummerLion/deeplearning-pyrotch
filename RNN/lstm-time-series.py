"""
    时间序列预测
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

look_back = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_csv = pd.read_csv('./data.csv', usecols=[1])
# plt.plot(data_csv)
# plt.show()

# 数据预处理
data_csv = data_csv.dropna()
dataset = data_csv.values  # 142
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# 创建好输入输出

input, target = create_dataset(dataset, look_back=look_back)
# print('input+++>', input)  # (142,2,1)
# print('input===>', input.shape)  # (142,2,1)
# print('target--->', target)  #
# print('target--->', target.shape)  # (142,1)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(input) * 0.7)  # 99

train_X = input[:train_size]
train_Y = target[:train_size]
test_X = input[train_size:]
test_Y = target[train_size:]

# 最后，我们需要将数据改变一下形状，因为 RNN 读入的数据维度是 (seq, batch, feature)，所以要重新改变一下数据的维度，
# 这里只有一个序列，所以 batch 是 1，而输入的 feature 就是我们希望依据的几个月份，这里我们定的是两个月份，所以 feature 就是 2.
train_X = train_X.reshape(-1, 1, look_back)  # -1代表动态调整这个维度上的数据,以保证形状变换前后元素数量不变
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, look_back)
print('train_X', train_X)
print('train_X.reshape', train_X.shape)  # train_X.shape (99, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
print('train_x3.shape', train_x.shape)  # train_x.shape torch.Size([99, 1, 2])


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        seq, batch, hidden = x.shape
        x = x.view(seq * batch, hidden)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(seq, batch, -1)
        return x


net = lstm_reg(input_size=look_back, hidden_size=4, num_layers=2, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 开始训练
for i in range(1000):
    var_x = train_x.to(device)
    var_y = train_y.to(device)

    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(i + 1, loss.item()))
# 训练完成之后，我们可以用训练好的模型去预测后面的结果
net = net.eval()  # 转换成测试模式
input = input.reshape(-1, 1, look_back)
input = torch.from_numpy(input)
var_data = input.to(device)
pred_test = net(var_data)  # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.cpu().view(-1).data.numpy()
# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
plt.show()
