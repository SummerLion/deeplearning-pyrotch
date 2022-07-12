import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn

#import torch.optim as optim

# ！！！！！！！！！！！！！！！！！！！！！！这个代码执行起来会报错！！！！！！！！！！！！！！！
# 书上的3.2.4中的代码。一维线性回归代码实现
# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# 定义一个简单的模型
class LinearRegression(nn.Module):
    def __int__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

# 定义损失函数和优化函数 (均方误差作为优化函数，梯度下降法进行优化)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameter(), lr=1e-3)

# 开始训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out.target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2 == 0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch + 1, num_epochs, loss.data[0]))
# 做完训练后预测一下结果
model.eval()
model.cpu()
predict = model(Variable(x_train))
predict = predict.data.mumpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Lint')
plt.show()
