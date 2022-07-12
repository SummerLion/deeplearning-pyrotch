import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import net

# 超参数
batch_size = 64
learning_rate = 1e-2
num_epoches = 20


# 数据预处理
# data_tf = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ]
# )
# 对于神经网络，我们第一层的输入就是 28 x 28 = 784，所以必须将得到的数据我们做一个变换，
# 使用 reshape 将他们拉平成一个一维向量
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x


# 下载训练集 MINIST 手写数字训练集
train_dataset = datasets.MNIST(root='../data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=data_tf)

# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# a, a_label = next(iter(train_data))
# # 打印出一个批次的数据大小
# print(a.shape)
# print(a_label.shape)


# 导入网络，定义损失函数和优化方法
model = net.simpleNet(784, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练网络
# 3、执行训练
# for e in range(20):
#     out = net(Variable(x))
#     loss = criterion(out, Variable(y))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
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
        im = Variable(im)
        label = Variable(label)
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




# 测试
'''
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    out = model(img)
    loss = criterion(out,label)
    eval_loss+=loss.data[0]*label.size(0)
    _,pred = torch.max(out,1)
    num_correct = (pred == label).sum()
    eval_acc+=num_correct.data[0]
print('Test loss:{:.6f},Acc:{:.6f}'.format(
    eval_loss/(len(test_dataset)),
    eval_acc/(len(test_dataset))
    )
)'''
