import torch
from torch.autograd import Variable

'''
x = Variable(torch.Tensor([2]), requires_grad=True)
y = x + 2
z = y ** 2 + 3
print(z)
# 使用自动求导
z.backward()
print(x.grad)

x = Variable(torch.randn(10, 20), requires_grad=True)
y = Variable(torch.randn(10, 5), requires_grad=True)
w = Variable(torch.randn(20, 5), requires_grad=True)
out = torch.mean(y - torch.matmul(x, w))
out.backward()
# print(x.grad)
print(y.grad)
print(w.grad)
'''

'''
# 对多维数组的自动求导机制
m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True)
n = Variable(torch.zeros(1, 2))
print(m)
print(n)
n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3
print(n)
n.backward(torch.ones_like(n))  # 将 (w0, w1) 取成 (1, 1)
print(m.grad)
'''

# 多次自动求导
x = Variable(torch.FloatTensor([3]), requires_grad=True)
y = x * 2 + x ** 2 + 3
print(y)
y.backward(retain_graph=True)  # 设置 retain_graph 为 True 来保留计算
print(x.grad)
y.backward()
print(x.grad)

# 练习
x = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
k = Variable(torch.zeros_like(x))
k[0] = x[0] ** 2 + 3 * x[1]
k[1] = x[1] ** 2 + 2 * x[0]
j = torch.zeros(2, 2)

k.backward(torch.FloatTensor([1, 0]), retain_graph=True)
j[0] = x.grad.data
print('====-', j[0])
x.grad.data.zero_()
print('====>', x.grad.data)
k.backward(torch.FloatTensor([0, 1]))
j[1] = x.grad.data
print(j)
