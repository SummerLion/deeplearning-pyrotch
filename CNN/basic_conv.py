import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('data/cat.png').convert('L')  # 读入一张灰度图的图片
im = np.array(im, dtype='float32')  # 将其转换为一个矩阵
print('im----->1', im)
plt.imshow(im.astype('uint8'), cmap='gray')
plt.show()
print('im.shape---->', im.shape)
# 将图片矩阵转化为pytorch tensor，并适配卷积输入要求
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
print('im----->2', im)
# 定义一个算子对其进行轮廓检测
# ********************************卷积层**********************************
# 方式一：使用 nn.Conv2d
# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

conv1 = nn.Conv2d(1, 1, 3, bias=False)  # 定义卷积
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  # 定义轮廓检测算子
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))  # 适配卷积的输入输出
conv1.weight.data = torch.from_numpy(sobel_kernel)  # 给卷积的 kernel 赋值
edge1 = conv1(Variable(im))  # 作用在图片上
edge1 = edge1.data.squeeze().numpy()  # 将输出转换为图片的格式
# 可视化边缘检测之后的结果
plt.imshow(edge1, cmap='gray')
plt.show()

# 方式二：使用 F.conv2d
# sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  # 定义轮廓检测算子
# sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))  # 适配卷积的输入输出
# weight = Variable(torch.from_numpy(sobel_kernel))
# edge2 = F.conv2d(Variable(im), weight)  # 作用在图片上
# edge2 = edge2.data.squeeze().numpy()  # 将输出转换为图片的格式
# # 可视化边缘检测之后的结果
# plt.imshow(edge2, cmap='gray')
# plt.show()
# ********************************池化层**********************************
# 方式一：使用 nn.MaxPool2d
pool1 = nn.MaxPool2d(2, 2)
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
small_im1 = pool1(Variable(im))
small_im1 = small_im1.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
plt.imshow(small_im1, cmap='gray')
plt.show()
# 方式二：使用F.max_pool2d
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
small_im2 = F.max_pool2d(Variable(im), 2, 2)
small_im2 = small_im2.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
plt.imshow(small_im2, cmap='gray')
plt.show()