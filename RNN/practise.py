import torch
import torch.nn as nn

# 实例化一个rnn对象
# 第一个参数：input_size(输入张量x的维度)
# 第二个参数：hidden_size（隐藏层的维度，隐藏层神经元数量）
# 第三个参数：num_layers(隐藏层的层数)
basic_rnn = nn.RNN(input_size=20, hidden_size=50, num_layers=2)
basic_rnn.weight_ih_l0
# print(basic_rnn.weight_ih_l0.shape)
# print(basic_rnn.weight_ih_l1.shape)
# print(basic_rnn.bias_ih_l0.shape)
# print(basic_rnn.bias_ih_l1.shape)
# print(basic_rnn.bias_hh_l0.shape)
# print(basic_rnn.bias_hh_l1.shape)

cell = nn.RNNCell(input_size=3,hidden_size=5)
for name,param in cell.named_parameters():
    print('{} = {}'.format(name,param))