import torch
import numpy as np
from matplotlib import pyplot as plt

learning_rate = 0.01

# 1. 准备数据 y = 3x+0.8，准备参数
x = torch.rand([500, 1])  # 特征1个
y_true = 3 * x + 0.8

# 2. 通过模型计算y_predict
w = torch.rand([1, 1], requires_grad=True)
# 如果 x为[50,] w为一阶  如果x为[50,1] w为[1,1]
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)
# b初始值为1也行不用加dtype 用0报错


# 4 通过循环 反向传播 更新参数
for i in range(500):
    # 3 计算loss
    y_predict = torch.matmul(x, w) + b
    loss = (y_true - y_predict).pow(2).mean()

    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    loss.backward()
    w.data = w.data - learning_rate * w.grad
    b.data = b.data - learning_rate * b.grad
    if i % 50 == 0:
        print("w, b, loss", w.item(), b.item(), loss.item())

# 绘制图形，观察训练结束的预测值和真实值
plt.figure(figsize=(20, 8))
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1), c='r')
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1))
plt.show()
