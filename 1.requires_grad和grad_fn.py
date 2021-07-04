import torch

x = torch.ones(2, 2, requires_grad=True)  # 初始化参数x并设置requires_grad=True用来追踪其计算历史
print(x)
# tensor([[1., 1.],
#        [1., 1.]], requires_grad=True)

y = x + 2
print(y)
# tensor([[3., 3.],
#        [3., 3.]], grad_fn=<AddBackward0>)

z = y * y * 3  # 平方x3
print(z)
# tensor([[27., 27.],
#        [27., 27.]], grad_fn=<MulBackward0>)

out = z.mean()  # 求均值
print(out)
# tensor(27., grad_fn=<MeanBackward0>)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False
a.requires_grad_(True)  # 就地修改
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)  # <SumBackward0 object at 0x4e2b14345d21>
with torch.no_grad():
    c = (a * a).sum()  # tensor(151.6830),此时c没有gard_fn

print(c.requires_grad)  # False
