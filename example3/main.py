import torch
import gpu

len = 5
a = torch.randn(len, device="cuda")
b = torch.ones_like(a, device="cuda")
c = torch.empty_like(a)
gpu.vadd(a, b, c)

print(a)
print(b)
print(c)
