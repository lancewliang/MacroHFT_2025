import torch
print(torch.cuda.is_available())   # True 表示 CUDA 可用
print(torch.cuda.device_count())   # 查看可用 GPU 数量
print(torch.cuda.get_device_name(0))  # 查看第一个 GPU 名称

import torch
import time
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device ="cpu"
# 创建一个大张量
x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)
 
# 计算并计时
start_time = time.time()
z = x @ y
torch.cuda.synchronize()  # 等待 GPU 计算完成
end_time = time.time()
 
print(f"矩阵乘法耗时: {end_time - start_time:.4f} 秒")
print(f"结果张量设备: {z.device}")



import torch
import time
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {torch.cuda.get_device_name(0)}")
 
# 创建超大矩阵
size = 30000
x = torch.randn(size, size, device=device)
y = torch.randn(size, size, device=device)
 
start = time.time()
z = x @ y
torch.cuda.synchronize()
end = time.time()
 
print(f"大矩阵乘法耗时: {end - start:.2f} 秒")