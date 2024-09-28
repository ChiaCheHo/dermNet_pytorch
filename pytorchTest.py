import torch

# 檢查是否有可用的 GPU
if torch.cuda.is_available():
    print(f"PyTorch 正在使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("沒有檢測到 GPU，使用的是 CPU")
