import torch
from openvino import Core

print(f"Using device: {torch.xpu.get_device_name(0)}")
print(f"Memory allocated: {torch.xpu.memory_allocated(0)/1024**2:.2f} MB")

print(Core().available_devices)