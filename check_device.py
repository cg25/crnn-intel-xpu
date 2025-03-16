import torch
from openvino import Core

print(f"Using device: {torch.xpu.get_device_name(0)}")

print(Core().available_devices)