from safetensors.torch import save_file
import torch

tensors = {"a": torch.randn((4, 3)), "b": torch.randn((2, 5))}
save_file(tensors, "test.safetensors")
print(tensors)