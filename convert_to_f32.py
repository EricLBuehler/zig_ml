import torch
from safetensors.torch import save_file, load_file

file = "model-00001-of-00002.safetensors"

tensors = load_file(file)
for name, tensor in tensors.items():
    tensors[name] = tensor.to(torch.float)
save_file(tensors, file)