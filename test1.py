import torch

ckpt = torch.load(r"H:\Research\saves\MRScore_save_2025-04-01_05-18-00\v1\checkpoints\epoch=13-step=5000.ckpt", map_location="cpu")
print(ckpt.keys())