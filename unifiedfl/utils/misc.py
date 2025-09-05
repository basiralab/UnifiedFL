import os
import psutil
import torch

def count_learnable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def create_dir_if_not_exists(path: str):
    os.makedirs(path, exist_ok=True)
