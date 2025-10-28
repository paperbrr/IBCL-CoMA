import torch
from torch.utils.data import DataLoader, TensorDataset
import math

def make_dataloader_from_buffer(buffer, batch_size, shuffle=True):
    if not buffer:
        return None
    x_buf, y_buf = zip(*buffer)
    x_buf = torch.stack(x_buf)
    y_buf = torch.tensor(y_buf)
    dataset = TensorDataset(x_buf, y_buf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_buffer_samples(train_loader, sample_count, device='cuda'):
    x_list, y_list = [], []
    for x_batch, y_batch in train_loader:
        x_list.append(x_batch)
        y_list.append(y_batch)

    x_all = torch.cat(x_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    total_samples = x_all.size(0)
    indices = torch.randperm(total_samples)

    sampled_idx = indices[:sample_count]
    unsampled_idx = indices[sample_count:]

    sampled = [(x_all[i], y_all[i].item()) for i in sampled_idx]
    unsampled = [(x_all[i], y_all[i].item()) for i in unsampled_idx]

    return sampled, unsampled