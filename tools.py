import torch
from param import device, device_cpu


def my_zeros(shape):
    return torch.zeros(shape, device=device, dtype=torch.float32)


def my_ones(shape):
    return torch.ones(shape, device=device, dtype=torch.float32)


def my_eye(n):
    return torch.eye(n, device=device, dtype=torch.float32)


def my_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.float32)


def my_long_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.long)


def my_range(start, end, step=1):
    return torch.arange(start=start, end=end, step=step, device=device, dtype=torch.float32)


def dist_mat(x, y, inplace=True):
    d = torch.mm(x, y.transpose(0, 1))
    v_x = torch.sum(x ** 2, 1).unsqueeze(1)
    v_y = torch.sum(y ** 2, 1).unsqueeze(0)
    d *= -2
    if inplace:
        d += v_x
        d += v_y
    else:
        d = d + v_x
        d = d + v_y

    return d


def nn_search(y, x):
    d = dist_mat(x, y)
    return torch.argmin(d, dim=1)
