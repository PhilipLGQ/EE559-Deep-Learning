import io
import pickle
import torch
from torch import arange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def padding_stride(tensor, stride):
    batch_size, channel, hight, width = tensor.shape
    A = torch.zeros((batch_size, channel, hight, width * stride - (stride - 1))).to(device)
    A[:, :, :, arange(width) * stride] = tensor
    B = torch.zeros((batch_size, channel, hight * stride - (stride - 1), width * stride - (stride - 1))).to(device)
    B[:, :, arange(hight) * stride, :] = A
    return B

def circular_padding(tensor, h, w):
    batch_size, channel, hight, width = tensor.shape
    out = torch.zeros((batch_size, channel, hight + 2 * h, width + 2 * w)).to(device)
    out[:, :, h:h + hight, w:w + width] = tensor
    return out

def padding_inverse(tensor, stride):
    p = arange(1, tensor.shape[2] + 1, stride) - 1
    q = arange(1, tensor.shape[3] + 1, stride) - 1
    row = tensor[:, :, p, :]
    res = row[:, :, :, q]
    return res

def circular_inverse(tensor, h, w):
    _,_, hight, width = tensor.shape
    out = tensor[:, :, h:hight - h, w:width - w]
    return out

def output_padding(tensor, h, w):
    batch_size, channel, hight, width = tensor.shape
    out = torch.zeros((batch_size, channel, hight + h, width + w)).to(device)
    out[:, :, 0:hight, 0:width] = tensor
    return out

def inverse_output(tensor, h, w):
    batch_size, channel, hight, width = tensor.shape
    out = tensor[:, :, 0:hight - h, 0:width - w]
    return out

def load_data(train_path, val_path):
    train_path = train_path + 'train_data.pkl'
    val_path = val_path + 'val_data.pkl'

    train_input0, train_input1 = torch.load(train_path)
    val_input, val_target = torch.load(val_path)

    return train_input0, train_input1, val_input, val_target


# [0, 1] Normalization
def zero_one_norm(data_tensor):
    return data_tensor.float() / 255.0


def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


class StatsTracer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.avg = 0
        self.nb_batch = 0

    def update(self, value, count=1):
        self.value = value
        self.sum += value * count
        self.nb_batch += count
        self.avg = self.sum / self.nb_batch

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)