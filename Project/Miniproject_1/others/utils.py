import torch
import torch.nn as nn

from torchvision import transforms


def load_data(train_path, val_path):
    train_dir = train_path + 'train_data.pkl'
    val_dir = val_path + 'val_data.pkl'

    tr_tensor1, tr_tensor2 = torch.load(train_dir)
    val_tensor, clean_tensor = torch.load(val_dir)

    return tr_tensor1, tr_tensor2, val_tensor, clean_tensor


# [0, 1] Normalization
def zero_one_norm(data_tensor):
    return data_tensor.float() / 255


# def augment_data(img_tensor):
#     device = img_tensor.device
#     _, _, H, W = img_tensor.shape
#
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(90),
#
#     ])


def psnr(denoised, ground_truth):
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10 ** -8)


def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


# Tracks stats for average mini-batch loss
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
