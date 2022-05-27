import torch

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8 )

def zero_one_norm(data_tensor):
    return data_tensor.float() / 255

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


def add_Gaussian_noise(inputs, mean=0, std=1):

    inputs = inputs.double()
    std = torch.rand(1)*std
    noise = torch.randn_like(inputs)*std+torch.tensor(mean)
    output = torch.clamp(inputs + noise, min=0, max=255)
    return output.type(torch.uint8)


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
