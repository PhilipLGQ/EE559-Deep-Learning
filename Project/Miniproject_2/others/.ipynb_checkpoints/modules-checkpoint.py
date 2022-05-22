import math
import torch

from torch import empty, cat, arange, Tensor
from torch.nn.functional import fold, unfold


# ===========================================
# Module Superclass
class Module(object):
    def forward(self, *_input):
        raise NotImplementedError

    def backward(self, *_grad_outputs):
        raise NotImplementedError

    def zero_grad(self):
        return

    def param(self):
        return []

    def to(self):
        return

# ===========================================
# Noise2Noise model class (nn.Module)
# class Noise2Noise(Module):
#     def __init__(self, in_channel=3, out_channel=3):
#         """N2N Initialization"""
#         super(Noise2Noise, self).__init__()
#         pass
#
#     def forward(self, x):
#         pass

# ===========================================


# ===========================================
# Functional layer modules
# Conv2d as unfold + matrix multiplication + fold (zero padding, squared kernel, and same stride on h,w directions)
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            self.k_size = (kernel_size, kernel_size)
        else:
            self.k_size = kernel_size

        # Uniform k for parameter initialization
        k = 1 / (out_channels * self.k_size[0] * self.k_size[1])

        self.w = empty(out_channels, in_channels, self.k_size[0], self.k_size[1], dtype=torch.float32). \
            uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = empty(out_channels, dtype=torch.float32).zero_()

        self.b = empty(out_channels, dtype=torch.float32).uniform_(-math.sqrt(k), math.sqrt(k))
        self.db = empty(out_channels, dtype=torch.float32).zero_()

    def forward(self, input):
        self.input = input.float()
        x_unfold = unfold(self.input, kernel_size=self.k_size, stride=self.stride)
        wxb = self.w.view(self.out_channels, -1) @ x_unfold + self.b.view(1, -1, 1)
        output = wxb.view(self.input.size(0), self.out_channels,
                          math.floor((self.input.size(2) - self.k_size[0]) / self.stride) + 1, -1)
        return output

    def backward(self, grad_output):
        grad_reshape = grad_output.permute(1, 2, 3, 0).reshape(self.out_channels, -1)
        x_unfold = unfold(self.input, kernel_size=self.k_size, stride=self.stride)
        x_reshape = x_unfold.permute(2, 0, 1).reshape(grad_reshape.size(2), -1)

        # dl_db = dl_ds
        self.db.data = grad_output.sum(axis=(0, 2, 3))

        # dl_dw = dl_ds * (x) ^ T
        self.dw.data = (grad_reshape @ x_reshape).reshape(self.w.size())

        # dl_dx = (w) ^ T * dl_ds
        dx_unfold = (self.w.reshape(self.out_channels, -1).t()) @ grad_reshape
        dx_unfold = dx_unfold.reshape(x_unfold.permute(1, 2, 0).size()).permute(2, 0, 1)
        dx = fold(dx_unfold, (self.input.size(2), self.input.size(3)), kernel_size=self.k_size, stride=self.stride)

        return dx

    def param(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()


# ConvTranspose2d
class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if isinstance(kernel_size, int):
            self.k_size = (kernel_size, kernel_size)
        else:
            self.k_size = kernel_size

        # Uniform k for parameter initialization
        k = 1 / (out_channels * self.k_size[0] * self.k_size[1])

        self.w = empty(out_channels, in_channels, self.k_size[0], self.k_size[1], dtype=torch.float32) \
            .uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = empty(out_channels, in_channels, self.k_size[0], self.k_size[1], dtype=torch.float32)\
            .zero_().float()

        self.b = empty(out_channels, dtype=torch.float32).uniform_(-math.sqrt(k), math.sqrt(k))
        self.db = empty(out_channels, dtype=torch.float32).zero_()

    def forward(self, input):
        self.input = input.float()
        H_out = 1 + self.stride * (input.size(2) - 1) + (self.k_size[0] - 1)
        W_out = 1 + self.stride * (input.size(3) - 1) + (self.k_size[1] - 1)

        x_reshape = self.input.permute(1, 2, 3, 0).reshape(self.in_channels, -1)
        w_reshape = self.w.reshape(self.in_channels, -1)

        out_unfold = ((w_reshape.t()) @ x_reshape).permute(2, 0, 1)
        out = fold(out_unfold, (H_out, W_out), kernel_size=self.k_size, stride=self.stride) + self.b.view(1, -1, 1, 1)
        return out

    def backward(self, grad_outputs):
        grad_unfold = unfold(grad_outputs, kernel_size=self.k_size, stride=self.stride)
        x_reshape = self.input.permute(1, 2, 3, 0).reshape(self.in_channels, -1)
        grad_reshape = grad_unfold.permute(2, 0, 1).reshape(self.in_channels, -1)

        # dl_dw = dl_ds * (x) ^ T
        self.dw.data = (x_reshape @ grad_reshape).reshape(self.w.size())

        # dl_db = dl_ds
        self.db.data = grad_outputs.sum(axis=(0, 2, 3))

        # dl_dx = (w) ^ T * dl_ds
        dx_raw = self.w.reshape(self.in_channels, -1) @ grad_unfold
        dx = dx_raw.view(dx_raw.size(0), self.in_channels, self.input.size(2), self.input.size(3))
        return dx

    def param(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()

# ===========================================


# ===========================================
# Optimizer modules
# Stochastic gradient descent (SGD)
class SGD(Module):
    def __init__(self, params, lr=0.01, maximize=False):
        if lr <= 0.0:
            raise ValueError("Learning rate should be positive. Please use 'maximize=True' for maximization")
        self.maxflag = maximize
        self.params = params
        self.lr = lr

    def step(self):
        # single step for each parameter in the parameter list
        for param in self.params:
            if len(param) > 0:
                w, grad = param
                if not self.maxflag:
                    w.add_(-self.lr * grad)
                else:
                    w.add_(self.lr * grad)

    def zero_grad(self):
        # initialize/cleanse gradient in model
        for param in self.params:
            if len(param) > 0:
                w, grad = param
                if w is not None:
                    grad.zero_()


# ===========================================


# ===========================================
# Loss function modules
# Mean Square Loss
class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.pred = None
        self.target = None

    def forward(self, prediction, target):
        self.pred = prediction
        self.target = target.view(prediction.shape)

        # MSE = (y_pred - y_true) ** 2 / sample_size (or batch_size)
        loss = (self.pred - self.target).pow(2).sum(1).mean()
        return loss

    def backward(self):
        # dl_d(y_pred) = 2 * (y_pred - y_true) / sample_size (or batch_size)
        return 2 * (self.pred - self.target) / self.pred.size(0)


# ===========================================


# ===========================================
# Activation function modules
# ReLU
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.s = 0.

    def forward(self, s):
        self.s = s.clone()
        return self.s.relu()

    def backward(self, dl_dx):
        # dx_ds = 1 (s>0), else 0
        dx_ds = self.s.gt(0).float()

        # Hadamard multiplication
        dl_ds = dl_dx.mul(dx_ds)
        return dl_ds


# Sigmoid
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.s = 0.

    def forward(self, s):
        self.s = s.clone()
        ones = empty(s.shape).fill_(1)
        return ones.div((-self.s).exp())

    def backward(self, dl_dx):
        ones = empty(self.s.shape).fill_(1)
        sigma = ones.div((-self.s).exp())

        # dx_ds = sigmoid * (1 - sigmoid) by calculation
        dl_ds = sigma * (1 - sigma) * dl_dx
        return dl_ds


# ===========================================


# ===========================================
# Sequential block module
class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.layer_list = list(args)

    def forward(self, x):
        out = x
        for layer in self.layer_list:
            out = layer.forward(out)
        return out

    def backward(self, dl_dx):
        front = dl_dx
        for layer in reversed(self.layer_list):
            front = layer.backward(front)
        return front

    def zero_grad(self):
        for layer in self.layer_list:
            layer.zero_grad()

    def param(self):
        return [p for layer in self.layer_list for p in layer.params()]

    # def to(self, device):
    #

# ===========================================
