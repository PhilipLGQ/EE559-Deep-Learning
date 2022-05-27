import math
import torch
from .utils import *
from torch import empty
from torch.nn.functional import fold, unfold

# Set autograd globally off
#torch.set_grad_enabled(False)


# ===========================================
# Module Superclass
class Module(object):
    def forward(self, inp):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        return

    def param(self):
        return []

# ===========================================


# ===========================================
# Functional layer modules
# Conv2d as unfold + matrix multiplication + fold

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.name = 'Conv2d'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            self.k_size = (kernel_size, kernel_size)
        else:
            self.k_size = kernel_size

        # Uniform k for parameter initialization
        k = 1 / (in_channels * self.k_size[0] * self.k_size[1])

        self.weight = empty(out_channels, in_channels, self.k_size[0], self.k_size[1], dtype=torch.float32). \
            uniform_(-k ** .5, k ** .5)
        self.dw = empty(out_channels, in_channels, self.k_size[0], self.k_size[1], dtype=torch.float32).zero_()

        self.bias = empty(out_channels, dtype=torch.float32).uniform_(-k ** .5, k ** .5)
        self.db = empty(out_channels, dtype=torch.float32).zero_()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(self.device)
        self.weight = self.weight.to(self.device)
        self.dw = self.dw.to(self.device)
        self.bias = self.bias.to(self.device)
        self.db = self.db.to(self.device)

    def forward(self, inp):
        self.input = inp.float().to(self.device)
        # print(inp.size())
        x_unfold = unfold(self.input, kernel_size=self.k_size, stride=self.stride,
                          padding=self.padding)

        # wxb = self.w.view(self.out_channels, -1).to(self.device) @ x_unfold + self.b.view(1, -1, 1).to(self.device)
        wxb = self.weight.view(self.out_channels, -1) @ x_unfold + self.bias.view(1, -1, 1)
        output = wxb.view(self.input.size(0), self.out_channels,
                          math.floor((self.input.size(2) - self.k_size[0] + 2 * self.padding) / self.stride) + 1, -1)
        # print(output.size())
        # print(output.max())
        return output

    def backward(self, grad_output):
        h = self.input.size(2) - self.k_size[0] + 2 * self.padding + 1
        w = self.input.size(3) - self.k_size[1] + 2 * self.padding + 1

        grad_output = grad_output.to(self.device)

        if self.stride == 1:
            grad_unfold_dw = grad_output.permute(1, 0, 2, 3).reshape(self.out_channels, -1)
            grad_unfold_dx = unfold(grad_output, kernel_size=self.k_size,
                                    padding=(self.k_size[0] - 1, self.k_size[1] - 1))
        else:
            grad_pad = empty((grad_output.size(0), self.out_channels, h, w),
                             dtype=torch.float32, device=self.device).zero_()
            grad_pad[:, :, ::self.stride, ::self.stride] = grad_output
            grad_unfold_dw = grad_pad.permute(1, 0, 2, 3).reshape(self.out_channels, -1)
            grad_unfold_dx = unfold(grad_pad, kernel_size=self.k_size,
                                    padding=(self.k_size[0] - 1, self.k_size[1] - 1))

        x_unfold = unfold(self.input.permute(1, 0, 2, 3), kernel_size=(h, w), padding=self.padding)  # .to(self.device)

        # dl_dw = dl_ds * (x) ^ T
        # self.dw.data = grad_unfold.matmul(x_unfold).permute(1, 0, 2, 3).reshape(self.w.size())
        self.dw.data = grad_unfold_dw.matmul(x_unfold).permute(1, 0, 2).reshape(self.weight.size())

        # dl_db = dl_ds
        self.db.data = grad_output.sum(dim=(0, 2, 3))

        # dl_dx = (w) ^ T * dl_ds
        w_unfold = torch.flip(self.weight, [2, 3]).permute(1, 0, 2, 3).reshape(self.weight.size(1), -1)
        dx_unfold = w_unfold.matmul(grad_unfold_dx)

        if self.padding == 0:
            dx = dx_unfold.view(self.input.size())
        else:
            dx = dx_unfold.view((self.input.size(0), self.input.size(1),
                                 2 * self.padding + self.input.size(2),
                                 2 * self.padding + self.input.size(3)))[:, :, self.padding:-self.padding,
                 self.padding:-self.padding]

        return dx

    def param(self):
        return [(self.weight, self.dw), (self.bias, self.db)]

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()


# ConvTranspose2d
class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, out_padding=0):
        self.name = 'ConvTranspose2d'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.out_padding = out_padding

        if isinstance(kernel_size, int):
            self.k_size = (kernel_size, kernel_size)
        else:
            self.k_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(out_padding, int):
            self.out_padding = (out_padding, out_padding)
        else:
            self.out_padding = out_padding

        # Uniform k for parameter initialization
        k = 1 / (in_channels * self.k_size[0] * self.k_size[1])

        self.weight = empty(in_channels, out_channels, self.k_size[0], self.k_size[1], dtype=torch.float32) \
            .uniform_(-k ** .5, k ** .5)
        self.dw = empty(in_channels, out_channels, self.k_size[0], self.k_size[1], dtype=torch.float32) \
            .zero_().float()

        self.bias = empty(out_channels, dtype=torch.float32).uniform_(-k ** .5, k ** .5)
        self.db = empty(out_channels, dtype=torch.float32).zero_()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(self.device)
        self.weight = self.weight.to(self.device)
        self.dw = self.dw.to(self.device)
        self.bias = self.bias.to(self.device)
        self.db = self.db.to(self.device)

    def forward(self, inp):
        self.input = inp.to(device)
        self.input = output_padding(circular_padding(padding_stride(self.input, self.stride),
                                           (self.k_size[0] - 1),
                                           (self.k_size[1] - 1)),
                                            self.out_padding[0],
                                            self.out_padding[1])

        x_unfold = unfold(self.input, kernel_size=self.k_size)
        w_unfold = self.weight.permute(1, 0, 2, 3).flip([3, 2])

        out_unfold = w_unfold.reshape(self.out_channels, -1).matmul(x_unfold) + self.bias.view(1, -1, 1)

        out = out_unfold.view(self.input.size(0), self.out_channels,
                              1 + self.input.size(2) - self.k_size[0], -1)

        out = circular_inverse(out, self.padding[0], self.padding[1])
        return out

    def backward(self, grad_outputs):
        grad_outputs = grad_outputs.to(self.device)
        grad_pad = circular_padding(grad_outputs, self.padding[0], self.padding[1])

        # dl_dw = dl_ds * (x) ^ T
        grad_unfold = grad_pad.view(grad_pad.size(0), grad_pad.size(1), -1)
        x_unfold = unfold(self.input, kernel_size=self.k_size).permute(0, 2, 1)

        # dw = grad_unfold.matmul(x_unfold).sum(axis=0).view(self.w.size())
        dw = grad_unfold.matmul(x_unfold).sum(axis=0).view(self.out_channels, self.in_channels,
                                                           self.k_size[0], self.k_size[1])

        self.dw.data = dw.permute(1, 0, 2, 3).flip([3, 2])

        # dl_db = dl_ds
        self.db.data = grad_outputs.sum(axis=(0, 2, 3))

        # dl_dx = (w) ^ T * dl_ds
        w_unfold = self.weight.permute(1, 0, 2, 3).flip([3, 2]).reshape(self.out_channels, -1)
        dx_unfold = w_unfold.t().matmul(grad_unfold)
        dx_pad = fold(dx_unfold, output_size=(self.input.size(2), self.input.size(3)), kernel_size=self.k_size)

        dx = padding_inverse(circular_inverse(inverse_output(dx_pad, self.padding[0], self.padding[1]),
                            self.k_size[0] - 1,
                            self.k_size[1] - 1),
                            self.stride)
        return dx

    def param(self):
        return [(self.weight, self.dw), (self.bias, self.db)]

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()


# ===========================================


# ===========================================
# Optimizer modules
# Stochastic gradient descent (SGD)
class SGD(Module):
    def __init__(self, params, lr=0.01, momentum=0):
        if lr <= 0.0:
            raise ValueError("Learning rate should be positive. Please use 'maximize=True' for maximization")
        self.velo_w = []
        self.velo_b = []

        self.params = params
        self.m = momentum
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_v()

    def init_v(self):
        for layer in self.params:
            if layer.name in ['Conv2d', 'ConvTranspose2d']:
                self.velo_w.append(empty(layer.dw.size()).zero_())
                self.velo_b.append(empty(layer.db.size()).zero_())
            else:
                self.velo_w.append(empty(1).zero_())
                self.velo_b.append(empty(1).zero_())

    def step(self):
        # single step for each parameter in the parameter list
        for i, layer in enumerate(self.params):
            if layer.name in ['Conv2d', 'ConvTranspose2d']:
                self.velo_w[i] = self.velo_w[i].to(self.device) * self.m + layer.dw
                self.velo_b[i] = self.velo_b[i].to(self.device) * self.m + layer.db

                layer.weight -= self.lr * self.velo_w[i]
                layer.bias -= self.lr * self.velo_b[i]

    def zero_grad(self):
        for layer in self.params:
            if layer.name in ['Conv2d', 'ConvTranspose2d']:
                layer.weight.zero_()
                layer.bias.zero_()


# ===========================================


# ===========================================
# Loss function modules
# Mean Square Loss
class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.name = 'MSELoss'
        self.diff = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, prediction, target):
        # MSE = (y_pred - y_true) ** 2 / sample_size (or batch_size)
        self.diff = (prediction - target).to(self.device)
        loss = self.diff.pow(2).mean()
        return loss

    def backward(self):
        # dl_d(y_pred) = 2 * (y_pred - y_true) / nb_elements
        return 2 * self.diff / torch.numel(self.diff)

    def param(self):
        return []

    def zero_grad(self):
        pass


# ===========================================


# ===========================================
# Activation function modules
# ReLU
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.name = 'ReLU'
        self.input = None

    def forward(self, inp):
        self.input = inp
        out = inp * (inp > 0)
        return out

    def backward(self, dl_dx):
        # dx_ds = 1 (s>0), else 0
        dl_ds = dl_dx * (self.input > 0)
        return dl_ds


# Sigmoid
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = 'Sigmoid'
        self.input = None
        self.sigma = None

    def forward(self, inp):
        self.input = inp
        self.sigma = 1.0 / (1 + (-1 * inp).exp())
        return self.sigma

    def backward(self, dl_dx):
        # dx_ds = sigma * (1 - sigma) by calculation
        dl_ds = self.sigma * (1 - self.sigma) * dl_dx
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
        return [p for layer in self.layer_list for p in layer.param()]

    def params(self):
        return self.layer_list

    def load_model(self, load_params):
        self.load_params = load_params

        for i, layer in enumerate(self.layer_list):
            if layer.name in ['Conv2d', 'ConvTranspose2d']:
                layer.weight = self.load_params[i][0]
                layer.dw = self.load_params[i][1]
                layer.bias = self.load_params[i + 1][0]
                layer.db = self.load_params[i + 1][1]

# ===========================================
