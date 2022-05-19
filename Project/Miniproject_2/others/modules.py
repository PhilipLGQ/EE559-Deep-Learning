import math
import torch

from torch import empty, cat, arange, Tensor
from torch.nn.functional import fold, unfold


# Module Superclass
class Module(object):
    def forward(self, *_input):
        raise NotImplementedError

    def backward(self, *_grad_outputs):
        raise NotImplementedError

    def zero_grad(self):
        return

    def params(self):
        return []


# ===========================================
# Functional layer modules
# Conv2d as unfold + matrix multiplication + fold (zero padding, squared kernel, and same stride on h,w directions)
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.b_flag = bias

        self.w = empty(out_channels, in_channels, kernel_size, kernel_size)
        uniform_k = 1. / math.sqrt(self.w.size(1))
        self.w.uniform_(-math.sqrt(uniform_k), math.sqrt(uniform_k)).double()

        if bias:
            self.b = empty(out_channels).uniform_(-math.sqrt(uniform_k), math.sqrt(uniform_k)).double()

    def forward(self, inp):
        k_size = self.k_size
        stride = self.stride
        padding = self.padding

        # batch_size = inp.shape[0]
        H_in, W_in = inp.shape[2], inp.shape[3]
        H_out = int(1 + (H_in + 2 * padding - (k_size - 1) - 1) / stride)
        W_out = int(1 + (W_in + 2 * padding - (k_size - 1) - 1) / stride)

        inp_unfold = unfold(inp, (k_size, k_size), stride=stride, padding=padding)
        out_unfold = inp_unfold.transpose(1, 2).matmul(self.w.view(self.w.size(0), -1).t()).transpose(1, 2) + self.b
        out = fold(out_unfold, (H_out, W_out), (1, 1))
        return out

    def backward(self, grad_output):



    def params(self):
        pass

    def zero_grad(self):
        pass


# Upsampling2D
class Upsampling2d(Module):
    def __init__(self):
        super(Upsampling2d, self).__init__()


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

    def params(self):
        return [p for layer in self.layer_list for p in layer.params()]

# ===========================================
