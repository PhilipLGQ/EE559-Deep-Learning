from others.modules import *
import torch
import torch.nn as nn
from torch.nn import Unfold, Fold


in_channel = 3
out_channel = 4
kernel_size = (2, 3)

conv = nn.Conv2d(in_channel, out_channel, kernel_size)
x = torch.randn((4, in_channel, 32, 32), dtype=torch.float32, requires_grad=True)

# Output of PyTorch convolution
expected = conv(x)


# Output of convolution as a matrix product
unfolded = nn.functional.unfold(x, kernel_size=kernel_size)
wxb = conv.weight.view(out_channel, -1) @ unfolded + conv.bias.view(1, -1, 1)
actual = wxb.view(4, out_channel, x.shape[2] - kernel_size[0] + 1, x.shape[3] - kernel_size[1] + 1)

conv2 = Conv2d(3, 4, (2, 3))
test1 = conv2.forward(x)

unfolded = nn.functional.unfold(x, kernel_size=kernel_size)
wxb = conv2.w.view(out_channel, -1) @ unfolded + conv2.b.view(1, -1, 1)
actual = wxb.view(4, out_channel, x.shape[2] - kernel_size[0] + 1, x.shape[3] - kernel_size[1] + 1)

torch.testing.assert_allclose(test1, actual)

test2 = conv2.backward(torch.randn_like(actual))


# Transconv
transconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size)

expected2 = transconv(x)

stride = 1
H_out = 1 + stride * (x.size(2) - 1) + (kernel_size[0] - 1)
W_out = 1 + stride * (x.size(3) - 1) + (kernel_size[1] - 1)

x_reshape = x.permute(1, 2, 3, 0).reshape(in_channel, -1)
w_reshape = transconv.weight.reshape(in_channel, -1)
out_unfold = (w_reshape.t()) @ x_reshape
out_unfold = out_unfold.reshape(w_reshape.size(1), -1, x.size(0)).permute(2, 0, 1)
actual2 = fold(out_unfold, (H_out, W_out), kernel_size=kernel_size, stride=stride) + transconv.bias.view(1, -1, 1, 1)

torch.testing.assert_allclose(expected2, actual2)

convtrans = ConvTranspose2d(3, 4, (2, 3))
test3 = convtrans.forward(x)

x_reshape = x.permute(1, 2, 3, 0).reshape(in_channel, -1)
w_reshape = convtrans.w.reshape(in_channel, -1)
out_unfold = (w_reshape.t()) @ x_reshape
out_unfold = out_unfold.reshape(w_reshape.size(1), -1, x.size(0)).permute(2, 0, 1)
actual2 = fold(out_unfold, (H_out, W_out), kernel_size=kernel_size, stride=stride) + convtrans.b.view(1, -1, 1, 1)

torch.testing.assert_allclose(test3, actual2)

test4 = convtrans.backward(torch.randn_like(actual2))
