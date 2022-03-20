import torch
import numpy as np
import dlc_practical_prologue as prologue


# 1. Activation function
# component-wise tanh function
def sigma(x):
    return x.tanh()


# first derivative of tanh
def dsigma(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)


# 2. Loss
# sum of loss
def loss(v, t):
    return (v - t).pow(2).sum()


# derivative of loss w.r.t. v (predicted tensor)
def dloss(v, t):
    return 2 * (v - t)


# 3. Forward & Backward Pass
# forward pass
def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)

    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)

    return x0, s1, x1, s2, x2


# backward pass
def backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dl_dw1, dl_db1, dl_dw2, dl_db2):
    # propagate backward the derivatives of the loss w.r.t. the activations
    x0 = x
    dl_dx2 = dloss(x2, t)
    dl_ds2 = dl_dx2 * dsigma(s2)
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dl_dx1 * dsigma(s1)

    # update the derivatives w.r.t. the parameters
    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)


if __name__ == '__main__':
    # Initialization/Training Parameters
    zeta = 0.9
    unit = 50
    cls = 10
    epsilon = 1e-6
    steps = 1000

    # Load data & preprocessing
    tr_input, tr_target, te_input, te_target = prologue.load_data(one_hot_labels=True, normalize=True)
    tr_target *= zeta
    te_target *= zeta
    num_tr_sample = tr_input.size(0)
    num_te_sample = te_input.size(0)
    num_feature = tr_input.size(1)
    lr = 0.1 / num_tr_sample

    # Create weight & bias tensors and fill with preset epsilon
    w1 = torch.empty(unit, num_feature).normal_(0, epsilon)
    b1 = torch.empty(unit).normal_(0, epsilon)
    w2 = torch.empty(cls, unit).normal_(0, epsilon)
    b2 = torch.empty(cls).normal_(0, epsilon)

    # Create tensors for gradient sum up w.r.t. weight & bias
    dl_dw1 = torch.empty(w1.size())
    dl_db1 = torch.empty(b1.size())
    dl_dw2 = torch.empty(w2.size())
    dl_db2 = torch.empty(b2.size())

    # print(tr_target.size())
    # print(tr_target[0])
    # print(torch.unique(tr_target), torch.unique(te_target))

    # Perform 1000 GD steps with step size 0.1
    for step_idx in range(steps):
        tr_false = 0
        te_false = 0
        tr_loss = 0

        # gradient tensors reset to zero
        dl_dw1.zero_()
        dl_db1.zero_()
        dl_dw2.zero_()
        dl_db2.zero_()

        # forward & backward pass for each training sample
        for sample_idx in range(num_tr_sample):
            # forward pass
            x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, tr_input[sample_idx])
            pred_cls = x2.max(0)[1].item()

            # training error
            if tr_target[sample_idx, pred_cls] < 0.9:
                tr_false += 1

            # training loss
            tr_loss += loss(x2, tr_target[sample_idx])

            # backward pass
            backward_pass(w1, b1, w2, b2, tr_target[sample_idx],
                          x0, s1, x1, s2, x2, dl_dw1, dl_db1,
                          dl_dw2, dl_db2)

        # gradient update
        w1 -= lr * dl_dw1
        b1 -= lr * dl_db1
        w2 -= lr * dl_dw2
        b2 -= lr * dl_db2

        # test error
        for te_sample_idx in range(num_te_sample):
            _, _, _, _, x2 = forward_pass(w1, b1, w2, b2, te_input[te_sample_idx])
            pred = x2.max(0)[1].item()

            if te_target[te_sample_idx, pred] < 0.9:
                te_false += 1

        # output training/test error after each step
        # training acc: 99.8%, test acc: 84.7% (1000 steps)
        print("*************************************")
        print("Step:{a}, Training Acc:{b}%, Test Acc:{c}%".format(a=step_idx+1,
                                                                  b=round(100 * (1 - tr_false / num_tr_sample), 4),
                                                                  c=round(100 * (1 - te_false / num_te_sample), 4)))
