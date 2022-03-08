import torch
from torch import Tensor

''' 
1 - Nearest Neighbor 
Arguments:
    train_input - 2d tensor of Dim(n x d) containing training vectors
    train_traget - 1d tensor of Dim(n) containing training labels
    x - 1d tensor of Dim(d) containing test vector
    
Return:
    The nearest classification result of the given test vector
'''
def nearest_classification(train_input, train_target, x):
    dist_euc = (train_input - x).pow(2).view(-1)
    label = torch.argmin(dist_euc)

    return train_target[label]


'''
2 - Error Estimation
Arguments:
    train_input - 2d float tensor of Dim(n x d) containing training vectors
    train_target - 1d long tensor of Dim(n) containing train labels
    test_input - 2d float tensor of Dim(n x d) containing test vectors
    test_target - 1d long tensor of Dim(m) containing test labels
    mean - None / 1d float tensor of Dim(d)
    proj - None / 2d float tensor of Dim(c x d)

Return:
    nb_error - number of classification errors, integer
'''
def compute_nb_errors(train_input, train_target, test_input, test_target, mean=None, proj=None):
    if mean is not None:
        train_input -= mean
        test_input -= mean
    if proj is not None:
        train_input = torch.mm(train_input, proj.t())
        test_input = torch.mm(test_input, proj.t())

    nb_error = 0
    for idx in range(test_input.shape[0]):
        if test_target[idx] != nearest_classification(train_input, train_target, test_input[idx]):
            nb_error += 1

    return nb_error


'''
3 - PCA
Arguments:
    x: 2d float tensor of Dim(n x d)

Return:
    mean of x and decreasing order of eigenvalues;
'''
def PCA(x):
    mean_x = x.mean(dim=0)
    center_x = x - mean_x
    e = torch.eig(torch.mm(center_x.t(), center_x), eigenvectors=True)

    t = e[0].narrow(1, 0, 1).reshape((e[0].shape[0]))
    _, indices = torch.sort(t, descending=True)

    return mean_x, e[1][indices]