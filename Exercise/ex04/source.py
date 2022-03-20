import torch
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels=True, normalize=True, flatten=False)


# Model training
def train_model(model, train_input, train_target, mini_batch_size=100, eta=1e-1, epochs=100):
    criterion = nn.MSELoss()
    for epoch_idx in range(epochs):
        tr_loss = 0

        for batch_idx in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, batch_idx, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, batch_idx, mini_batch_size))

            # training loss
            tr_loss += loss
            model.zero_grad()
            loss.backward()

            # after backward gradient calculation, disable it to save memory
            with torch.no_grad():
                # update each parameter in model
                for p in model.parameters():
                    p -= eta * p.grad

        print("Epoch:{}, Training Loss:{:.4f}".format(epoch_idx+1, tr_loss.item()))


# Calculate number of false prediction
def compute_nb_errors(model, input, target, mini_batch_size):
    error_count = 0
    for batch_idx in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, batch_idx, mini_batch_size))
        _, pred_class = output.max(1)

        for order_idx in range(mini_batch_size):
            if target[batch_idx+order_idx, pred_class[order_idx]] <= 0:
                error_count += 1

    return error_count


class Net(nn.Module):
    def __init__(self, num_hidden_unit):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, num_hidden_unit)
        self.fc2 = nn.Linear(num_hidden_unit, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(576, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 576)))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # # Q2
    # for k in range(10):
    #     model = Net(200)
    #     train_model(model, train_input, train_target, 100)
    #     nb_test_errors = compute_nb_errors(model, test_input, test_target, 100)
    #     print('*******************************************')
    #     print('Test Error Rate for Net {:0.2f}% {:d}/{:d}'
    #           .format((100 * nb_test_errors) / test_input.size(0), nb_test_errors, test_input.size(0)))
    #     print('*******************************************\n')


    # # Q3
    # for nh in [10, 50, 200, 500, 2500]:
    #     model = Net(nh)
    #     train_model(model, train_input, train_target, 100)
    #     nb_test_errors = compute_nb_errors(model, test_input, test_target, 100)
    #     print('*******************************************')
    #     print('Test Error Rate for Net with {:d} Hidden Layers: {:0.2f}%% {:d}/{:d}'
    #           .format(nh, (100 * nb_test_errors) / test_input.size(0), nb_test_errors, test_input.size(0)))
    #     print('*******************************************\n')

    # Q4
    model = Net2()
    train_model(model, train_input, train_target, 100)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, 100)
    print('*******************************************')
    print('Test Error Rate for Net2 {:0.2f}% {:d}/{:d}'
          .format((100 * nb_test_errors) / test_input.size(0), nb_test_errors, test_input.size(0)))
    print('*******************************************\n')
