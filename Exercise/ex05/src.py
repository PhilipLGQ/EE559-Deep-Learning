import torch
import math
from torch import nn
from torch import optim


# Generate uniform input samples and sample targets
def generate_disc_set(nb):
    _input = torch.empty((nb, 2), dtype=torch.float32).uniform_(-1, 1)
    _target = torch.full((nb, 1), 2/math.pi, dtype=torch.float32)\
                   .squeeze().sub(_input.pow(2).sum(1)).sign().add(1).div(2).long()

    return _input, _target


# Model training by standard SGD (mini-batch of size 100)
def train_model(model, train_input, train_target, batch_size=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epochs = 250

    for epoch in range(epochs):
        for sample_idx in range(0, train_input.size(0), batch_size):
            y_output = model(train_input.narrow(0, sample_idx, batch_size))
            loss = criterion(y_output, train_target.narrow(0, sample_idx, batch_size))

            model.zero_grad()
            loss.backward()
            optimizer.step()


# Compute error by cross entropy
def compute_nb_errors(model, data_input, data_target, batch_size=100):
    nb_error = 0

    for sample_idx in range(0, data_input.size(0), batch_size):
        y_pred = model(data_input.narrow(0, sample_idx, batch_size))
        _, pred_class = torch.max(y_pred, dim=1)

        for idx in range(batch_size):
            if pred_class[idx] != data_target[sample_idx + idx]:
                nb_error += 1

    return nb_error


# MLP with 2 input units, a single hidden layer of size 128 and 2 output units
def create_shallow_model():
    model = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
    return model


# MLP with 2 input units, hidden layers of sizes 4, 8, 16, 32, 64, 128, and 2 output units
def create_deep_model():
    model = nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
    return model


if __name__ == "__main__":
    # Generate train and test set, normalize to mean 0 and variance 1
    X_tr, y_tr = generate_disc_set(1000)
    X_te, y_te = generate_disc_set(1000)

    X_tr.sub_(X_tr.mean()).div_(X_tr.std())
    X_te.sub_(X_te.mean()).div_(X_te.std())

    # Train with a normal distribution of std: 1e-3 / 1e-2 / 1e-1 / 1 / 10
    for std in [1e-3, 1e-2, 1e-1, 1, 10]:
        for mdl in [create_shallow_model, create_deep_model]:
            model = mdl()
            with torch.no_grad():
                for p in model.parameters(): p.normal_(0, std)

            train_model(model, X_tr, y_tr)
            tr_nb_error = compute_nb_errors(model, X_tr, y_tr)
            te_nb_error = compute_nb_errors(model, X_te, y_te)

            print("*****************************************************************")
            print("Model: {a}, std: {b}, train error rate: {c}%, test error rate: {d}%".format(a=mdl.__name__,
                                                                                               b=std,
                                                                                               c=tr_nb_error * 100 / X_tr.size(0),
                                                                                               d=te_nb_error * 100 / X_te.size(0)))