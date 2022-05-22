from Miniproject_2.model import Model
from argparse import ArgumentParser
from Miniproject_2.others.utils import load_data, zero_one_norm


# Function for training settings
def parse_arg():
    parser = ArgumentParser(description="Training Setup")

    # Data params
    parser.add_argument('-t', '--train-dir', help='Path of training data', default='Miniproject_1/data/')
    parser.add_argument('-v', '--val-dir', help='Path of validation data', default='Miniproject_1/data/')
    parser.add_argument('-m', '--model-dir', help='Path of model saving', default='Miniproject_1/')
    parser.add_argument('-n', '--normalize', help='Normalize from (0, 255) to (0, 1)', default=True, type=bool)
    parser.add_argument('-a', '--augment', help='Augment training data', default=False, type=bool)

    # Training settings
    parser.add_argument('-lr', '--learning-rate', help='learning rate for Adam', default=0.001)
    parser.add_argument('-b', '--betas', help='betas for Adam', default=[0.9, 0.999], type=list)
    parser.add_argument('-bt', '--batch-size', help='size of batch', default=100)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=100)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--cuda', help='Use cuda(GPU) for training', default=False, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    params = parse_arg()

    # Load dataset
    noisy_tr1, noisy_tr2, noisy_val, clean_val = load_data(params.train_dir, params.val_dir)

    if params.normalize:
        noisy_tr1 = zero_one_norm(noisy_tr1)
        noisy_tr2 = zero_one_norm(noisy_tr2)
        noisy_val = zero_one_norm(noisy_val)
        clean_val = zero_one_norm(clean_val)

    # Model training
    print("Model training started...")
    model = Model()
    model.train(noisy_tr1, noisy_tr2, params.epochs, params.batch_size)
    print("Model training finished...")

    # Testing on Validation Data
    model.eval(noisy_val, clean_val, params.epochs)
    print("Model evaluation finished...")

    # Load dataset
    noisy_tr1, noisy_tr2, noisy_val, clean_val = load_data(params.train_dir, params.val_dir)

    if params.normalize:
        noisy_tr1 = zero_one_norm(noisy_tr1)
        noisy_tr2 = zero_one_norm(noisy_tr2)
        noisy_val = zero_one_norm(noisy_val)
        clean_val = zero_one_norm(clean_val)

    # Model training
    print("Model training started...")
    model = Model()
    model.train(noisy_tr1, noisy_tr2, params.epochs, params.batch_size)
    print("Model training finished...")

    # Testing on Validation Data
    model.eval(noisy_val, clean_val, params.epochs)
    print("Model evaluation finished...")

