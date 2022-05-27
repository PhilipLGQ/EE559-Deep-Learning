from .others.modules import *
from pathlib import Path
from .others.utils import CPU_Unpickler
from Miniproject_2.others.utils import zero_one_norm, StatsTracer
import pickle
import torch


class Model():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Sequential(
            Conv2d(3, 48, kernel_size=(3, 3), stride=2, padding=1),
            ReLU(),
            Conv2d(48, 96, kernel_size=(3, 3), stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(96, 48, kernel_size=(3, 3), stride=2, padding=1, out_padding=1),
            ReLU(),
            ConvTranspose2d(48, 3, kernel_size=(3, 3), stride=2, padding=1, out_padding=1),
            Sigmoid()
        )

    def load_pretrained_model(self) -> None:
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, 'rb') as f:
            if torch.cuda.is_available():
                load_params = pickle.load(f)
            else:
                load_params = CPU_Unpickler(f).load()
        self.model.load_model(load_params)

    def train(self, train_input, train_target, num_epochs=100, batch_size=16) -> None:
        self.loss = MSELoss()
        self.optim = SGD(self.model.params(), lr=0.1)

        train_input = zero_one_norm(train_input)
        train_target = zero_one_norm(train_target)
        train_input = train_input.float().to(self.device)
        train_target = train_target.float().to(self.device)

        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))
            tr_loss = StatsTracer()

            for sample_idx in range(0, train_input.size(0), batch_size):
                output = self.model.forward(train_input.narrow(0, sample_idx, batch_size))
                loss = self.loss.forward(output, train_target.narrow(0, sample_idx, batch_size))
                tr_loss.update(loss.item())

                self.model.zero_grad()
                self.model.backward(self.loss.backward())
                self.optim.step()

            print("Epoch: {a}/{b}, Training Loss: {c}".format(a=epoch + 1, b=num_epochs, c=tr_loss.avg))
            tr_loss.reset()
        #self.epoch_end(num_epochs)

    def predict (self, test_input ) -> torch.Tensor :
        #: test_input : tensor of size (N1 , C, H, W) with values in range 0 -255 that has to
        #be denoised by the trained or the loaded network .
        ##: returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.

        test_input = zero_one_norm(test_input)
        test_input = test_input.float().to(self.device)

        source_denoised = self.model.forward(test_input) * 255
        return source_denoised

    def epoch_end(self, epoch):
        # Save model
        with open('bestmodel.pth', 'wb') as f:
            pickle.dump(self.model.param(), f)
        print("Model checkpoint saved...")


