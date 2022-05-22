import torch
from torch import Tensor

from .others.modules import *
from .others.utils import StatsTracer, compute_psnr
from pathlib import Path


class Model():
    def __init__(self) -> None:
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sequential(
            Conv2d(stride=2),
            ReLU(),
            Conv2d(stride=2),
            ReLU(),
            ConvTranspose2d(stride=2),
            ReLU(),
            ConvTranspose2d(stride=2),
            Sigmoid()
        )

    def load_pretrained_model(self) -> None:
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model = torch.load(model_path)

    def train(self, train_input, train_target, num_epochs=50, batch_size=100) -> None:
        self.loss = MSELoss()
        self.optim = SGD(self.model.param(), lr=0.01, maximize=False)

        # train_input = train_input.to(self.device)
        # train_target = train_target.to(self.device)

        # stats to track during training
        # stats = {
        #     'train_loss': [],
        #     'val_loss': [],
        #     'val_psnr': []
        # }

        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))
            tr_loss = StatsTracer()

            for sample_idx in range(0, train_input.size(0), batch_size):
                output = self.model.forward(train_input.narrow(0, sample_idx, batch_size))
                loss = self.loss.forward(output, train_target.narrow(0, sample_idx, batch_size))
                tr_loss.update(loss.item())

                self.model.zero_grad()
                loss.backward()
                self.optim.step()

            self.epoch_end(epoch)
            print("Epoch: {a}/{b}, Training Loss: {c}".format(a=epoch + 1, b=num_epochs, c=tr_loss.avg))
            tr_loss.reset()

    def predict(self, test_input, batch_size=100) -> torch.Tensor:
        # test_input.to(self.device)
        test_pred = []

        for sample_idx in range(0, test_input.size(0), batch_size):
            prediction = self.model.forward(test_input.narrow(0, sample_idx, batch_size))
            test_pred.append(prediction.cpu())

        return torch.cat(test_pred, dim=0)

    def epoch_end(self, epoch):
        # Save model
        torch.save(self.model.cpu(), 'model_loss{}_epoch{}.pth'.format(self.loss.__name__, epoch))
        print("Model checkpoint saved...")

    def eval(self, val_input, val_target, batch_size=100):
        # val_input = val_input.to(self.device)
        # val_target = val_target.to(self.device)

        val_loss = StatsTracer()
        model_output = []

        for sample_idx in range(0, val_input.size(0), val_target.size(0)):
            output = self.model.forward(val_input.narrow(0, sample_idx, batch_size))
            loss = self.loss.forward(output, val_target)
            val_loss.update(loss.item())

            model_output.append(output.cpu())

        model_output = torch.cat(model_output, dim=0)
        val_psnr = compute_psnr(model_output, val_target)

        return val_loss.avg, val_psnr

