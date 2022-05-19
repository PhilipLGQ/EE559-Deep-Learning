import torch
import torch.nn as nn

from .others.utils import StatsTracer, compute_psnr
from torch.optim import Adam, lr_scheduler, SGD
from pathlib import Path


class Model():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Noise2Noise().to(self.device)

    def load_pretrained_model(self) -> None:
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model = torch.load(model_path).to(self.device)

    def train(self, train_input, train_target, num_epochs=50, batch_size=100) -> None:
        # val_input = torch.load()

        self.loss = nn.L1Loss().to(self.device)
        # self.loss = nn.MSELoss()
        # self.loss = HDRLoss()

        self.optim = Adam(self.model.parameters(), lr=0.001,
                          betas=(0.9, 0.999), eps=1e-08)
        # self.optim = SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.1, patience=5, verbose=True)

        train_input = train_input.to(self.device)
        train_target = train_target.to(self.device)

        # stats to track during training
        stats = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': []
        }

        for epoch in range(num_epochs):
            tr_loss = StatsTracer()

            for sample_idx in range(0, train_input.size(0), batch_size):
                output = self.model(train_input.narrow(0, sample_idx, batch_size))
                loss = self.loss(output, train_target.narrow(0, sample_idx, batch_size))
                tr_loss.update(loss.item())

                self.model.zero_grad()
                loss.backward()
                self.optim.step()

            # self.epoch_end(stats, epoch, tr_loss.avg, val_input, val_target, batch_size)
            self._epoch_end(epoch)
            print("Epoch: {a}/{b}, Training Loss: {c}".format(a=epoch+1, b=num_epochs, c=tr_loss.avg))
            tr_loss.reset()

    def predict(self, test_input, batch_size=100) -> torch.Tensor:
        test_input.to(self.device)
        test_pred = []

        for sample_idx in range(0, test_input.size(0), batch_size):
            prediction = self.model(test_input.narrow(0, sample_idx, batch_size))
            test_pred.append(prediction.cpu())

        return torch.cat(test_pred, dim=0)

    def eval(self, val_input, val_target, batch_size=100):
        val_input = val_input.to(self.device)
        val_target = val_target.to(self.device)

        val_loss = StatsTracer()
        model_output = []

        for sample_idx in range(0, val_input.size(0), val_target.size(0)):
            output = self.model(val_input.narrow(0, sample_idx, batch_size))
            loss = self.loss(output, val_target)
            val_loss.update(loss.item())

            model_output.append(output.cpu())

        model_output = torch.cat(model_output, dim=0)
        val_psnr = compute_psnr(model_output, val_target)

        return val_loss.avg, val_psnr

    def _epoch_end(self, epoch):
        # Save model
        torch.save(self.model.cpu(), 'model_loss{}_epoch{}.pth'.format(self.loss.__name__, epoch))
        print("Model checkpoint saved...")


# Noise2Noise model class (nn.Module)
class Noise2Noise(nn.Module):
    def __init__(self, in_channel=3, out_channel=3):
        """N2N Initialization"""
        super(Noise2Noise, self).__init__()

        self._encode1 = nn.Sequential(
            nn.Conv2d(in_channel, 48, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self._encode2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self._bottom = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self._decode1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self._decode2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self._decode3 = nn.Sequential(
            nn.Conv2d(96 + in_channel, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channel, 3, padding=1, stride=1),
            nn.LeakyReLU(0.1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        # encode
        pool1 = self._encode1(x)
        pool2 = self._encode2(pool1)
        pool3 = self._encode2(pool2)
        pool4 = self._encode2(pool3)
        pool5 = self._encode2(pool4)

        # bottom
        btm = self._bottom(pool5)

        # decode
        cat5 = torch.cat((btm, pool4), dim=1)
        up1 = self._decode1(cat5)
        cat4 = torch.cat((up1, pool3), dim=1)
        up2 = self._decode2(cat4)
        cat3 = torch.cat((up2, pool2), dim=1)
        up3 = self._decode2(cat3)
        cat2 = torch.cat((up3, pool1), dim=1)
        up4 = self._decode2(cat2)
        cat1 = torch.cat((up4, x), dim=1)
        output = self._decode3(cat1)

        return output
