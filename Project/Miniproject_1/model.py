import torch
import torch.nn as nn

from .others.utils import StatsTracer, compute_psnr
from torch.optim import Adam, lr_scheduler
from pathlib import Path


class Model():
    def __init__(self) -> None:
        self.model = Noise2Noise()
        pass

    def load_pretrained_model(self) -> None:
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model = torch.load(model_path)

        pass

    def train(self, params, train_input, train_target, val_input, val_target) -> None:
        self.params = params

        if self.params.loss == 'l1':
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

        self.optim = Adam(self.model.parameters(), lr=self.params.learning_rate,
                          betas=(self.params.betas[0], self.params.betas[1]))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.1, patience=5, verbose=True)

        if self.params.cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            train_input = train_input.cuda()
            train_target = train_target.cuda()

        # stats to track during training
        stats = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': []
        }

        for epoch in range(params.epochs):
            print("Epoch: {}/{}".format(epoch + 1, params.epochs))
            tr_loss = StatsTracer()

            for sample_idx in range(0, train_input.size(0), params.batch_size):
                output = self.model(train_input.narrow(0, sample_idx, params.batch_size))
                loss = self.loss(output, train_target.narrow(0, sample_idx, params.batch_size))
                tr_loss.update(loss.item())

                self.model.zero_grad()
                loss.backward()
                self.optim.step()

            self.epoch_end(stats, epoch, tr_loss.avg, val_input, val_target, params.batch_size)
            tr_loss.reset()

    def predict(self, test_input, batch_size=100) -> torch.Tensor:
        if torch.cuda.is_available():
            test_input = test_input.cuda()

        test_pred = []

        for sample_idx in range(0, test_input.size(0), batch_size):
            prediction = self.model(test_input.narrow(0, sample_idx, batch_size))
            test_pred.append(prediction.cpu())

        test_pred = torch.cat(test_pred, dim=0)
        return test_pred

    def epoch_end(self, stats, epoch, train_loss, val_input, val_target, batch_size=100):
        print("\rTesting on validation set...", end='')
        val_loss, val_psnr = self.eval(val_input, val_target, batch_size)

        self.scheduler.step(val_loss)

        stats['train_loss'].append(train_loss)
        stats['val_loss'].append(val_loss)
        stats['val_psnr'].append(val_psnr)

        # Save model
        torch.save(self.model.cpu(), 'bestmodel_epoch{}.pth'.format(epoch))

        # Print epoch result
        print(f"[Train Loss: {train_loss:4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB]")

    def eval(self, val_input, val_target, batch_size=100):
        if self.params.cuda and torch.cuda.is_available():
            val_input = val_input.cuda()
            val_target = val_target.cuda()

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
            nn.Conv2d(96 + in_channel, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channel, 3, padding=1, stride=1),
            nn.LeakyReLU(0.1)
        )

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
        up2 = self._decode1(cat4)
        cat3 = torch.cat((up2, pool2), dim=1)
        up3 = self._decode1(cat3)
        cat2 = torch.cat((up3, pool1), dim=1)
        up4 = self._decode1(cat2)
        cat1 = torch.cat((up4, x), dim=1)
        output = self._decode2(cat1)

        return output
