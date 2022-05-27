import torch
import torch.nn as nn
import torch.optim as optim
from .others.utils import StatsTracer, zero_one_norm
from .others.denoiser import Denoiser
from pathlib import Path
#from tensorboardX import SummaryWriter


class Model ():
    def __init__ ( self ) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.model = Denoiser(in_channels=3)
        self.batch_size = 100
        self.loss = nn.MSELoss()
        #self.loss = nn.L1Loss()
        self.optim = optim.Adam(self.model.parameters(),lr=0.001,betas=(0.9, 0.99),eps=1e-08)
        #self.optim = optim.SGD(self.model.parameters(),lr=0.0005, weight_decay=1e-5, momentum =0.99, nesterov =True)
        # Check GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

    def load_pretrained_model ( self ) -> None :
        ## This loads the parameters saved in bestmodel .pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))

    def train (self , train_input , train_target , num_epochs) -> None :
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the
        #same images , which only differs from the input by their noise.
        self.model.train(True)
        #writer = SummaryWriter(log_dir='./runs/L1Loss_Adam_0.001')
        train_input = zero_one_norm(train_input)
        train_target = zero_one_norm(train_target)
        train_input = train_input.float().to(self.device)
        train_target = train_target.float().to(self.device)

        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)
        #start.record()
        # Main training loop
        for epoch in range(num_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, num_epochs))
            tr_loss = StatsTracer()
            counter =0
            # Minibatch
            for sample_idx in range(0, train_input.size(0), self.batch_size):
                source_denoised = self.model(train_input.narrow(0, sample_idx, self.batch_size))
                loss = self.loss(source_denoised, train_target.narrow(0, sample_idx, self.batch_size))
                tr_loss.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                counter = counter+1

            print("Epoch: {a}/{b}, Training Loss: {c}".format(a=epoch + 1, b=num_epochs, c=tr_loss.avg))
            #writer.add_scalar('trainning loss (iteration)', tr_loss.avg, (epoch + 1) * train_input.size(0)/(self.batch_size))
            tr_loss.reset()
        #end.record()
        #torch.cuda.synchronize()
        #print(start.elapsed_time(end)/1000/60," minutes")

        #torch.save(self.model.state_dict(), '{}_epoch{}.pth'.format('L1Loss_Adam_0.001', epoch))
        print("Model checkpoint saved...")
            #self.scheduler.step()



    def predict (self , test_input ) -> torch . Tensor :
        #: test_input : tensor of size (N1 , C, H, W) with values in range 0 -255 that has to
        #be denoised by the trained or the loaded network .
        ##: returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.
        self.model.train(False)
        test_input = zero_one_norm(test_input)
        test_input=test_input.float().to(self.device)

        source_denoised = self.model(test_input)
        #print(source_denoised.min()*255,source_denoised.max()*255)
        source_denoised = (source_denoised-source_denoised.min())/(source_denoised.max()-source_denoised.min())*255

        return  source_denoised

