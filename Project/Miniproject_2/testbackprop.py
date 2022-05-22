import torch
import torch.nn as nn
from others.utils import load_data
from others.modules import *

torch.manual_seed(0)

criterion = nn.MSELoss()

tr1, tr2, val_n, val_c = load_data("data/", "data/")
print(tr1.shape, tr2.shape, val_n.shape, val_c.shape)

test1 = torch.clone(tr1[:3, :, :, :]).float()
test2 = torch.clone(tr2[:3, :, :, :]).float()
print(test1.shape, test2.shape)

torchconv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2)
testconv = Conv2d(3, 32, kernel_size=(3, 3), stride=2)

output1 = torchconv(test1)
output2 = testconv.forward(test1)

torch.testing.assert_allclose(torchconv.weight, testconv.w)

# loss1 = criterion(test2, output1)
# loss2 = criterion(test2, output2)
torch.testing.assert_allclose(output1, output2)
