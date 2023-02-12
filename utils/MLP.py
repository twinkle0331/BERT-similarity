import torch
import torch.nn as nn
import torch.nn.functional as F

# ResBlock  # <4>
# one group of convolutions, activation and skip connection
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(768, 768)
        self.fc4 = nn.Linear(768, 768)
        torch.nn.init.kaiming_normal_(self.fc1.weight,
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight,
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight,
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc4.weight,
                                      nonlinearity='relu')
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        out = self.gelu(out)
        out = self.fc3(out)
        out = self.gelu(out)
        out = self.fc4(out)
        return out