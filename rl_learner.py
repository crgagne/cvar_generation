import torch
import torch.nn as nn

class TD_Learner(nn.Module):

    def __init__(self, input_dim, num_quant, hidden_dim=256):
        nn.Module.__init__(self)
        self.num_quant = num_quant
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, num_quant)
        else: # simple linear layer
            self.layer1 = nn.Linear(input_dim, num_quant)

    def forward(self, x):
        if self.hidden_dim is not None:
            x = self.layer1(x)
            x = torch.tanh(x)
            x = self.layer2(x)
        else:
            x = self.layer1(x)
        return(x)



# loss function ...
