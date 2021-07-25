import torch
import torch.nn as nn
import typing as tp
from torch.autograd import Variable

class Sampler(nn.Module):
    def __init__(self,
                 input: int,
                 embedding_dim: int,
                 out: int):
        super(Sampler, self).__init__()

        self.mu = nn.Linear(input, embedding_dim)      # output = CNN embedding latent variables
        self.logvar = nn.Linear(input, embedding_dim) 


        self.fc1 = nn.Linear(embedding_dim, input)
        self.bn1 = nn.BatchNorm1d(input)
        self.fc2 = nn.Linear(input, out)
        self.bn2 = nn.BatchNorm1d(out)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self,
                z: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, 
                  torch.Tensor, 
                  torch.Tensor, 
                  torch.Tensor]:

        mu, logvar = self.mu(z), self.logvar(z)

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            mu = eps.mul(std).add_(mu)

        x = self.fc1(mu)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)#.view(.....)
        return x
