import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_hiddens: int,
                 num_residual_hiddens: int):
        super(Residual, self).__init__()
        self.case = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, 
                x: torch.Tensor
    ) -> torch.Tensor:
        x += self.case(x)
        return x


class ResidualCase(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int):
        super(ResidualCase, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([Residual(in_channels, 
                                              num_hiddens, 
                                              num_residual_hiddens)
                                    for _ in range(self.num_residual_layers)])
        self.activation = nn.LeakyReLU(0.2)

    def forward(self,
                x: torch.Tensor
    ) -> torch.Tensor:
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        x = self.activation(x)
        return x
