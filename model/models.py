import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from encoder import Encoder
from VQ import VectorQuantizerEMA
from decoder import Decoder
from sampler import Sampler

class VQVAE(nn.Module):
    def __init__(self,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float,
                 image_channels: int = 3,
                 decay: float = 0.25):
        super(VQVAE, self).__init__()
        
        self._encoder = Encoder(image_channels,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        
        self._vq_vae = VectorQuantizerEMA(num_embeddings,
                                          embedding_dim,
                                          commitment_cost,
                                          decay)
        
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
    def forward(self,
                x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, 
                  torch.Tensor, 
                  torch.Tensor]:
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

class VAE(nn.Module):
    def __init__(self,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int,
                 embedding_dim: int,
                 image_channels: int = 3):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(image_channels,
                                num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        
        self.sample_vector = Sampler(num_hiddens,
                                     embedding_dim,
                                     num_hiddens * 4 * 4)

        self.decoder = Decoder(num_hiddens,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        
    def forward(self,
                x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, 
                  torch.Tensor, 
                  torch.Tensor]:
        z = self.encoder(x)
        z = self.sample_vector(z)
        x_recon = self.decoder(z)
        return x_recon