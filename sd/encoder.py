import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(3,128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128,256),
            VAE_ResidualBlock(256,256),
            nn.Conv2d(256,256, kernel_size=3, stride=2,padding=0),
            VAE_ResidualBlock(256,512),
            VAE_ResidualBlock(512,512),
            nn.Conv2d(512,512, kernel_size=3, stride=2,padding=0),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512,512),
            VAE_ResidualBlock(512,512),
            nn.GroupNorm(32,512),
            nn.SiLU(),
            nn.Conv2d(512,512, kernel_size=3, stride=2,padding=0),
            nn.Conv2d(8,8, kernel_size=1, padding=0),
            )
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channel, Height, Width)
        # noise: (Batch, Out_Channels, Height/8, Width/8)

        for module in self:
            #only padding to Right and Down
            if getattr(module, "stride", None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            x = module(x)

        # (Batch_Size, 8, Height/8, Width/8) -> two tensors with channels = 4
        mean, log_variance = torch.chunk(x,2, dim=1)
        variance = (torch.clamp(log_variance, -30, 20)).exp()
        stdev = variance.sqrt()
        
        #Z = N(0,1) -> N(mean, variance): X = mean + Z * stdev
        x = mean + stdev * noise
        
        #scale by a constant for with no added reason
        x *= 0.18215

        return x