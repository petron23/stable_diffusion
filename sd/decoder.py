import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.module):
    def __init__(self, channels: int):
        super().__init__
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        n, c, h, w = x.shape


        #(batch, channel/feature, height, width) -> (bacth, height*width, feat)
        x = x.view(n,c, h*w)
        
        #(batch, channel/feature, height, width) -> (bacth, feat, height*width)
        x = x.transpose(-1, -2)

        #(batch, feat, height*width) -> (batch, feat, height, width)
        x = x.view((n,c, h,w))

        x +=residue
        return x


class VAE_ResidualBlock(nn.module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, height, width)
        
        # skip connection
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x =self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

    