import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DenoisingAE(nn.Module):
    """
    A simple encoder-decoder for denoising audio. 
    
    This serves as the
    generator in the GAN, and ultimately as the trained denoiser itself
    when detached from the GAN.
    
    Inputs:  (N, in_channels, T)   - T audio samples, N batch size
    Outputs: (N, out_channels, T) - same temporal dimension T as the input
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        sample_rate: int = 1024,
        num_filters: int = 64,
        filter_size: int = 32,
        num_downsamples: int = 5,
        leakiness: float = 0.3,
    ):
        super().__init__()

        # ---------------
        #   Encoder Block
        # ---------------

        # Conv block #1
        self.enc_conv1 = nn.Conv1d(
            in_channels,       # e.g. 1 if mono audio
            base_channels,     # e.g. 32
            kernel_size,
            stride=2,          # reduce temporal dimension by factor of 2
            padding=(kernel_size // 2),
        )
        self.enc_bn1 = nn.BatchNorm1d(base_channels)

        # Conv block #2
        self.enc_conv2 = nn.Conv1d(
            base_channels,
            base_channels * 2,
            kernel_size,
            stride=2,          # reduce temporal dimension by factor of 2 again
            padding=(kernel_size // 2),
        )
        self.enc_bn2 = nn.BatchNorm1d(base_channels * 2)



        # ---------------
        #   Decoder Block
        # ---------------

        # Deconv block #1
        self.dec_conv1 = nn.ConvTranspose1d(
            base_channels * 2,
            base_channels,
            kernel_size,
            stride=2,  
            padding=(kernel_size // 2),
            output_padding=1  # helps align the shapes after strided conv
        )
        self.dec_bn1 = nn.BatchNorm1d(base_channels)

        # Deconv block #2
        self.dec_conv2 = nn.ConvTranspose1d(
            base_channels,
            out_channels,
            kernel_size,
            stride=2,  
            padding=(kernel_size // 2),
            output_padding=1
        )
        # out_channels = 1 for mono audio output
        # Typically you can do a final activation (e.g. Tanh) if you want 
        # your samples bounded between -1 and 1

        self.init_weights()

    def init_weights(self):
        # A quick initialization example
        torch.manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                nn.init.normal_(
                    module.weight, 
                    mean=0.0, 
                    std=sqrt(1.0 / (module.in_channels * (module.kernel_size[0])))
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the network.
        x shape = (N, in_channels, T)
        """

        # ---------------
        #   Encoder Block
        # ---------------
        e1 = F.relu(self.enc_bn1(self.enc_conv1(x)))  # (N, base_channels, T/2)
        e2 = F.relu(self.enc_bn2(self.enc_conv2(e1))) # (N, base_channels*2, T/4)

        # Note: e2 is out latent representation!

        # ---------------
        #   Decoder Block
        # ---------------
        d1 = F.relu(self.dec_bn1(self.dec_conv1(e2))) # (N, base_channels, T/2)
        d2 = self.dec_conv2(d1)                       # (N, out_channels, T)

        # Optionally add skip connections:
        # For instance, if you want a skip from e1 -> d1 or from x -> d2, you can 
        # combine them with something like `torch.cat(...)` or a residual add.

        # If you want the output in [-1, +1], you can do a final Tanh:
        # out = torch.tanh(d2)
        # return out

        return d2
