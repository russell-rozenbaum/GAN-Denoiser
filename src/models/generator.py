import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DenoisingAE(nn.Module):
    """
    A simple encoder-decoder for denoising audio. 
    
    This serves as the generator in the GAN, and ultimately as the 
    trained denoiser itself when detached from the GAN.
    
    Inputs:  (N, in_channels, T)   - T audio samples, N batch size
    Outputs: (N, out_channels, T) - same temporal dimension T as the input
    """
    def __init__(
        self,
        # This can be adjusted for multichannel input in other cases, 
        # eg. stereo input from bioacoustic recording device
        in_channels: int = 1,
        # Latent space size
        base_channels: int = 32,
        # We actually just want this to match our in_channels (most likely)
        out_channels: int = 1,
        signal_length: int = 1024, # i.e. sample rate * lowest duration of audio
        num_enc_filters: int = 64,
        num_dec_filters: int = 128,
        filter_size: int = 128,
        stride: int = 2,
        num_downsamples: int = 5,
        num_upsamples: int = 5,
        leakiness: float = 0.3,
        gamma=1,
        rho=1,
    ):
        super().__init__()

        # Adversarial and Reconstruction Loss Multipliers, respectively
        self.gamma=gamma
        self.rho=rho

        # --------------- |
        #   Encoder Block |
        # --------------- v

        self.encoder_layers = nn.ModuleList()
        self.skip_connections = []

        for _ in range(num_downsamples):
            self.encoder_layers.append(
                nn.Conv1d(
                in_channels, 
                num_enc_filters, 
                filter_size,
                stride=stride,
                padding=(filter_size // stride) - 1)
                )
            self.encoder_layers.append(nn.BatchNorm1d(num_enc_filters))
            self.encoder_layers.append(nn.LeakyReLU(leakiness, inplace=True))
            in_channels = num_enc_filters
        
        self.encoder = nn.Sequential(*self.encoder_layers)

        # --------------- ^
        #   Encoder Block |
        # --------------- |

        # --------------- |
        #   Decoder Block |
        # --------------- v

        self.decoder_layers = nn.ModuleList()
        in_channels = num_enc_filters  # Start from latent space size

        for _ in range(num_upsamples):
            self.decoder_layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_channels, 
                    out_channels=num_dec_filters,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=(filter_size // stride) - 1,
                    output_padding=0
                )
            )
            self.decoder_layers.append(nn.BatchNorm1d(num_dec_filters))
            self.decoder_layers.append(nn.LeakyReLU(leakiness, inplace=True))
            in_channels = num_dec_filters

        # Core of decoder block
        self.decoder = nn.Sequential(*self.decoder_layers)
        
        # Final head of decoder block, to output the (hopefully) clean signal
        self.final_layer = nn.Conv1d(
            in_channels=num_dec_filters, 
            out_channels=out_channels, 
            kernel_size=filter_size, 
            padding=(filter_size // 2)
        )

        # --------------- ^
        #   Decoder Block |
        # --------------- |

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
        # --------------- |
        #   Encoder Block |
        # --------------- v
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i % 3 == 0:  # Every Conv layer
                skip_connections.append(x)  # Save the feature map for skip connection

        # --------------- ^
        #   Encoder Block |
        # --------------- |

        # x is now a latent space representation

        # --------------- |
        #   Decoder Block |
        # --------------- v
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            # Add skip connections
            if i % 3 == 0 and len(self.skip_connections) > 0:  # Only at ConvTranspose layers
                skip = self.skip_connections.pop()  # Pop the corresponding skip connection
                x = x + skip  # Add skip connection to the feature map
        # --------------- ^
        #   Decoder Block |
        # --------------- |

        # Final output layer to restore to the original/desired signal dimension
        x = self.final_layer(x)
        x = x[..., :1024]
        # TODO: Find a more elegant mathematical mapping so we don't have to cut-off data

        return x