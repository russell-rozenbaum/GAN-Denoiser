import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Discriminator(nn.Module):
    """
    A discriminator network, tasked with binary classification between
    a true-clean signal and a faux-clean signal
    
    This serves as the
    generator in the GAN, and ultimately as the trained denoiser itself
    when detached from the GAN.
    
    Inputs:  (N, in_channels, T)   - T audio samples, N batch size
    Outputs: (N, out_channels, T) - same temporal dimension T as the input
    """
    def __init__(
        self,
        # This can be adjusted for multichannel input in other cases, 
        # eg. stereo input from bioacoustic recording device
        in_channels: int = 1,
        signal_length: int = 1024, # i.e. sample rate * lowest duration of audio
        num_convolutions: int = 5,
        num_filters: int = 64,
        filter_size: int = 64,
        stride: int = 2,
        leakiness: float = 0.3,
        num_fc_layers: int = 2,  # Number of fully connected layers
        fc_units: int = 128,     # Number of units in the fully connected layers
    ):
        super().__init__()

        # ---------------------- |
        #  Convolutional Blocks  |
        # ---------------------- v

        layers = []
        in_channels = in_channels

        for _ in range(num_convolutions):
            layers.append(
                nn.Conv1d(
                in_channels, 
                num_filters, 
                filter_size, 
                stride=stride,
                padding=filter_size // stride)
                )
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.LeakyReLU(leakiness, inplace=True))
            in_channels = num_filters
        
        self.conv_blocks = nn.Sequential(*layers)

        # ---------------------- ^
        #  Convolutional Blocks  |
        # ---------------------- |

        # ------------- |
        #   FC NN Block |
        # ------------- v

        self.fc_layers = nn.Sequential(
            nn.Linear(num_filters * (signal_length // (stride ** num_convolutions)), fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, 1),  # Output a single value
            nn.Sigmoid()  # To output a probability between 0 and 1
        )

        # ------------- ^
        #   FC NN Block |
        # ------------- |

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.normal_(
                    module.weight, 
                    mean=0.0, 
                    std=sqrt(1.0 / (module.in_channels * module.kernel_size[0]))
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
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
        # Pass through the convolutional layers
        x = self.conv_blocks(x)

        # Flatten the output of the convolutional layers to pass to fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers
        x = self.fc_layers(x)

        # Return prediction x
        return x
