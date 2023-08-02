"""
Model definitions to be used in runs.
"""

# Author: Josh Breckenridge
# Data: 7-21-2023

import torch


class ConvRegv1(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=5) , 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5), 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2), 
            torch.nn.LeakyReLU(),
            torch.nn.Flatten()
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*5, out_features=100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=100, out_features=1)
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x)))).unsqueeze(dim=1)

class ConvRegv2(torch.nn.Module):
    def __init__(self, in_channels, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=5) , 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5), 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2), 
            torch.nn.LeakyReLU(),
            torch.nn.Flatten()
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*5, out_features=100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=100, out_features=2)
        )
    
    def forward(self, x) -> torch.Tensor:
         return self.block4(self.block3(self.block2(self.block1(x)))).unsqueeze(dim=1)
