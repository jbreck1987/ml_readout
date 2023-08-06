"""
Model definitions to be used in runs.
"""

# Author: Josh Breckenridge
# Data: 7-21-2023

import torch
    
class BranchedConvReg(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=5) , 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5), 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2), 
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Dropout()
        )
        self.arrival_regression = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*5, out_features=100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=100, out_features=1)
        )
        self.height_regression = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*5, out_features=100),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=100, out_features=1)
        )

    
    def forward(self, x) -> torch.Tensor:
        # Want to keep the parameters in the arrival and height FC brances separate from each other,
        # the two graphs will be merged in the feature extractor
        out_height = self.height_regression(self.feature_extractor(x))
        out_arrival = self.arrival_regression(self.feature_extractor(x))
        return out_arrival, out_height
