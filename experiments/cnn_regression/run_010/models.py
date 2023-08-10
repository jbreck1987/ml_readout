"""
Model definitions to be used in runs.
"""

# Author: Josh Breckenridge
# Data: 7-21-2023

import torch
    
class BranchedConvReg(torch.nn.Module):
    def __init__(self, in_channels, height_hidden_units) -> None:
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=5) , 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5), 
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2), 
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5)
        )
        self.arrival_regression = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*5, out_features=100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=100, out_features=1)
        )
        self.height_regression = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*5, out_features=height_hidden_units),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=height_hidden_units, out_features=height_hidden_units),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=height_hidden_units, out_features=height_hidden_units),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=height_hidden_units, out_features=height_hidden_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=height_hidden_units, out_features=height_hidden_units),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=height_hidden_units, out_features=height_hidden_units),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=height_hidden_units, out_features=1)
        )

    
    def forward(self, x) -> torch.Tensor:
        # Want to keep the parameters in the arrival and height FC brances separate from each other,
        # the two graphs will be merged in the feature extractor
        out_height = self.height_regression(self.feature_extractor(x))
        out_arrival = self.arrival_regression(self.feature_extractor(x))
        return out_arrival, out_height
