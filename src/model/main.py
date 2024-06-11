import torch
from torch import nn

from src.data.factory import FAMILIES

# This module recieves an array of n points and returns an array representing the type of functions that can be fitted to the points and their parameters
class RegressionModel (nn.Module):
  def __init__(self, n_points, families):
    super(RegressionModel, self).__init__()

    self.linear_layers = nn.ModuleList([
      nn.Linear(2 * n_points, 128, dtype=torch.float64),
      nn.ReLU(),
      nn.Linear(128, 64, dtype=torch.float64),
      nn.ReLU(),
      nn.Linear(64, 64, dtype=torch.float64),
      nn.ReLU(),
      nn.Linear(64, sum([len(FAMILIES[family]().arguments()) + 1 for family in families]), dtype=torch.float64)
    ])

    # count and print the number of parameters in the network
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Initialized NN with {} trainable parameters".format(trainable_params))

  def forward(self, x):
    for layer in self.linear_layers:
      x = layer(x)

    return x
