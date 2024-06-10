import torch
from torch import nn
from torch.nn import functional as F

class RegressionModel (nn.Module):
  def __init__(self):
    super(RegressionModel, self).__init__()

    # count and print the number of parameters in the network
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Initialized NN with {} trainable parameters".format(trainable_params))

  def forward(self, clusters, tracks):
    for layer in self.cluster_layers:
      clusters = layer(clusters)

    clusters = clusters.view(-1, 16 * (self.resolution // 4) * (self.resolution // 4))
    tracks = tracks.view(-1, 16 * (self.resolution // 4) * (self.resolution // 4))    

    x = torch.cat([clusters, tracks], dim=1)

    for layer in self.linear_layers:
      x = layer(x)

    return x.view(-1, self.resolution, self.resolution)
