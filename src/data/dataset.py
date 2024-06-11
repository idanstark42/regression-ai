from torch.utils.data import Dataset
import numpy as np

from src.data.factory import FunctionFactory, FAMILIES
from src.utils import *

class FunctionsDataset (Dataset):
  def __init__(self, size, n_points, families):
    super().__init__()
    self.n_points = n_points
    self.factory = FunctionFactory(families)
    self._cache = {}
    
    self.generate(size)

  def generate(self, size):
    self.items = [self.factory.random() for _ in range(size)]

  def __getitem__(self, index):
    if index in self._cache:
      return self._cache[index]

    function = self.items[index]
    input = function.random_points(self.n_points)
    target = self.argument_array(function)
    self._cache[index] = (input, target)
    return input, target

  def __len__(self):
    return len(self.items)
  
  def argument_array(self, function):
    families_args = [[1 if function.family_name == family_name else 0] + [function.args[arg_name] if function.family_name == family_name else 0 for arg_name in FAMILIES[family_name]().arguments()] for family_name in FAMILIES.keys() if family_name in self.factory.families.keys()]
    return np.array([x for xs in families_args for x in xs])