from torch.utils.data import Dataset
import numpy as np

from src.data.factory import FunctionFactory, FAMILIES
from src.utils import *

class FunctionsDataset (Dataset):
  def __init__(self, size, families):
    super().__init__()
    self.size = size
    self.factory = FunctionFactory(families)
    self._cache = {}
    
    self.generate()

  def generate(self):
    self.items = [self.factory.random() for _ in range(self.size)]

  def __getitem__(self, index):
    if index in self._cache:
      return self._cache[index]

    function = self.items[index]
    input = function.random_points()
    target = self.argument_array(function)
    self._cache[index] = (input, target)
    return input, target

  def __len__(self):
    return len(self.raw_data['event'])
  
  def argument_array(self, function):
    families_args = [[1 if function.family_name == family_name else 0] + [function.args[arg_name] if function.family_name == family_name else 0 for arg_name in FAMILIES[family_name]().arguments()] for family_name in FAMILIES.keys()]
    return np.array([x for xs in families_args for x in xs])