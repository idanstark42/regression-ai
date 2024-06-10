import numpy as np

class Function:
  def __init__(self, family_name, family, args):
    self.family_name = family_name
    self.family = family
    self.args = args

  def eval(self, x):
    return self.family.eval(x, self.args)
  
  def random_points(self, n=100):
    x = np.random.rand(n)
    y = np.array([self.eval(xi) for xi in x])
    points = np.array([x, y]).T
    return points
  
  def __str__(self) -> str:
    return f'{self.family.__class__.__name__}({self.args})'