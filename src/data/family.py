import numpy as np

class Family:
  def arguments(self):
    return []

  def eval(x, args):
    # unimplemented
    return 0

class Polynomial(Family):
  def __init__(self, degree):
    self.degree = degree

  def arguments(self):
    return [f'a{i}' for i in range(self.degree + 1)]

  def eval(self, x, args):
    return sum([args[f'a{i}'] * (x ** i) for i in range(self.degree + 1)])

class InversePolynomial(Polynomial):
  def eval(self, x, args):
    return sum([args[f'a{i}'] * (x ** (self.degree - i)) for i in range(self.degree + 1)])

class Harmonic(Family):
  def arguments(self):
    return ['amplitude', 'angular_frequency', 'phase']
  
  def eval(self, x, args):
    return args['amplitude'] * np.sin(args['angular_frequency'] * x + args['phase'])

class Exponential(Family):
  def arguments(self):
    return ['base', 'exponent']
  
  def eval(self, x, args):
    return args['base'] ** (args['exponent'] * x)
  
class Logarithmic(Family):
  def arguments(self):
    return ['factor', 'base']
  
  def eval(self, x, args):
    return np.log(x * args['factor']) / np.log(args['base'])
  
class Gaussian(Family):
  def arguments(self):
    return ['amplitude', 'mean', 'std_dev']
  
  def eval(self, x, args):
    return args['amplitude'] * np.exp(-((x - args['mean']) ** 2) / (2 * args['std_dev'] ** 2))
  
class Step(Family):
  def arguments(self):
    return ['amplitude', 'step']
  
  def eval(self, x, args):
    return args['amplitude'] if x > args['step'] else 0
  
class Sigmoid(Family):
  def arguments(self):
    return ['amplitude', 'slope']
  
  def eval(self, x, args):
    return args['amplitude'] / (1 + np.exp(-args['slope'] * x))
  