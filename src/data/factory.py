import numpy as np
from src.data.family import Polynomial, InversePolynomial, Harmonic, Exponential, Logarithmic, Gaussian, Step, Sigmoid
from src.data.func import Function

FAMILIES = {
  'constant': lambda: Polynomial(0),
  'linear': lambda: Polynomial(1),
  'quadratic': lambda: Polynomial(2),
  'cubic': lambda: Polynomial(3),
  'quartic': lambda: Polynomial(4),
  'quintic': lambda: Polynomial(5),
  'sextic': lambda: Polynomial(6),
  'septic': lambda: Polynomial(7),
  'octic': lambda: Polynomial(8),
  'inverse_linear': lambda: InversePolynomial(1),
  'inverse_quadratic': lambda: InversePolynomial(2),
  'inverse_cubic': lambda: InversePolynomial(3),
  'inverse_quartic': lambda: InversePolynomial(4),
  'inverse_quintic': lambda: InversePolynomial(5),
  'inverse_sextic': lambda: InversePolynomial(6),
  'inverse_septic': lambda: InversePolynomial(7),
  'inverse_octic': lambda: InversePolynomial(8),
  'harmonic': Harmonic,
  'exponential': Exponential,
  'logarithmic': Logarithmic,
  'gaussian': Gaussian,
  'step': Step,
  'sigmoid': Sigmoid,
}

class FunctionFactory:
  def __init__(self, families):
    self.families = { name: FAMILIES[name] for name in families }

  def create(self, name, args):
    return Function(name, self.families[name](), args)
  
  def random(self):
    name = np.random.choice(list(self.families.keys()))
    family = self.families[name]()
    args = { arg: np.random.rand() for arg in family.arguments() }
    return self.create(name, args)