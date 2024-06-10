import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data'
MODELS_DIR = 'models'

def relative_position(position):
  return np.array([(position[0] + np.pi) / (2 * np.pi), (position[1] + 2.5) / 5])

def modelfile_path (name):
  # go to parent directory of this file, then go to the models directory and add pth suffix
  return os.path.join(os.path.dirname(os.path.dirname(__file__)), MODELS_DIR, name + '.pth')
