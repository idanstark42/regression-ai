import sys
import time
from src.data.dataset import FunctionsDataset
from src.model.main import RegressionModel

from commands.train import train_module
from commands.detect import detect

from src.utils import modelfile_path

DATASET_SIZE = 500000
N_POINTS = 10
FAMILIES = ['constant', 'linear', 'quadratic']

MODELS = {
  'default': lambda: RegressionModel(N_POINTS, FAMILIES),
}

if __name__ == '__main__':
  command = sys.argv[1]
  start = time.time()
  dataset = FunctionsDataset(DATASET_SIZE, N_POINTS, FAMILIES)

  if command == 'train':
    module = MODELS[sys.argv[2]]() if len(sys.argv) > 2 else MODELS['default']()
    output = modelfile_path(sys.argv[3]) if len(sys.argv) > 3 else modelfile_path('model_' + str(round(time.time() * 1000)))
    train_module(dataset, module, output)
    exit()

  # if command == 'detect':
  #   module = MODELS[sys.argv[2] or 'default']()
  #   model_file = modelfile_path(sys.argv[2])
  #   params = sys.argv[4:]
  #   detect(dataset, module, model_file)
  #   exit()

  print(f'Unknown command: {command}')