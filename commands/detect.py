import torch

def detect (dataset, module, model_file):
  module.load_state_dict(torch.load(model_file))
  module.eval()
  for index in range(len(dataset)):
    if index % 1000 == 0:
      print(f'event {index}')
    event = dataset.get_event(index)
    clusters = event.clusters_and_tracks_density_map(100)
    clusters = torch.tensor(clusters).unsqueeze(0).unsqueeze(0).float()
    prediction = module(clusters)
    print(prediction)