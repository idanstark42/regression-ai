import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split

import matplotlib.pyplot as plt

EPOCHS = 10
BATCH_SIZE = 32

TRAINING_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.2

def train_module(dataset, model, output):
  start_time = time.time()
  # Create the dataloaders
  train_loader, validation_loader, test_loader = init_dataloaders(dataset)
  print(f'training over {len(train_loader.dataset)} samples, validating over {len(validation_loader.dataset)} samples, testing over {len(test_loader.dataset)} samples')

  # Create the optimizer
  optimizer = Adam(model.parameters(), lr=0.001)

  # Create the loss function
  criterion = nn.MSELoss()

  # Train the model
  print('training')
  best_validation_loss = float('inf')
  best_model = None
  losses = []
  epoch_start_times = []
  for epoch in range(EPOCHS):
    epoch_start_times.append(time.time())
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    training_loss = train(train_loader, model, criterion, optimizer)
    validation_loss = validate(validation_loader, model, criterion)
    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      best_model = model.state_dict()
    losses.append((training_loss, validation_loss))

  # Load the best model
  model.load_state_dict(best_model)

  # Test the best model
  test_start_time = time.time()
  print('testing')
  test(test_loader, model, criterion)

  # Save the model
  torch.save(model.state_dict(), output)

  # print summary
  print(f'Time: {time.time() - start_time} (trainig: {sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)])}, testing: {time.time() - test_start_time})')
  print(f'Average Epoch Time: {sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]) / len(epoch_start_times)}')
  print(f'Best Validation Loss: {best_validation_loss}')

  # Plot the losses as a function of epoch
  plt.plot(range(EPOCHS), [loss[0] for loss in losses], label='Training Loss')
  plt.plot(range(EPOCHS), [loss[1] for loss in losses], label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

def init_dataloaders (dataset):
  train_size = int(len(dataset) * TRAINING_PERCENTAGE)
  validation_size = int(len(dataset) * VALIDATION_PERCENTAGE)
  test_size = len(dataset) - train_size - validation_size

  train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
  
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
  
  return train_loader, validation_loader, test_loader

# train the model
def train(train_loader, model, criterion, optimizer):
  total_loss = 0
  model.train()
  for batch_idx, (inputs, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(*inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    if batch_idx % 10 == 0:
      print(f'training [{batch_idx * BATCH_SIZE}/{len(train_loader.dataset)}'
            f'({100. * batch_idx / len(train_loader):.0f}%)]\t\tLoss: {loss.item() / BATCH_SIZE:.4f}')
  return total_loss / len(train_loader)

# validate the model
def validate(val_loader, model, criterion):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for batch_idx, (inputs, target) in enumerate(val_loader):
      output = model(*inputs)
      loss = criterion(output, target)
      total_loss += loss.item()
      if batch_idx % 10 == 0:
        print(f'validating [{batch_idx * BATCH_SIZE}/{len(val_loader.dataset)}'
              f'({100. * batch_idx / len(val_loader):.0f}%)]\t\tLoss: {loss.item() / BATCH_SIZE:.4f}')
  return total_loss / len(val_loader)

# test the model
def test(test_loader, model, criterion):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for batch_idx, (inputs, target) in enumerate(test_loader):
      output = model(*inputs)
      loss = criterion(output, target).item()
      total_loss += loss.item()
      if batch_idx % 10 == 0:
        print(f'testing [{batch_idx * BATCH_SIZE}/{len(test_loader.dataset)}'
              f'({100. * batch_idx / len(test_loader):.0f}%)]\t\tLoss: {loss.item():.4f}')
  print(f'\nTest set: Average loss: {total_loss / len(test_loader):.4f}\n')
