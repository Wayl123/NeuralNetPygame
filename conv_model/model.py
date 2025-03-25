import torch
import torch.nn as nn
import torch.optim as optim
import os
from helper import uniquify
import numpy as np

class CNN(nn.Module):
  def __init__(self, output_size):
    super().__init__()

    self.convolution_layers = nn.Sequential(
      # input 128 x 128 (h x w)
      nn.Conv2d(3, 8, kernel_size = 5, padding = 2),
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2),
      # 64 x 64
      nn.Conv2d(8, 16, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2),
      # 32 x 32
      nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2),
      # 16 x 16
    )

    self.fully_connected_layers = nn.Sequential(
      nn.Linear(32 * 16 * 16, 128),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(64, output_size),
      nn.Tanh()
    )

    self.record = nn.Parameter(torch.tensor(0, dtype = torch.int), False)

  def forward(self, x):
    x = self.convolution_layers(x)
    # Flatten
    x = x.view(x.size(0), -1)
    x = self.fully_connected_layers(x)
    return x
  
  def save(self, file_name = "model.pth"):
    model_folder_path = os.path.join(os.path.dirname(__file__), "model")
    if not os.path.exists(model_folder_path):
      os.makedirs(model_folder_path)

    file_name = os.path.join(model_folder_path, file_name)
    torch.save(self.state_dict(), file_name)

  def load(self, file_name = "model.pth"):
    model_folder_path = os.path.join(os.path.dirname(__file__), "model")
    file_name = os.path.join(model_folder_path, file_name)
    if os.path.exists(file_name):
      self.load_state_dict(torch.load(file_name))
      self.eval()

      os.rename(file_name, uniquify(file_name))

  def load_record(self, file_name = "model.pth"):
    model_folder_path = os.path.join(os.path.dirname(__file__), "model")
    file_name = os.path.join(model_folder_path, file_name)
    if os.path.exists(file_name):
      self.record = nn.Parameter(torch.load(file_name)["record"], False)

class Trainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
    self.loss_fn = nn.MSELoss()

  # Old state, action taken, reward gained, new state after action, game over
  def train_step(self, state, action, reward, next_state, done):
    state = torch.from_numpy(state)
    action = torch.tensor(action, dtype = torch.bool)
    reward = torch.tensor(reward, dtype = torch.float)
    next_state = torch.from_numpy(next_state)

    # If 1d array of size x, change to 2d array of size (1, x)
    if len(action.shape) == 1:
      state = state.unsqueeze(0)
      action = action.unsqueeze(0)
      reward = reward.unsqueeze(0)
      next_state = next_state.unsqueeze(0)
      done = (done, )

    for idx in range(len(done)):
      # 1: predicted Q values with current state
      Q_value = self.model(state[idx].unsqueeze(0))
      Q_value = Q_value.squeeze()
      Q_new_value = Q_value.clone()
      # 2: Q_new = reward + gamma * max(next_predicted Q value) -> only do this if not done
      Q_new = reward[idx]
      if not done[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))

      Q_new_value[action[idx]] = Q_new

      self.optimizer.zero_grad()
      loss = self.loss_fn(Q_value, Q_new_value)
      loss.backward()

      self.optimizer.step()


