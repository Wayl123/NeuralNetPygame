import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnFunc
import os
from helper import uniquify

class Linear_QNet(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    
    self.linear_layers = nn.Sequential(
      nn.Linear(input_size, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(128, output_size),
      nn.Tanh()
    )

    self.record = nn.Parameter(torch.tensor(0, dtype = torch.int), False)

  def forward(self, x):
    x = self.linear_layers(x)

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

      # os.rename(file_name, uniquify(file_name))

  def load_record(self, file_name = "model.pth"):
    model_folder_path = os.path.join(os.path.dirname(__file__), "model")
    file_name = os.path.join(model_folder_path, file_name)
    if os.path.exists(file_name):
      self.record = nn.Parameter(torch.load(file_name)["record"], False)

class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
    self.loss_fn = nn.MSELoss()

  # Old state, action taken, reward gained, new state after action, game over
  def train_step(self, state, action, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float)
    next_state = torch.tensor(next_state, dtype=torch.float)

    # If 1d array of size x, change to 2d array of size (1, x)
    if len(state.shape) == 1:
      state = torch.unsqueeze(state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      next_state = torch.unsqueeze(next_state, 0)
      done = (done, )

    # 1: predicted Q values with current state
    Q_value = self.model(state)
    Q_new_value = Q_value.clone()

    # 2: Q_new = reward + gamma * max(next_predicted Q value) -> only do this if not done
    for idx in range(len(done)):
      Q_new = reward[idx]
      if not done[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

      Q_new_value[idx][action[idx]] = Q_new

    self.optimizer.zero_grad()
    loss = self.loss_fn(Q_value, Q_new_value)
    loss.backward()

    self.optimizer.step()