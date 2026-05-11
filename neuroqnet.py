import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from collections import deque
import random
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from game_environment import MarbleGame

LOAD_MODEL = False
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
RUN_AMOUNT = 16384
GAME_PROCESSES = 4
RAY_CAST_COUNT = 32

# Model
class Linear_QNet(nn.Module):
  def __init__(self, obs_length, act_length, bias = True, **kwargs):
    super().__init__()
    
    self.linear_1_layers = nn.Sequential(
      nn.Linear(obs_length, act_length, bias = bias),
      nn.Tanh()
    )

    self.linear_2_layers = nn.Sequential(
      nn.Linear(obs_length, 64, bias = bias),
      nn.LeakyReLU(),
      nn.Linear(64, act_length, bias = bias),
      nn.Tanh()
    )

    self.linear_3_layers = nn.Sequential(
      nn.Linear(obs_length, 64, bias = bias),
      nn.LeakyReLU(),
      nn.Linear(64, 32, bias = bias),
      nn.LeakyReLU(),
      nn.Linear(32, act_length, bias = bias),
      nn.Tanh()
    )

    self.record = nn.Parameter(torch.tensor(0, dtype = torch.float64), False)

  def forward(self, x):
    x = self.linear_1_layers(x)

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

class Agent:
  def __init__(self):
    self.n_games = 1
    self.gamma = 0.9 # Discount rate
    self.memory = deque(maxlen = MAX_MEMORY)
    self.model = Linear_QNet((RAY_CAST_COUNT) + 4, 7)
    self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)

  def remember(self, obs, action, reward, next_obs, done):
    self.memory.append((obs, action, reward, next_obs, done)) # Automatically discard oldest item when MAX_MEMORY is reached

  def train_long_memory(self):
    # Grab a batch, if not enough memory for a batch grab entire memory
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    else:
      mini_sample = self.memory

    obss, actions, rewards, next_obss, dones = zip(*mini_sample)
    self.trainer.train_step(np.array(obss), np.array(actions), np.array(rewards), np.array(next_obss), np.array(dones))

  def train_short_memory(self, obs, action, reward, next_obs, done):
    self.trainer.train_step(obs, action, reward, next_obs, done)

  def get_action(self, obs):
    prediction = self.model(torch.tensor(obs, dtype = torch.float))
    final_move = [move > 0 for move in prediction]

    return final_move
  
# Helper
def uniquify(path, sep = '_'):
  count = 1

  path = os.path.normpath(path)
  filename, ext = os.path.splitext(path)

  while os.path.exists(path):
    path = "{f}{s}{n:d}{e}".format(f = filename, s = sep, n = count, e = ext)
    count += 1

  return path

def plot(scores, mean_scores, result_file_name):
  result_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
  if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

  result_full_file_name = os.path.join(result_folder_path, result_file_name)

  display.clear_output(wait=True)
  display.display(plt.gcf())
  plt.clf()
  plt.title('Training')
  plt.xlabel('Number of Games')
  plt.ylabel('Score')
  plt.plot(scores)
  plt.plot(mean_scores)
  plt.ylim(ymin = 0)
  plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
  plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
  plt.show(block=False)
  plt.pause(.1)
  plt.savefig(result_full_file_name)

# Training
def render(agent):
  game = MarbleGame()
  state = game.reset()
  obs = game.get_state_observation(state)
  images = [game.render(state)]
  step = 0
  score = 0
  done = False

  while not done:
    # Get move
    action = agent.get_action(obs)

    # Perform move and get new state
    state, _, done = game.step(action)
    new_obs = game.get_state_observation(state)

    obs = new_obs
    step += 1

    # Save image every 4 step
    if step % 4 == 0:
      images.append(game.render(state))

    if done:
      scores = state[5]
      score = sum(scores)

  file_name = "result_{score}.gif".format(score = score)
  gif_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
  if not os.path.exists(gif_folder_path):
    os.makedirs(gif_folder_path)

  gif_file = os.path.join(gif_folder_path, file_name)
  images[0].save(gif_file, save_all = True, append_images = images[1:], duration = 60, loop = 0)

def train_game(agent, record, plot_data):
  game = MarbleGame()
  state = game.reset()
  obs = game.get_state_observation(state)
  images = [game.render(state)]
  step = 0
  
  while agent.n_games < RUN_AMOUNT:
    # Get move
    action = agent.get_action(obs)

    # Perform move and get new state
    state, reward, done = game.step(action)
    new_obs = game.get_state_observation(state)

    # Train short memory
    agent.train_short_memory(obs, action, reward, new_obs, done)

    # Remember
    agent.remember(obs, action, reward, new_obs, done)

    obs = new_obs
    step += 1

    # Save image every 4 step
    if step % 4 == 0:
      images.append(game.render(state))

    if done:
      score = state[6]

      if score > record.value:
        record.value = score
        agent.model.record = nn.Parameter(torch.tensor(score, dtype = torch.int), False)
        agent.model.save()

        file_name = "test_result_{process_num}_{game_num}.gif".format(process_num = mp.current_process().name, game_num = agent.n_games)
        gif_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
        if not os.path.exists(gif_folder_path):
          os.makedirs(gif_folder_path)

        gif_file = os.path.join(gif_folder_path, file_name)
        images[0].save(gif_file, save_all = True, append_images = images[1:], duration = 60, loop = 0)

      print("Process", mp.current_process().name, "Game", agent.n_games, "Score", score, "Record", record.value)

      plot_data["plot_scores"].append(score)
      plot_data["total_score"] += score
      mean_score = plot_data["total_score"] / agent.n_games
      plot_data["plot_mean_scores"].append(mean_score)

      # Train long memory
      state = game.reset()
      obs = game.get_state_observation(state)
      agent.n_games += 1
      agent.train_long_memory()
      images = [game.render(state)]

  if len(plot_data["plot_scores"]) > 0:
    plot(plot_data["plot_scores"], plot_data["plot_mean_scores"], "process_{process_num}_training_plot.png".format(process_num = mp.current_process().name))
  # render(agent)

def train():
  load_saved_model = LOAD_MODEL
  agent = Agent()
  record = mp.Value("f", 0)

  plot_data = {
    "total_score": 0,
    "plot_scores": [],
    "plot_mean_scores": []
  }

  print("start: {time}".format(time = datetime.datetime.now()))

  processes = []

  if load_saved_model:
    agent.model.load()
  else:
    agent.model.load_record()

  record.value = agent.model.record.data.item()

  agent.model.share_memory()

  for _ in range(GAME_PROCESSES):
    p = mp.Process(target = train_game, args = (agent, record, plot_data))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

  print("end: {time}".format(time = datetime.datetime.now()))

if __name__ == '__main__':
  train()