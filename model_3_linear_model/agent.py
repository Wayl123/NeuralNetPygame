import torch
import torch.nn as nn
import torch.multiprocessing as mp
import random
import numpy as np
from collections import deque
# from pygame_environment import MarbleGameManager, MarbleGame
from game_environment import MarbleGame
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
MAX_GAME = 256
GAME_PROCESSES = 1

class Agent:
  def __init__(self):
    self.n_games = 1
    self.gamma = 0.9 # Discount rate
    self.memory = deque(maxlen = MAX_MEMORY)
    self.model = Linear_QNet((32 * 5) + 6, 7)
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

def train_game(agent, record, plot_data):
  game = MarbleGame()
  state = game.reset()
  obs = game.get_state_observation(state)
  
  while agent.n_games < MAX_GAME:
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

    if done:
      score = state[5]

      if score > record.value:
        record.value = score
        agent.model.record = nn.Parameter(torch.tensor(score, dtype = torch.int), False)
        agent.model.save()

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

def render(agent):
  game = MarbleGame()
  obs = game.get_state_observation(game.state)

  images = []
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
      score = state[5]

  file_name = "result_{score}.gif".format(score = score)
  gif_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
  if not os.path.exists(gif_folder_path):
    os.makedirs(gif_folder_path)

  gif_file = os.path.join(gif_folder_path, file_name)
  images[0].save(gif_file, save_all = True, append_images = images[1:], duration = 60, loop = 0)

def train():
  load_saved_model = False
  agent = Agent()
  record = mp.Value("i", 0)

  plot_data = {
    "total_score": 0,
    "plot_scores": [],
    "plot_mean_scores": []
  }

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

  if len(plot_data["plot_scores"]) > 0:
    plot(plot_data["plot_scores"], plot_data["plot_mean_scores"])
  render(agent)

if __name__ == '__main__':
  train()