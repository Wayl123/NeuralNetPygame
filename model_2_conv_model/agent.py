import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import random
from collections import deque
from pygame_environment import MarbleGameManager, MarbleGame
from model import CNN, Trainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
MAX_GAME = 256
GAME_PROCESSES = 4

class Agent:
  def __init__(self):
    self.n_games = 1
    self.gamma = 0.9 # Discount rate
    self.memory = deque(maxlen = MAX_MEMORY)
    self.model = CNN(7)
    self.trainer = Trainer(self.model, lr = LR, gamma=self.gamma)

  def get_state(self, game):
    state = game.get_state() # width x height x dimension(channel size)
    # 512 x 512
    bin_size = 4
    state = state.reshape((state.shape[0]//bin_size, bin_size, state.shape[1]//bin_size, bin_size, 3)).mean(3).mean(1)
    # 128 x 128
    state = state.transpose((2, 1, 0)) # channel x height x width

    return state

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done)) # Automatically discard oldest item when MAX_MEMORY is reached

  def train_long_memory(self):
    # Grab a batch, if not enough memory for a batch grab entire memory
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    else:
      mini_sample = self.memory

    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

  def train_short_memory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)

  def get_action(self, state):
    tensor_image = torch.from_numpy(state) 
    tensor_image = tensor_image.unsqueeze(0) # batch size x channel x height x width
    prediction = self.model(tensor_image)
    final_move = [move > 0 for move in prediction.tolist()[0]]

    return final_move

def train_game(agent, record):
  plot_scores = []
  plot_mean_scores = []
  total_score = 0
  game_manager = MarbleGameManager()
  game_manager.scene = MarbleGame(game_manager)
  game_manager.running = True

  while agent.n_games < MAX_GAME:
    # Get old state
    state = agent.get_state(game_manager)

    # Get move
    final_move = agent.get_action(state)

    # Perform move and get new state
    reward, done = game_manager.game_step(final_move)
    next_state = agent.get_state(game_manager)

    # Train short memory
    agent.train_short_memory(state, final_move, reward, next_state, done)

    # Remember
    agent.remember(state, final_move, reward, next_state, done)

    if done:
      score = game_manager.scene.time_score + game_manager.scene.enemy_score

      if score > record.value:
        record.value = score
        agent.model.record = nn.Parameter(torch.tensor(score, dtype = torch.int), False)
        agent.model.save()

      print("Process", mp.current_process().name, "Game", agent.n_games, "Score", score, "Record", record.value)

      plot_scores.append(score)
      total_score += score
      mean_score = total_score / agent.n_games
      plot_mean_scores.append(mean_score)

      # Train long memory
      game_manager.running = True
      game_manager.scene.reset()
      agent.n_games += 1
      agent.train_long_memory()

  plot(plot_scores, plot_mean_scores)

def train():
  load_saved_model = True
  agent = Agent()
  record = mp.Value("i", 0)

  processes = []

  if load_saved_model:
    agent.model.load()
  else:
    agent.model.load_record()

  record.value = agent.model.record.data.item()

  agent.model.share_memory()

  for _ in range(GAME_PROCESSES):
    p = mp.Process(target = train_game, args = (agent, record))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

if __name__ == '__main__':
  train()