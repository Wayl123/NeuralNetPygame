import torch
import torch.nn as nn
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import PGPE
from evotorch.logging import PandasLogger
import numpy as np
from collections import deque
from game_environment import MarbleGame

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
MAX_GAME = 256
GAME_PROCESSES = 1

class LinearNetwork(nn.Module):
  def __init__(self, obs_len, act_len, bias = True, **kwargs):
    super().__init__()
    self.linear = nn.Linear(obs_len, act_len, bias = bias)

  def forward(self, obs):
    return self.linear(obs)

def train_game(network : torch.nn.Module):
  game = MarbleGame()
  state = game.reset()
  obs = game.get_state_observation(state)
  # images = [game.render(state)]
  step = 0
  score = 0
  done = False
  
  while not done:
    # Get move
    prediction = network(torch.tensor(obs, dtype = torch.float))
    action = [move > 0 for move in prediction]

    # Perform move and get new state
    state, done = game.step(action)
    new_obs = game.get_state_observation(state)

    obs = new_obs
    step += 1

    # # Save image every 4 step
    # if step % 4 == 0:
    #   images.append(game.render(state))

    if done:
      scores = state[5]
      score = sum(scores)

  return score

def train():
  # Set the NeuroEvolution Problem
  problem = NEProblem(
    objective_sense = "max",
    network = LinearNetwork((32 * 5) + 6, 7),
    network_eval_func = train_game,
    num_actors = 4
  )

  # Apply Searcher to the Problem
  searcher = PGPE(
    problem,
    popsize = 64,
    radius_init = 2.25,
    center_learning_rate = 0.2,
    stdev_learning_rate = 0.1
  )

  # Log result
  logger = PandasLogger(searcher)

  # Run searcher
  searcher.run(64)

  # Plot mean score
  logger.to_dataframe().mean_eval.plot()
  print(logger.to_dataframe())

if __name__ == '__main__':
  train()