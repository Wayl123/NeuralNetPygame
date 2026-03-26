import torch
import torch.nn as nn
from evotorch.neuroevolution import GymNE
from evotorch.algorithms import PGPE, Cosyne
from evotorch.logging import PandasLogger
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os
import datetime
import numpy as np
from evotorch.decorators import pass_info
import pickle

LOAD_MODEL = False
POP_SIZE = 128
RUN_AMOUNT = 256
RAY_CAST_COUNT = 32

class BinaryActionWrapper(gym.ActionWrapper):
  def __init__(self, env):
    super().__init__(env)
    # Ensure the space is treated as Box by GymNE
    # but represents the binary nature
    self.action_space = gym.spaces.Box(0, 1, shape = (env.action_space.n, ), dtype = np.float32)

  def action(self, action):
    # Convert continuous [-1, 1] floats to binary [0, 1] integers
    # Any value > 0 becomes 1, else 0
    return (action > 0).astype(np.int8)

def init_env(render_mode = None):
  import gymnasium_env

  env = gym.make('gymnasium_env/MarbleGame-v0', render_mode = render_mode)
  env = FlattenObservation(env)
  env = BinaryActionWrapper(env)

  return env

@pass_info
class LinearNetwork(nn.Module):
  def __init__(self, obs_length, act_length, bias = True, **kwargs):
    super().__init__()
    
    self.linear_layers = nn.Sequential(
      nn.Linear(obs_length, act_length, bias = bias)
    )

  def forward(self, obs):
    return self.linear_layers(obs)

def train():
  load_saved_model = LOAD_MODEL

  print("start: {time}".format(time = datetime.datetime.now()))

  # Set the NeuroEvolution Problem
  problem = GymNE(
    env = init_env,
    network = LinearNetwork,
    num_actors = 4
  )

  searcher = Cosyne(
    problem,
    num_elites = 1,
    popsize = POP_SIZE,
    tournament_size = 4,
    mutation_stdev = 0.3,
    mutation_probability = 0.5,
    permute_all = True
  )

  # Log result
  logger = PandasLogger(searcher)

  # Run searcher
  searcher.run(RUN_AMOUNT)

  print("train end: {time}".format(time = datetime.datetime.now()))

  # Plot mean score
  plot_figure = logger.to_dataframe().mean_eval.plot().get_figure()

  result_file_name = "training_plot.png"
  result_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
  if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

  result_full_file_name = os.path.join(result_folder_path, result_file_name)
  plot_figure.savefig(result_full_file_name)

  print(logger.to_dataframe())

  # Best population solution
  center_solution = searcher.status["pop_best"]

  # Save best population policy
  policy_file_name = "policy.pickle"
  policy_folder_path = os.path.join(os.path.dirname(__file__), "policy")
  if not os.path.exists(policy_folder_path):
    os.makedirs(policy_folder_path)

  policy_full_file_name = os.path.join(policy_folder_path, policy_file_name)
  problem.save_solution(center_solution, policy_full_file_name)

  # Visualize best population
  policy_net = problem.to_policy(center_solution)
  for _ in range(10): # Visualize 10 episodes
    result = problem.visualize(policy_net)
    print('Visualised episode has cumulative reward:', result['cumulative_reward'])

  print("end: {time}".format(time = datetime.datetime.now()))

# Test saved model without the training
def test_model():
  import gymnasium_env
  print("start: {time}".format(time = datetime.datetime.now()))

  model = None

  policy_file_name = "policy.pickle"
  policy_folder_path = os.path.join(os.path.dirname(__file__), "policy")
  policy_full_file_name = os.path.join(policy_folder_path, policy_file_name)
  with open(policy_full_file_name, "rb") as input_file:
    model = pickle.load(input_file)

  policy = model["policy"]

  env = init_env(render_mode="human")
  obs, _ = env.reset()

  score = 0
  done = False

  while not done:
    # Get move
    prediction = policy(torch.tensor(obs, dtype = torch.float32))
    
    obs, reward, done, _, info = env.step(env.action(prediction.detach().numpy()))

    score += reward

  print("score: {score}".format(score = score))
  print("end: {time}".format(time = datetime.datetime.now()))

if __name__ == '__main__':
  # for _ in range(1):
  #   train()
  test_model()