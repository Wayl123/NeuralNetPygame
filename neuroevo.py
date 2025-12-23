import torch
import torch.nn as nn
from evotorch.neuroevolution import GymNE
from evotorch.algorithms import PGPE
from evotorch.logging import PandasLogger
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os
import datetime
import numpy as np

LOAD_MODEL = False
POP_SIZE = 64
RUN_AMOUNT = 64
RADIUS_INIT = 5.0
RAY_CAST_COUNT = 32

class BinaryActionWrapper(gym.ActionWrapper):
  def __init__(self, env):
    super().__init__(env)
    # Ensure the space is treated as Box by GymNE
    # but represents the binary nature
    self.action_space = gym.spaces.Box(0, 1, shape = (env.action_space.n, ), dtype = np.float32)

  def action(self, action):
    # Convert continuous [0, 1] floats to binary [0, 1] integers
    # Any value > 0.5 becomes 1, else 0
    return (action > 0.5).astype(np.int8)

def init_env():
  import gymnasium_env

  env = gym.make('gymnasium_env/MarbleGame-v0')
  env = FlattenObservation(env)
  env = BinaryActionWrapper(env)

  return env

class LinearNetwork(nn.Module):
  def __init__(self, obs_len, act_len, bias = True, **kwargs):
    super().__init__()
    
    self.linear_layers = nn.Sequential(
      nn.Linear(obs_len, 128, bias = bias),
      nn.ReLU(),
      nn.Linear(128, 256, bias = bias),
      nn.ReLU(),
      nn.Linear(256, 512, bias = bias),
      nn.ReLU(),
      nn.Linear(512, 256, bias = bias),
      nn.ReLU(),
      nn.Linear(256, 128, bias = bias),
      nn.ReLU(),
      nn.Linear(128, act_len, bias = bias),
      nn.Tanh()
    )

  def forward(self, obs):
    return self.linear_layers(obs)
  
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

def train():
  load_saved_model = LOAD_MODEL
  model = LinearNetwork(4 + RAY_CAST_COUNT, 7)

  if load_saved_model:
    model.load()

  print("start: {time}".format(time = datetime.datetime.now()))

  # Set the NeuroEvolution Problem
  problem = GymNE(
    env = init_env,
    network = model,
    num_actors = 4
  )

  max_speed = RADIUS_INIT / 15
  center_learning_rate = max_speed / 2

  # Apply Searcher to the Problem
  searcher = PGPE(
    problem,
    popsize = POP_SIZE,
    radius_init = max_speed,
    center_learning_rate = center_learning_rate,
    stdev_learning_rate = 0.1,
    optimizer="clipup",
    optimizer_config = {
        'max_speed': max_speed,
        'momentum': 0.9
    }
  )

  # Log result
  logger = PandasLogger(searcher)

  # Run searcher
  searcher.run(RUN_AMOUNT)

  print("train end: {time}".format(time = datetime.datetime.now()))

  # Plot mean score
  plot_figure = logger.to_dataframe().mean_eval.plot().get_figure()

  file_name = "training_plot.png"
  result_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
  if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

  png_file = os.path.join(result_folder_path, file_name)
  plot_figure.savefig(png_file)

  print(logger.to_dataframe())

  # Visualize best population
  center_solution = searcher.status["pop_best"]
  policy_net = problem.to_policy(center_solution)
  for _ in range(10): # Visualize 10 episodes
    result = problem.visualize(policy_net)
    print('Visualised episode has cumulative reward:', result['cumulative_reward'])

  print("end: {time}".format(time = datetime.datetime.now()))

# Test saved model without the training
def test_model():
  print("start: {time}".format(time = datetime.datetime.now()))

  model = LinearNetwork(4 + RAY_CAST_COUNT, 7)
  model.load()

  env = gym.make('gymnasium_env/MarbleGame-v0', render_mode="human")
  obs, _ = env.reset()

  score = 0
  done = False

  while not done:
    # Get move
    prediction = model(torch.tensor(obs, dtype = torch.float32))
    action = [move > 0 for move in prediction]

    obs, reward, done, _, info = env.step(action)

    score += reward

  print("score: {score}".format(score = score))
  print("end: {time}".format(time = datetime.datetime.now()))

if __name__ == '__main__':
  for _ in range(1):
    train()
  # test_model()