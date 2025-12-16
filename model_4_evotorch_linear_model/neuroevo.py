import torch
import torch.nn as nn
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import PGPE
from evotorch.logging import PandasLogger
from game_environment import MarbleGame
import os
import datetime

LOAD_MODEL = False
POP_SIZE = 64
RUN_AMOUNT = 64
RADIUS_INIT = 5.0

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

def train_game(network : torch.nn.Module, save_image = False):
  game = MarbleGame()
  state = game.reset()
  obs = game.get_state_observation(state)
  images = None
  if save_image: 
    images = [game.render(state)]
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

    # Save image every 4 step
    if save_image and step % 4 == 0:
      images.append(game.render(state))

    if done:
      scores = state[5]
      score = sum(scores)

  if save_image:
    print("image: {time}".format(time = datetime.datetime.now()))
    file_name = "result_{score}.gif".format(score = score)
    result_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
    if not os.path.exists(result_folder_path):
      os.makedirs(result_folder_path)

    gif_file = os.path.join(result_folder_path, file_name)
    images[0].save(gif_file, save_all = True, append_images = images[1:], duration = 60, loop = 0)

  return score

def train():
  load_saved_model = LOAD_MODEL
  model = LinearNetwork((32 * 5) + 6, 7)

  if load_saved_model:
    model.load()

  print("start: {time}".format(time = datetime.datetime.now()))

  # Set the NeuroEvolution Problem
  problem = NEProblem(
    objective_sense = "max",
    network = model,
    network_eval_func = train_game,
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

  # Run model one more time to generate image
  train_game(model, True)

  # Save model
  model.save()

  # Plot mean score
  plot_figure = logger.to_dataframe().mean_eval.plot().get_figure()

  file_name = "training_plot.png"
  result_folder_path = os.path.join(os.path.dirname(__file__), "result_display")
  if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

  png_file = os.path.join(result_folder_path, file_name)
  plot_figure.savefig(png_file)

  print(logger.to_dataframe())
  print("end: {time}".format(time = datetime.datetime.now()))

# Test saved model without the training
def test_model():
  print("start: {time}".format(time = datetime.datetime.now()))

  model = LinearNetwork((32 * 5) + 6, 7)
  model.load()
  train_game(model, True)

  print("end: {time}".format(time = datetime.datetime.now()))

if __name__ == '__main__':
  for _ in range(1):
    train()
  # test_model()