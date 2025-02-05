import torch
import random
import numpy as np
from collections import deque
import time
import math
from operator import itemgetter
from pygame_environment import MarbleGameManager, MarbleGame
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
MAX_GAME = 1000

class Agent:
  def __init__(self):
    self.n_games = 1
    self.epsilon = 0 # Randomness
    self.gamma = 0.9 # Discount rate
    self.memory = deque(maxlen = MAX_MEMORY)
    self.model = Linear_QNet(16, 256, 7)
    self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)

  def get_state(self, game):
    player_center = game.player.center
    game_size = game.manager.rect.size
    player_dir = game.player.direction
    bullet_cooldown = time.time() - game.player.cooldownStart

    enemies = game.enemies
    enemies_dist = []

    for enemy in enemies:
      enemy_center = enemy.center
      dist_from_player = enemy_center.distance_to(player_center)
      rad = math.atan2(player_center.y - enemy_center.y, enemy_center.x - player_center.x) - (math.pi / 2)
      enemies_dist.append((dist_from_player, math.sin(rad), math.cos(rad)))

    enemies_dist.sort()
    enemy_count = len(enemies_dist)
    
    # Input to the nn model
    state = [
      # Player dist from edge
      player_center.x, # Left
      game_size[0] - player_center.x, # Right
      player_center.y, # Top
      game_size[1] - player_center.y, # Bottom

      # Shooting direction
      player_dir.x,
      player_dir.y,

      # Bullet cooldown
      bullet_cooldown,

      # Dist and direction of closest 3 enemies (need some testing to see how to do this)
      -1 if enemy_count < 1 else enemies_dist[0][0],
      0 if enemy_count < 1 else enemies_dist[0][1],
      0 if enemy_count < 1 else enemies_dist[0][2],

      -1 if enemy_count < 2 else enemies_dist[1][0],
      0 if enemy_count < 2 else enemies_dist[1][1],
      0 if enemy_count < 2 else enemies_dist[1][2],

      -1 if enemy_count < 3 else enemies_dist[2][0],
      0 if enemy_count < 3 else enemies_dist[2][1],
      0 if enemy_count < 3 else enemies_dist[2][2],
    ]

    return np.array(state, dtype = int)

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
    # Reduce randomness overtime to help in gradient descent converge
    self.epsilon = 80 - self.n_games

    if random.randint(0, 200) < self.epsilon:
      final_move = [bool(random.getrandbits(1)) for _ in range(7)]
    else:
      prediction = self.model(torch.tensor(state, dtype = torch.float))
      final_move = [move > 0 for move in prediction]

    return final_move

def train():
  plot_scores = []
  plot_mean_scores = []
  total_score = 0
  record = 0
  load_saved_model = True
  agent = Agent()
  game_manager = MarbleGameManager(800, 600)
  game_manager.scene = MarbleGame(game_manager)
  game_manager.running = True

  if load_saved_model:
    agent.model.load()

  while agent.n_games < MAX_GAME:
    # Get old state
    state_old = agent.get_state(game_manager.scene)

    # Get move
    final_move = agent.get_action(state_old)

    # Perform move and get new state
    reward, done = game_manager.game_step(final_move)
    state_new = agent.get_state(game_manager.scene)

    # Train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done)

    # Remember
    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
      score = game_manager.scene.time_score + game_manager.scene.enemy_score

      if score > record:
        record = score
        agent.model.save()

      print('Game', agent.n_games, 'Score', score, 'Record:', record)

      plot_scores.append(score)
      total_score += score
      mean_score = total_score / agent.n_games
      plot_mean_scores.append(mean_score)

      # Train long memory
      game_manager.running = True
      game_manager.scene.reset()
      agent.n_games += 1
      agent.train_long_memory()

if __name__ == '__main__':
  train()