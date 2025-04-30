import numpy as np
import time
import math
import copy
import random

PLAYER_SPEED = 0.05
PLAYER_ROT_SPEED = 0.005
PLAYER_SIZE = (16, 16)

ENEMY_SPEED = 0.02
ENEMY_ROT_SPEED = 0.005
ENEMY_SIZE = (8, 8)

ARROW_SPEED = 0.2
ARROW_SIZE = (4, 4)

SCREEN_SIZE = (512, 512)
STARTING_ANGLE = math.pi / 2
BASE_SPAWN_RATE = 5
BASE_SPAWN_AMOUNT = 1
MIN_SPAWN_RATE = 0.1

class BaseEntity:
  def __init__(self, size, pos, angle, speed, rot_speed):
    self.size = size
    self.pos = pos
    self.angle = angle
    self.speed = speed
    self.rot_speed = rot_speed

class PlayerEntity(BaseEntity):
  def __init__(self, size, pos, angle, speed, rot_speed):
    BaseEntity.__init__(self, size, pos, angle, speed, rot_speed)
    self.cooldown_start = 0
    self.cooldown = 0.1

class EnemyEntity(BaseEntity):
  def __init__(self, size, pos, angle, speed, rot_speed):
    BaseEntity.__init__(self, size, pos, angle, speed, rot_speed)

class ArrowEntity(BaseEntity):
  def __init__(self, size, pos, angle, speed, lifespan = 1):
    BaseEntity.__init__(self, pos, angle, speed, 0)
    self.lifespan = lifespan
    self.start = time.time()

def get_init_state():
  return np.array([PlayerEntity(PLAYER_SIZE, (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2), STARTING_ANGLE, PLAYER_SPEED, PLAYER_ROT_SPEED), # player
                  time.time(), # start_time
                  set([]), # enemy_list
                  set([]), # arrow_list
                  None, # enemy_last_spawn
                  np.array([0, 0])], # score (time_score, enemy_score)
                  False # game_over
  )

def get_current_state(state):
  player, _ = state

  cur_state = [
    # Player dist from edge
    player.pos[0] / SCREEN_SIZE[0], # Left
    (SCREEN_SIZE[0] - player.pos[0]) / SCREEN_SIZE[0], # Right
    player.pos[1] / SCREEN_SIZE[1], # Top
    (SCREEN_SIZE[1] - player.pos[1]) / SCREEN_SIZE[1], # Bottom

    # Shooting direction
    math.cos(player.angle),
    math.sin(player.angle)
  ]

  # List of ray cast from player and whether they have collided with an enemy or not
  # enemy_collision = check_ray_cast()

  # cur_state.extend(enemy_collision)

  return np.array(cur_state, dtype = float)

def update_state(action, state):
  player, start_time, enemy_list, arrow_list, enemy_last_spawn, score, game_over = state

  # Player action
  move_direction = np.array([0, 0])

  if action[0]:
    move_direction = np.add(move_direction, np.array([0, -1]))
  if action[1]:
    move_direction = np.add(move_direction, np.array([0, 1]))
  if action[2]:
    move_direction = np.add(move_direction, np.array([-1, 0]))
  if action[3]:
    move_direction = np.add(move_direction, np.array([1, 0]))

  norm = np.linalg.norm(move_direction)
  if norm != 0:
    move_direction /= norm

  player.pos = np.add(player.pos, move_direction * player.speed)

  rotation = 0
  
  if action[4]:
    rotation += player.rot_speed
  if action[5]:
    rotation -= player.rot_speed

  player.angle = (player.angle + rotation) % (2 * math.pi)

  # Enemy action
  enemy_list_update = copy.deepcopy(enemy_list)

  # Todo - enemy movement

  time_elapsed = time.time() - start_time
  time_score_update = int(time_elapsed)
  spawn_rate = max(MIN_SPAWN_RATE, BASE_SPAWN_RATE - (time_elapsed // 10) * 0.1)
  spawn_amount = BASE_SPAWN_AMOUNT + (time_elapsed // 60)

  if not enemy_last_spawn or time.time() - enemy_last_spawn > spawn_rate:
    for _ in range(int(spawn_amount)):
      random_coord = random_edge_pos()
      enemy_list_update.add(EnemyEntity(ENEMY_SIZE, random_coord, STARTING_ANGLE, ENEMY_SPEED, ENEMY_ROT_SPEED))
    enemy_last_spawn = time.time()

  # Arrow
  arrow_list_update = copy.deepcopy(arrow_list)

  if action[6]:
    if time.time() - player.cooldown_start > player.cooldown:
      player.cooldown_start = time.time()
      arrow_list_update.add(ArrowEntity(ARROW_SIZE, player.pos, player.angle, ARROW_SPEED))

  arrow_list_update = np.array([arrow for arrow in arrow_list_update if time.time() - arrow.start <= arrow.lifespan])

  # Collision
  game_over_update = False

  player_collision = get_entity_collision(player)

  enemy_list_to_remove = set([])
  arrow_list_to_remove = set([])

  for enemy in enemy_list_update:
    enemy_collision = get_entity_collision(enemy)

    if check_collision(player_collision, enemy_collision):
      game_over_update = True

    for arrow in arrow_list_update:
      arrow_collision = get_entity_collision(arrow)

      if check_collision(enemy_collision, arrow_collision):
        enemy_list_to_remove.add(enemy)
        arrow_list_to_remove.add(arrow)
        break

  enemy_list_update = enemy_list_update - enemy_list_to_remove
  arrow_list_update = arrow_list_update - arrow_list_to_remove

  # Score
  score_update = np.array([time_score_update])

  return np.array([player, start_time, enemy_list_update, arrow_list_update, enemy_last_spawn, score_update, game_over_update])

def random_edge_pos():
  random_edge = random.randrange(4)
  random_pos = random.randrange(SCREEN_SIZE[random_edge % 2])
  random_coord = tuple(random_pos if random_edge % 2 == i else SCREEN_SIZE[i] * (random_edge // 2) for i in range(2))

  return random_coord

def get_entity_collision(entity : BaseEntity):
  return (entity.pos[0] - (entity.size[0] / 2), entity.pos[0] + (entity.size[0] / 2), entity.pos[1] - (entity.size[1] / 2), entity.pos[1] + (entity.size[1] / 2))

def check_collision(colli_a, colli_b):
  if colli_a[0] > colli_b[1] or colli_a[1] < colli_b[0]:
    return False
  
  if colli_a[2] > colli_b[3] or colli_a[3] < colli_b[2]:
    return False
  
  return True

class MarbleGame:
  def __init__(self):
    self.state_shape = ((32 * 5) + 6,)
    self.act_shape = (7,)

    self.reset()

  def reset(self):
    self.time_score = 0
    self.enemy_score = 0

    self.state = get_init_state()

  def step(self, action):
    cur_state = update_state(action, self.state)
    # reward = get_reward()