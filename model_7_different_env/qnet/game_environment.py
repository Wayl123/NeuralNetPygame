import numpy as np
import time
import math
import copy
import random
from PIL import Image
from PIL import ImageDraw

PLAYER_SPEED = 1.0
PLAYER_HP = 10.0
PLAYER_SIZE = (16.0, 16.0)

MAX_FOOD = 10
FOOD_SPAWN_RATE = 0.1
FOOD_SIZE = (8.0, 8.0)

RAY_CAST_COUNT = 64

SCREEN_SIZE = 512

class BaseEntity:
  def __init__(self, size, pos, speed, hp):
    self.size = size
    self.pos = pos
    self.speed = speed
    self.hp = hp

class PlayerEntity(BaseEntity):
  def __init__(self, size, pos, speed, hp):
    BaseEntity.__init__(self, size, pos, speed, hp)

class FoodEntity(BaseEntity):
  def __init__(self, size, pos):
    BaseEntity.__init__(self, size, pos, 0.0, 1.0)

def get_init_state():
  return [PlayerEntity(PLAYER_SIZE, np.array([SCREEN_SIZE / 2.0, SCREEN_SIZE / 2.0], dtype = np.float64), PLAYER_SPEED, PLAYER_HP), # player
          np.asarray([512.0] * RAY_CAST_COUNT, dtype = np.float64), # ray_cast
          time.time(), # start_time
          set([]), # food_list
          None, # food_last_spawn
          0] # score

def update_state(action, state):
  player, _, start_time, food_list, food_last_spawn, score = state

  # Player action
  player_move_direction = np.array([0, 0], dtype = float)

  if action[0]:
    player_move_direction = np.add(player_move_direction, np.array([0, -1]))
  if action[1]:
    player_move_direction = np.add(player_move_direction, np.array([0, 1]))
  if action[2]:
    player_move_direction = np.add(player_move_direction, np.array([-1, 0]))
  if action[3]:
    player_move_direction = np.add(player_move_direction, np.array([1, 0]))

  norm = np.linalg.norm(player_move_direction)
  if norm != 0:
    player_move_direction /= norm

  player.pos = np.clip(np.add(player.pos, player_move_direction * player.speed), 0, SCREEN_SIZE - 1)

  # Food spawn
  food_list_update = copy.deepcopy(food_list)

  spawn_rate = FOOD_SPAWN_RATE

  if not food_last_spawn:
    while len(food_list_update) < MAX_FOOD:
      spawn_food(food_list_update)

    food_last_spawn = time.time()
  elif time.time() - food_last_spawn > spawn_rate and len(food_list_update) < MAX_FOOD:
    spawn_food(food_list_update)

    food_last_spawn = time.time()

  # Collision and Ray-cast
  reward = 0
  terminated = False

  player_collision = get_entity_collision(player)

  food_list_to_remove = set([])

  ray_cast_points = []
  ray_cast_update = []

  for n in range(RAY_CAST_COUNT):
    angle = ((2 * math.pi) / RAY_CAST_COUNT) * n

    ray_cast_points.append((player.pos[0], player.pos[1], 
                            player.pos[0] + math.sin(angle) * 512, player.pos[1] + math.cos(angle) * 512))
    
    ray_cast_update.extend([512])

  for food in food_list_update:
    food_collision = get_entity_collision(food)

    food_remove_flag = False

    if check_collision(player_collision, food_collision):
      food_list_to_remove.add(food)
      if player.hp < 10:
        player.hp = min(player.hp + 1, PLAYER_HP)
      reward += 1
      food_remove_flag = True

    if food_remove_flag:
      continue

    for index, points in enumerate(ray_cast_points):
      contact_dist = check_line_collision(food_collision, points)
      if contact_dist:
        ray_cast_update[index] = min(ray_cast_update[index], contact_dist)

  # Time drain
  player.hp -= 0.001
  if player.hp <= 0:
    terminated = True
  reward -= 0.001

  # Time limit
  if time.time() - start_time > 60:
    terminated = True

  food_list_update = food_list_update - food_list_to_remove

  # Scores
  score_update = score + reward

  return [player, np.asarray(ray_cast_update), start_time, food_list_update, food_last_spawn, score_update], reward, terminated

def random_pos():
  random_coord = np.array([random.randrange(SCREEN_SIZE), random.randrange(SCREEN_SIZE)])

  return random_coord

def spawn_food(food_list):
  if len(food_list) < MAX_FOOD:
    food_list.add(FoodEntity(FOOD_SIZE, random_pos()))

def get_entity_collision(entity : BaseEntity):
  return (entity.pos[0] - (entity.size[0] / 2), entity.pos[0] + (entity.size[0] / 2), entity.pos[1] - (entity.size[1] / 2), entity.pos[1] + (entity.size[1] / 2))

def check_collision(colli_1, colli_2):
  if colli_1[0] > colli_2[1] or colli_1[1] < colli_2[0]:
    return False
  
  if colli_1[2] > colli_2[3] or colli_1[3] < colli_2[2]:
    return False
  
  return True

# Liang_barsky algorithm, but just return boolean of whether it intersect or not
def check_line_collision(colli, line):
  x_min, x_max, y_min, y_max = colli
  x1, y1, x2, y2 = line

  dx = x2 - x1
  dy = y2 - y1
  p = [-dx, dx, -dy, dy]
  q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]
  t_enter = 0.0
  t_exit = 1.0

  for i in range(4):
    if p[i] == 0:  # Check if line is parallel to the clipping boundary -1/-2 3/2 2/1 1/-1
      if q[i] < 0:
        return False  # Line is outside and parallel, so completely discarded 0.5 1.0 0.0 1.0 
    else:
      t = q[i] / p[i]
      if p[i] < 0:
        if t > t_enter:
          t_enter = t
      else:
        if t < t_exit:
          t_exit = t

  if t_enter > t_exit:
    return None  # Line is completely outside

  x1_clip = x1 + t_enter * dx
  y1_clip = y1 + t_enter * dy
  
  # Return distance from clip object
  return math.dist([x1, y1], [x1_clip, y1_clip])

class MarbleGame:
  def __init__(self):
    self.reset()

  def reset(self):
    self.time_score = 0
    self.enemy_score = 0

    self.state = get_init_state()

    return self.state

  def step(self, action):
    cur_state, reward, done = update_state(action, self.state)
    self.state = cur_state
    
    return self.state, reward, done
  
  def get_state_observation(self, state):
    player, ray_cast, _, _, _, _ = state

    cur_state = [
      # Player dist from edge
      player.pos[0],
      player.pos[1]
    ]

    # List of ray cast from player and whether they have collided with an enemy or not
    cur_state.extend(ray_cast)

    return np.array(cur_state, dtype = float)

  @staticmethod
  def render(state) -> Image:
    img = Image.new("RGB", (SCREEN_SIZE, SCREEN_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    player, ray_cast, _, food_list, _, _ = state

    # Player
    draw.circle(
      tuple(player.pos),
      PLAYER_SIZE[0] // 2,
      (0, 0, 127),
      (0, 0, 0)
    )

    hp_ratio = player.hp / PLAYER_HP

    draw.rectangle(
      [tuple(np.add(player.pos, (2 - (PLAYER_SIZE[0] // 2), 2 - (PLAYER_SIZE[1] // 2)))), tuple(np.add(player.pos, ((12 * hp_ratio) + 2.1 - (PLAYER_SIZE[0] // 2), 4 - (PLAYER_SIZE[1] // 2))))], 
      (127, 0, 0)
    )

    # Food
    for food in food_list:
      draw.circle(
        tuple(food.pos),
        FOOD_SIZE[0] // 2,
        (0, 127, 0),
        (0, 0, 0)
      )

    # Ray cast
    red_line_colour = (255, 0, 0)
    green_line_colour = (0, 127, 0)

    for n in range(RAY_CAST_COUNT):
      angle = ((2 * math.pi) / RAY_CAST_COUNT) * n

      ray_cast_points = [(player.pos[0], player.pos[1]),
                        (player.pos[0] + math.sin(angle) * ray_cast[n], player.pos[1] + math.cos(angle) * ray_cast[n]),
                        (player.pos[0] + math.sin(angle) * 512.0, player.pos[1] + math.cos(angle) * 512.0)]

      draw.line([ray_cast_points[0], ray_cast_points[1]], red_line_colour)
      draw.line([ray_cast_points[1], ray_cast_points[2]], green_line_colour)
    
    return img