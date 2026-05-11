import numpy as np
import time
import math
import copy
import random
from PIL import Image
from PIL import ImageDraw

PLAYER_SPEED = 0.5
PLAYER_ROT_SPEED = 0.1
PLAYER_SIZE = (16, 16)
PLAYER_HEAD_SIZE = (4, 4)
PLAYER_SHOOTING_COOLDOWN = 0.5

ENEMY_SPEED = 0.2
ENEMY_ROT_SPEED = 0.1
ENEMY_SIZE = (8, 8)
ENEMY_HEAD_SIZE = (2, 2)

BULLET_SPEED = 2
BULLET_SIZE = (4, 4)
BULLET_LIFESPAN = 10.0

RAY_CAST_COUNT = 32

SCREEN_SIZE = 512
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
    self.cooldown = PLAYER_SHOOTING_COOLDOWN

class EnemyEntity(BaseEntity):
  def __init__(self, size, pos, angle, speed, rot_speed):
    BaseEntity.__init__(self, size, pos, angle, speed, rot_speed)

class BulletEntity(BaseEntity):
  def __init__(self, size, pos, angle, speed, lifespan = BULLET_LIFESPAN):
    BaseEntity.__init__(self, size, pos, angle, speed, 0)
    self.lifespan = lifespan
    self.start = time.time()

def get_init_state():
  return [PlayerEntity(PLAYER_SIZE, np.array([SCREEN_SIZE / 2.0, SCREEN_SIZE / 2.0], dtype = np.float64), STARTING_ANGLE, PLAYER_SPEED, PLAYER_ROT_SPEED), # player
          np.asarray([512.0] * RAY_CAST_COUNT, dtype = np.float64), # ray_cast
          time.time(), # start_time
          set([]), # enemy_list
          set([]), # bullet_list
          None, # enemy_last_spawn
          0] # score

def update_state(action, state):
  player, _, start_time, enemy_list, bullet_list, enemy_last_spawn, score = state

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

  rotation = 0
  
  if action[4]:
    rotation += player.rot_speed
  if action[5]:
    rotation -= player.rot_speed

  player.angle = (player.angle + rotation) % (2 * math.pi)

  # Enemy action
  enemy_list_update = copy.deepcopy(enemy_list)

  for enemy in enemy_list_update:
    enemy.angle = math.atan2(player.pos[1] - enemy.pos[1], player.pos[0] - enemy.pos[0])
    enemy_move_direction = np.array([math.cos(enemy.angle), math.sin(enemy.angle)])

    enemy.pos = np.add(enemy.pos, enemy_move_direction * enemy.speed)

  time_elapsed = time.time() - start_time
  spawn_rate = max(MIN_SPAWN_RATE, BASE_SPAWN_RATE - (time_elapsed // 10) * 0.1)
  spawn_amount = BASE_SPAWN_AMOUNT + (time_elapsed // 60)

  if not enemy_last_spawn or time.time() - enemy_last_spawn > spawn_rate:
    for _ in range(int(spawn_amount)):
      random_coord = random_edge_pos()
      enemy_list_update.add(EnemyEntity(ENEMY_SIZE, random_coord, STARTING_ANGLE, ENEMY_SPEED, ENEMY_ROT_SPEED))
    enemy_last_spawn = time.time()

  # Bullet
  bullet_list_update = copy.deepcopy(bullet_list)

  for bullet in bullet_list_update:
    bullet_move_direction = np.array([math.cos(bullet.angle), math.sin(bullet.angle)])

    bullet.pos = np.add(bullet.pos, bullet_move_direction * bullet.speed)

  if action[6]:
    if time.time() - player.cooldown_start > player.cooldown:
      player.cooldown_start = time.time()
      bullet_list_update.add(BulletEntity(BULLET_SIZE, player.pos, player.angle, BULLET_SPEED))

  bullet_list_update = {bullet for bullet in bullet_list_update if time.time() - bullet.start <= bullet.lifespan}

  # Collision and Ray-cast
  game_over = False

  player_collision = get_entity_collision(player)

  ray_cast_points = []
  ray_cast_update = []

  for n in range(RAY_CAST_COUNT):
    angle = ((2 * math.pi) / RAY_CAST_COUNT) * n

    ray_cast_points.append((player.pos[0], player.pos[1], 
                            player.pos[0] + math.sin(angle) * 512, player.pos[1] + math.cos(angle) * 512))
    
    ray_cast_update.extend([512])

  reward = 0

  enemy_list_to_remove = set([])
  bullet_list_to_remove = set([])

  for enemy in enemy_list_update:
    enemy_collision = get_entity_collision(enemy)

    if check_collision(player_collision, enemy_collision):
      game_over = True

    enemy_remove_flag = False

    for bullet in bullet_list_update:
      bullet_collision = get_entity_collision(bullet)

      if check_collision(enemy_collision, bullet_collision):
        enemy_list_to_remove.add(enemy)
        bullet_list_to_remove.add(bullet)
        reward += 1
        enemy_remove_flag = True
        break

    if enemy_remove_flag:
      continue

    for index, points in enumerate(ray_cast_points):
      contact_dist = check_line_collision(enemy_collision, points)
      if contact_dist:
        ray_cast_update[index] = min(ray_cast_update[index], contact_dist)

  reward += 0.0001

  enemy_list_update = enemy_list_update - enemy_list_to_remove
  bullet_list_update = bullet_list_update - bullet_list_to_remove

  # Scores
  score_update = score + reward

  return [player, ray_cast_update, start_time, enemy_list_update, bullet_list_update, enemy_last_spawn, score_update], reward, game_over

def random_edge_pos():
  random_edge = random.randrange(4)
  random_pos = random.randrange(SCREEN_SIZE)
  random_coord = np.array([random_pos if random_edge % 2 == i else SCREEN_SIZE * (random_edge // 2) for i in range(2)])

  return random_coord

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
    player, ray_cast, _, _, _, _, _ = state

    cur_state = [
      # Player dist from edge
      player.pos[0],
      player.pos[1],

      # Shooting direction
      math.cos(player.angle),
      math.sin(player.angle)
    ]

    # List of ray cast from player and whether they have collided with an enemy or not
    cur_state.extend(ray_cast)

    return np.array(cur_state, dtype = float)

  @staticmethod
  def render(state) -> Image:
    img = Image.new("RGB", (SCREEN_SIZE, SCREEN_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    player, ray_cast, _, enemy_list, bullet_list, _, _ = state

    # Player
    draw.circle(
      tuple(player.pos),
      PLAYER_SIZE[0] // 2,
      (0, 0, 127),
      (0, 0, 0)
    )
    draw.circle(
      np.add(player.pos, np.array([math.cos(player.angle), math.sin(player.angle)]) * ((PLAYER_SIZE[0] - PLAYER_HEAD_SIZE[0]) // 2)),
      PLAYER_HEAD_SIZE[0] // 2,
      (255, 255, 255)
    )

    # Enemy
    for enemy in enemy_list:
      draw.circle(
        tuple(enemy.pos),
        ENEMY_SIZE[0] // 2,
        (127, 0, 0),
        (0, 0, 0)
      )
      draw.circle(
        np.add(enemy.pos, np.array([math.cos(enemy.angle), math.sin(enemy.angle)]) * ((ENEMY_SIZE[0] - ENEMY_HEAD_SIZE[0]) // 2)),
        ENEMY_HEAD_SIZE[0] // 2,
        (255, 255, 255)
      )

    # Bullet
    for bullet in bullet_list:
      draw.circle(
        tuple(bullet.pos),
        BULLET_SIZE[0] // 2,
        (255, 255, 0),
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