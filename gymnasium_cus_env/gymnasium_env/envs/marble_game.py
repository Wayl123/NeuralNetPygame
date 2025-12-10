import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math
import time
import copy
import random

PLAYER_SPEED = 0.5
PLAYER_ROT_SPEED = 0.1
PLAYER_SIZE = (16, 16)
PLAYER_HEAD_SIZE = (4, 4)
PLAYER_SHOOTING_COOLDOWN = 0.5

ENEMY_SPEED = 0.2
ENEMY_ROT_SPEED = 0.1
ENEMY_SIZE = (8, 8)
ENEMY_HEAD_SIZE = (2, 2)

BULLET_SPEED = 2.0
BULLET_SIZE = (4, 4)
BULLET_LIFESPAN = 10.0

RAY_CAST_COUNT = 32
RAY_CAST_SECTION = 5

SCREEN_SIZE = (512, 512)
STARTING_ANGLE = math.pi
BASE_SPAWN_RATE = 20.0
BASE_SPAWN_AMOUNT = 1
MIN_SPAWN_RATE = 0.5

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
    self.cooldown_start = 0.0
    self.cooldown = PLAYER_SHOOTING_COOLDOWN

class EnemyEntity(BaseEntity):
  def __init__(self, size, pos, angle, speed, rot_speed):
    BaseEntity.__init__(self, size, pos, angle, speed, rot_speed)

class BulletEntity(BaseEntity):
  def __init__(self, size, pos, angle, speed, lifespan = BULLET_LIFESPAN):
    BaseEntity.__init__(self, size, pos, angle, speed, 0.0)
    self.lifespan = lifespan
    self.start = time.time()

class MarbleGameEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 64}
  
  def __init__(self, render_mode=None, size=5):
    self.size = size  # The size of the square grid
    self.window_size = 512  # The size of the PyGame window

    # Observation space 
    self.observation_space = spaces.Dict(
      {
        "position": spaces.Box(0, self.window_size - 1, shape = (2, ), dtype = np.float64),
        "angle": spaces.Box(-1, 1, shape = (2, ), dtype = np.float64),
        "ray_cast": spaces.Box(0, 1, shape = (RAY_CAST_COUNT, 5), dtype = np.bool)
      }
    )

    # 4 directional action
    self.action_space = spaces.MultiBinary(4)

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    self.window = None
    self.clock = None

  def _get_obs(self):
    return {"position": self._player.pos, "angle": np.array([math.sin(self._player.angle), math.cos(self._player.angle)], dtype = np.float32), "ray_cast": self._player_ray_cast}

  def _get_info(self):
    return {"player":self._player, "start_time": self._start_time, "enemy_list": self._enemy_list, "bullet_list": self._bullet_list, "enemy_last_spawn": self._enemy_last_spawn, "ray_cast_points": self._ray_cast_points}

  def reset(self, seed = None, options = None):
    super().reset(seed = seed)

    # Observation
    self._player = PlayerEntity(PLAYER_SIZE, np.array([SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2], dtype = np.int32), STARTING_ANGLE, PLAYER_SPEED, PLAYER_ROT_SPEED)
    self._player_ray_cast = np.asarray([[False, False, False, False, False]] * RAY_CAST_COUNT, dtype = np.bool)

    # Info
    self._start_time = time.time()
    self._enemy_list = set([])
    self._bullet_list = set([])
    self._enemy_last_spawn = None
    self._ray_cast_points = np.asarray([[(0,0)] * 6] * 32)

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, info
  
  def _random_edge_pos(self):
    random_edge = random.randrange(4)
    random_pos = random.randrange(SCREEN_SIZE[random_edge % 2])
    random_coord = np.array([random_pos if random_edge % 2 == i else SCREEN_SIZE[i] * (random_edge // 2) for i in range(2)])

    return random_coord
  
  def _get_entity_collision(self, entity : BaseEntity):
    return (entity.pos[0] - (entity.size[0] / 2), entity.pos[0] + (entity.size[0] / 2), entity.pos[1] - (entity.size[1] / 2), entity.pos[1] + (entity.size[1] / 2))

  def _check_collision(self, colli_1, colli_2):
    if colli_1[0] > colli_2[1] or colli_1[1] < colli_2[0]:
      return False
    
    if colli_1[2] > colli_2[3] or colli_1[3] < colli_2[2]:
      return False
    
    return True
  
  # Liang_barsky algorithm, but just return boolean of whether it intersect or not
  def _check_line_collision(self, colli, line):
    x_min, x_max, y_min, y_max = colli
    x1, y1, x2, y2 = line

    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]
    t_enter = 0.0
    t_exit = 1.0

    for i in range(4):
      if p[i] == 0:  # Check if line is parallel to the clipping boundary
        if q[i] < 0:
          return False  # Line is outside and parallel, so completely discarded
      else:
        t = q[i] / p[i]
        if p[i] < 0:
          if t > t_enter:
            t_enter = t
        else:
          if t < t_exit:
            t_exit = t

    if t_enter > t_exit:
      return False  # Line is completely outside

    return True
  
  def step(self, action):
    # Player action
    # Position
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

    self._player.pos = np.clip(np.add(self._player.pos, player_move_direction * self._player.speed), 0, self.window_size - 1)

    # Angle
    rotation = 0
  
    if action[4]:
      rotation -= PLAYER_ROT_SPEED
    if action[5]:
      rotation += PLAYER_ROT_SPEED

    self._player.angle = (self._player.angle + rotation) % (2 * math.pi)

    # Enemy action
    enemy_list_update = copy.deepcopy(self._enemy_list)

    for enemy in enemy_list_update:
      enemy.angle = math.atan2(self._player.pos[0] - enemy.pos[0], self._player.pos[1] - enemy.pos[1])
      enemy_move_direction = np.array([math.sin(enemy.angle), math.cos(enemy.angle)])

      enemy.pos = np.add(enemy.pos, enemy_move_direction * enemy.speed)

    time_elapsed = time.time() - self._start_time
    spawn_rate = max(MIN_SPAWN_RATE, BASE_SPAWN_RATE - (time_elapsed // 10) * 0.1)
    spawn_amount = BASE_SPAWN_AMOUNT + (time_elapsed // 60)

    if not self._enemy_last_spawn or time.time() - self._enemy_last_spawn > spawn_rate:
      for _ in range(int(spawn_amount)):
        random_coord = self._random_edge_pos()
        enemy_list_update.add(EnemyEntity(ENEMY_SIZE, random_coord, STARTING_ANGLE, ENEMY_SPEED, ENEMY_ROT_SPEED))
      self._enemy_last_spawn = time.time()

    # Bullet
    bullet_list_update = copy.deepcopy(self._bullet_list)

    for bullet in bullet_list_update:
      bullet_move_direction = np.array([math.sin(bullet.angle), math.cos(bullet.angle)])

      bullet.pos = np.add(bullet.pos, bullet_move_direction * bullet.speed)

    if action[6]:
      if time.time() - self._player.cooldown_start > self._player.cooldown:
        self._player.cooldown_start = time.time()
        bullet_list_update.add(BulletEntity(BULLET_SIZE, self._player.pos, self._player.angle, BULLET_SPEED))

    bullet_list_update = {bullet for bullet in bullet_list_update if time.time() - bullet.start <= bullet.lifespan}

    # Collision and Ray-cast
    reward = 0
    terminated = False

    player_collision = self._get_entity_collision(self._player)

    enemy_list_to_remove = set([])
    bullet_list_to_remove = set([])

    ray_cast_points = []
    ray_cast_update = []

    for n in range(RAY_CAST_COUNT):
      angle = ((2 * math.pi) / RAY_CAST_COUNT) * n

      ray_cast_points.append([(self._player.pos[0], self._player.pos[1]), 
                              (self._player.pos[0] + math.sin(angle) * 32, self._player.pos[1] + math.cos(angle) * 32),
                              (self._player.pos[0] + math.sin(angle) * 64, self._player.pos[1] + math.cos(angle) * 64),
                              (self._player.pos[0] + math.sin(angle) * 128, self._player.pos[1] + math.cos(angle) * 128),
                              (self._player.pos[0] + math.sin(angle) * 256, self._player.pos[1] + math.cos(angle) * 256),
                              (self._player.pos[0] + math.sin(angle) * 512, self._player.pos[1] + math.cos(angle) * 512)])
      
      ray_cast_update.append([False, False, False, False, False])

    for enemy in enemy_list_update:
      enemy_collision = self._get_entity_collision(enemy)

      if self._check_collision(player_collision, enemy_collision):
        terminated = True
        reward -= 10

      enemy_remove_flag = False

      for bullet in bullet_list_update:
        bullet_collision = self._get_entity_collision(bullet)

        if self._check_collision(enemy_collision, bullet_collision):
          enemy_list_to_remove.add(enemy)
          bullet_list_to_remove.add(bullet)
          reward += 1
          enemy_remove_flag = True
          break

      if enemy_remove_flag:
        continue

      for index, points in enumerate(ray_cast_points):
        for section in range(RAY_CAST_SECTION):
          ray_cast_update[index][section] = ray_cast_update[index][section] or self._check_line_collision(enemy_collision, points[section] + points[section + 1])

    # Update saved variable
    self._enemy_list = enemy_list_update - enemy_list_to_remove
    self._bullet_list = bullet_list_update - bullet_list_to_remove
    self._player_ray_cast = np.asarray(ray_cast_update)
    self._ray_cast_points = ray_cast_points

    # Observation and Info
    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, reward, terminated, False, info
  
  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode((self.window_size, self.window_size))
    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))

    # Player image
    player_image = pygame.Surface((16, 16), pygame.SRCALPHA)
    pygame.draw.circle(player_image, (0, 0, 127), (8, 8), 8)
    pygame.draw.circle(player_image, (255, 255, 255), (8, 2), 2)

    player_rect = player_image.get_rect()

    player_rotated = pygame.transform.rotate(player_image, math.degrees(self._player.angle + math.pi))
    player_rect.center = self._player.pos

    canvas.blit(player_rotated, player_rect)

    # Enemy image
    enemy_image = pygame.Surface((8, 8), pygame.SRCALPHA)
    pygame.draw.circle(enemy_image, (127, 0, 0), (4, 4), 4)
    pygame.draw.circle(enemy_image, (255, 255, 255), (4, 2), 2)
    
    for enemy in self._enemy_list:
      enemy_rect = enemy_image.get_rect()

      enemy_rotated = pygame.transform.rotate(enemy_image, math.degrees(enemy.angle + math.pi))
      enemy_rect.center = enemy.pos

      canvas.blit(enemy_rotated, enemy_rect)

    # Bullet image
    bullet_image = pygame.Surface((4, 4), pygame.SRCALPHA)
    bullet_image.fill(pygame.Color(255, 255, 0))

    for bullet in self._bullet_list:
      bullet_rect = bullet_image.get_rect()

      bullet_rotated = pygame.transform.rotate(bullet_image, math.degrees(bullet.angle + math.pi))
      bullet_rect.center = bullet.pos

      canvas.blit(bullet_rotated, bullet_rect)

    red_line_colour = (255, 0, 0)
    green_line_colour = (0, 127, 0)
      
    for index, points in enumerate(self._ray_cast_points):
      for section in range(RAY_CAST_SECTION):
        line_colour = green_line_colour if self._player_ray_cast[index, section] else red_line_colour
        pygame.draw.line(canvas, line_colour, points[section], points[section + 1])

    if self.render_mode == "human":
      # The following line copies our drawings from `canvas` to the visible window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      # We need to ensure that human-rendering occurs at the predefined framerate.
      # The following line will automatically add a delay to keep the framerate stable.
      self.clock.tick(self.metadata["render_fps"])
    else:  # rgb_array
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
      )
    
  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()