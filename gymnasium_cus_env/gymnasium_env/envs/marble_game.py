import gymnasium as gym
import pygame
import numpy as np
import math
import time
import copy
import random

PLAYER_SPEED = 1.0
PLAYER_HP = 10.0
PLAYER_SIZE = (16.0, 16.0)

MAX_FOOD = 10
FOOD_SPAWN_RATE = 0.1
FOOD_SIZE = (8.0, 8.0)

RAY_CAST_COUNT = 64

SCREEN_SIZE = (512, 512)

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

class MarbleGameEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 64}
  
  def __init__(self, render_mode=None, size=5):
    self.size = size  # The size of the square grid
    self.window_size = 512.0  # The size of the PyGame window

    # Observation space 
    self.observation_space = gym.spaces.Dict(
      {
        "position": gym.spaces.Box(0.0, self.window_size - 1, shape = (2, ), dtype = np.float64),
        "ray_cast": gym.spaces.Box(0.0, 512.0, shape = (RAY_CAST_COUNT, ), dtype = np.float64)
      }
    )

    # 4 directional action
    self.action_space = gym.spaces.MultiBinary(4)

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    self.window = None
    self.clock = None

  def _get_obs(self):
    return {"position": self._player.pos, "ray_cast": self._player_ray_cast}

  def _get_info(self):
    return {"player":self._player, "start_time": self._start_time, "food_list": self._food_list}

  def reset(self, seed = None, options = None):
    super().reset(seed = seed)

    # Observation
    self._player = PlayerEntity(PLAYER_SIZE, np.array([SCREEN_SIZE[0] / 2.0, SCREEN_SIZE[1] / 2.0], dtype = np.float64), PLAYER_SPEED, PLAYER_HP)
    self._player_ray_cast = np.asarray([512.0] * RAY_CAST_COUNT, dtype = np.float64)

    # Info
    self._start_time = time.time()
    self._food_list = set([])
    self._food_last_spawn = None

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, info
  
  def _random_pos(self):
    random_coord = np.array([random.randrange(SCREEN_SIZE[0]), random.randrange(SCREEN_SIZE[1])])

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
          return None  # Line is outside and parallel, so completely discarded
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
  
  def spawn_food(self, food_list):
    if len(food_list) < MAX_FOOD:
      food_list.add(FoodEntity(FOOD_SIZE, self._random_pos()))
  
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

    # Food spawn
    food_list_update = copy.deepcopy(self._food_list)

    spawn_rate = FOOD_SPAWN_RATE

    if not self._food_last_spawn:
      while len(food_list_update) < MAX_FOOD:
        self.spawn_food(food_list_update)

      self._food_last_spawn = time.time()
    elif time.time() - self._food_last_spawn > spawn_rate and len(food_list_update) < MAX_FOOD:
      self.spawn_food(food_list_update)

      self._food_last_spawn = time.time()

    # Collision and Ray-cast
    reward = 0
    terminated = False

    player_collision = self._get_entity_collision(self._player)

    food_list_to_remove = set([])

    ray_cast_points = []
    ray_cast_update = []

    for n in range(RAY_CAST_COUNT):
      angle = ((2 * math.pi) / RAY_CAST_COUNT) * n

      ray_cast_points.append((self._player.pos[0], self._player.pos[1], 
                              self._player.pos[0] + math.sin(angle) * 512, self._player.pos[1] + math.cos(angle) * 512))
      
      ray_cast_update.extend([512])

    for food in food_list_update:
      food_collision = self._get_entity_collision(food)

      food_remove_flag = False

      if self._check_collision(player_collision, food_collision):
        food_list_to_remove.add(food)
        if self._player.hp < 10:
          self._player.hp = min(self._player.hp + 1, PLAYER_HP)
        reward += 1
        food_remove_flag = True

      if food_remove_flag:
        continue

      for index, points in enumerate(ray_cast_points):
        contact_dist = self._check_line_collision(food_collision, points)
        if contact_dist:
          ray_cast_update[index] = min(ray_cast_update[index], contact_dist)

    # Time drain
    self._player.hp -= 0.001
    if self._player.hp <= 0:
      terminated = True
    reward -= 0.001

    # Time limit
    if time.time() - self._start_time > 60:
      terminated = True

    # Update saved variable
    self._food_list = food_list_update - food_list_to_remove
    self._player_ray_cast = np.asarray(ray_cast_update)

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
    player_image = pygame.Surface(PLAYER_SIZE, pygame.SRCALPHA)
    pygame.draw.circle(player_image, (0, 0, 127), (8, 8), 8)

    hp_ratio = self._player.hp / PLAYER_HP

    pygame.draw.rect(player_image, (127, 0, 0), (2, 2, 12 * hp_ratio, 2))

    player_rect = player_image.get_rect()

    player_rect.center = self._player.pos

    canvas.blit(player_image, player_rect)

    # Food image
    food_image = pygame.Surface(FOOD_SIZE, pygame.SRCALPHA)
    pygame.draw.circle(food_image, (0, 127, 0), (4, 4), 4)

    for food in self._food_list:
      food_rect = food_image.get_rect()

      food_rect.center = food.pos

      canvas.blit(food_image, food_rect)

    # Ray cast
    red_line_colour = (255, 0, 0)
    green_line_colour = (0, 127, 0)
      
    for n in range(RAY_CAST_COUNT):
      angle = ((2 * math.pi) / RAY_CAST_COUNT) * n

      ray_cast_points = [(self._player.pos[0], self._player.pos[1]),
                        (self._player.pos[0] + math.sin(angle) * self._player_ray_cast[n], self._player.pos[1] + math.cos(angle) * self._player_ray_cast[n]),
                        (self._player.pos[0] + math.sin(angle) * 512.0, self._player.pos[1] + math.cos(angle) * 512.0)]

      pygame.draw.line(canvas, red_line_colour, ray_cast_points[0], ray_cast_points[1])
      pygame.draw.line(canvas, green_line_colour, ray_cast_points[1], ray_cast_points[2])

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