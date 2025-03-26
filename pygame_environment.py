import pygame
import os
from pygame.sprite import Sprite
import math
import time
import random

class Scene:
  def on_draw(self, surface): pass
  def on_event(self, event): pass
  def on_update(self, delta): pass

class MarbleGameManager:
  def __init__(self, width = 512, height = 512, center = True):
    if center:
      os.environ["SDL_VIDEO_CENTERED"] = "1"

    # Basic setup
    pygame.display.set_caption('Marble')
    self.surface = pygame.display.set_mode((width, height))
    self.rect = self.surface.get_rect()
    self.clock = pygame.time.Clock()
    self.running = False
    self.delta = 0
    self.fps = 60

    # Scene interface
    self.scene = Scene

    self.running = False

  def game_step(self, ai_action = None):
    self.reward = 0

    for event in pygame.event.get():
      self.scene.on_event(event)
      
    self.scene.on_update(self.delta, ai_action)
    self.scene.on_draw(self.surface)
    pygame.display.flip()
    self.delta = self.clock.tick(self.fps)

    return self.reward, not self.running
  
  def get_state(self):
    subsurface = self.surface.subsurface(self.rect)
    pixel_array = pygame.surfarray.array3d(subsurface).astype("float32")
    return pixel_array

class BaseEntity(Sprite):
  def __init__(self, image, position, anchor = "topleft"):
    Sprite.__init__(self)
    self.original_image = image
    self.image = image
    self.rect = image.get_rect()
    setattr(self.rect, anchor, position)

    self.angle = 0
    self.speed = 0.05
    self.rotation_speed = 0.1
    self.cooldownStart = 0
    self.center = pygame.Vector2(self.rect.center)
    self.set_direction()

  def draw(self, surface):
    surface.blit(self.image, self.rect)

  def set_direction(self):
    rad = math.radians(self.angle)
    self.direction = pygame.Vector2(math.sin(rad), math.cos(rad))
      
  def do_rotate(self):
    self.image = pygame.transform.rotate(self.original_image, self.angle)
    self.rect = self.image.get_rect()
    self.rect.center = self.center
    self.set_direction()

  def on_keydown(self, keys_press, delta): pass
  def on_update(self, delta): pass

class PlayerEntity(BaseEntity):
  def __init__(self, image, position, anchor = "topleft"):
    BaseEntity.__init__(self, image, position, anchor)
    self.ray_cast(32)

  def on_keydown(self, keys_press, delta, game_size, spawn_func):
    if keys_press[pygame.K_UP] or keys_press[pygame.K_w]:
      self.center += pygame.Vector2(0, -1) * delta * self.speed
    if keys_press[pygame.K_DOWN] or keys_press[pygame.K_s]:
      self.center += pygame.Vector2(0, 1) * delta * self.speed
    if keys_press[pygame.K_LEFT] or keys_press[pygame.K_a]:
      self.center += pygame.Vector2(-1, 0) * delta * self.speed
    if keys_press[pygame.K_RIGHT] or keys_press[pygame.K_d]:
      self.center += pygame.Vector2(1, 0) * delta * self.speed

    if self.center.x < 0:
      self.center.x = 0
    elif self.center.x > game_size[0]:
      self.center.x = game_size[0]
    if self.center.y < 0:
      self.center.y = 0
    elif self.center.y > game_size[1]:
      self.center.y = game_size[1]

    self.rect.center = self.center
    self.ray_cast(32)

    if keys_press[pygame.K_q]:
      self.angle = (self.angle + self.rotation_speed * delta) % 360
    if keys_press[pygame.K_e]:
      self.angle = (self.angle - self.rotation_speed * delta) % 360
    self.do_rotate()

    if keys_press[pygame.K_SPACE]:
      if time.time() - self.cooldownStart > 0.1:
        self.cooldownStart = time.time()
        spawn_func()

  def ray_cast(self, ray_count):
    self.lines = []

    for n in range(ray_count):
      angle = ((2 * math.pi) / ray_count) * n

      self.lines.append((self.rect.center, (self.center.x - math.sin(angle) * 512, self.center.y + math.cos(angle) * 512)))

class EnemyEntity(BaseEntity):
  def __init__(self, image, position, anchor = "topleft", speed = 0.02):
    BaseEntity.__init__(self, image, position, anchor)
    self.speed = speed

  def on_update(self, delta, player_pos = None):
    if player_pos:
      current_pos = self.center
      self.angle = math.degrees(math.atan2(current_pos.y - player_pos.y, player_pos.x - current_pos.x)) - 90 # y is inverted with higher y meaning lower position
      self.do_rotate()
    self.center -= self.direction * delta * self.speed
    self.rect.center = self.center

class ArrowEntity(BaseEntity):
  def __init__(self, image, position, anchor = "topleft", lifespan = 1, angle = 0, speed = 0.2):
    BaseEntity.__init__(self, image, position, anchor)
    self.lifespan = lifespan
    self.start = time.time()
    self.angle = angle
    self.speed = speed
    self.do_rotate()

  def on_update(self, delta):
    self.center -= self.direction * delta * self.speed
    self.rect.center = self.center

class MarbleGame(Scene):
  def __init__(self, manager):
    self.manager = manager

    self.base_spawn_rate = 5
    self.base_spawn_amount = 1
    self.min_spawn_rate = 0.1

    self.reset()

  def reset(self):
    self.time_score = 0
    self.enemy_score = 0

    self.start_time = time.time()
    self.enemy_last_spawn = None

    self.create_player_image()
    self.player = PlayerEntity(self.player_image, self.manager.rect.center, "center")
    self.create_enemy_image()
    self.enemies = []
    self.create_arrow_image()
    self.arrows = []

  # Entity image
  def create_player_image(self):
    self.player_image = pygame.Surface((16, 16), pygame.SRCALPHA)
    pygame.draw.circle(self.player_image, (0, 0, 127), (8, 8), 8)
    pygame.draw.circle(self.player_image, (255, 255, 255), (8, 2), 2)

  def create_enemy_image(self):
    self.enemy_image = pygame.Surface((8, 8), pygame.SRCALPHA)
    pygame.draw.circle(self.enemy_image, (127, 0, 0), (4, 4), 4)
    pygame.draw.circle(self.enemy_image, (255, 255, 255), (4, 2), 2)

  def create_arrow_image(self):
    self.arrow_image = pygame.Surface((4, 8), pygame.SRCALPHA)
    self.arrow_image.fill(pygame.Color(255, 255, 0))

  # Spawn entity
  def spawn_enemy(self):
    game_size = self.manager.rect.size
    random_edge = random.randrange(4)
    random_pos = random.randrange(game_size[random_edge % 2])
    random_coord = tuple(random_pos if random_edge % 2 == i else game_size[i] * (random_edge // 2) for i in range(2))
    self.enemies.append(EnemyEntity(self.enemy_image, random_coord, "center"))
    
  def spawn_player_arrow(self):
    self.arrows.append(ArrowEntity(self.arrow_image, self.player.rect.center, "center", 5, self.player.angle))

  # Timer
  def enemy_spawn_timer(self):
    time_elapsed = time.time() - self.start_time
    self.time_score = int(time_elapsed)
    spawn_rate = max(self.min_spawn_rate, self.base_spawn_rate - (time_elapsed // 10) * 0.1)
    spawn_amount = self.base_spawn_amount + (time_elapsed // 60)

    if not self.enemy_last_spawn or time.time() - self.enemy_last_spawn > spawn_rate:
      for _ in range(int(spawn_amount)):
        self.spawn_enemy()
      self.enemy_last_spawn = time.time()

  # Event
  def on_draw(self, surface):
    surface.fill(pygame.Color(0, 127, 0))

    self.player.draw(surface)

    for enemy in self.enemies:
      enemy.draw(surface)

    for arrow in self.arrows:
      arrow.draw(surface)
      
  def on_event(self, event):
    if event.type == pygame.QUIT:
      self.manager.running = False

  def on_update(self, delta, ai_action = None):
    if ai_action:
      key_mapping = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_q, pygame.K_e, pygame.K_SPACE]
      keys = dict(zip(key_mapping, ai_action))
      other_key_dict = dict.fromkeys([pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d], False)
      keys.update(other_key_dict)
    else:
      keys = pygame.key.get_pressed()
    
    self.player.on_keydown(keys, delta, self.manager.rect.size, self.spawn_player_arrow)
    player_collison = self.player.rect.collidelist([enemy.rect for enemy in self.enemies])
    if player_collison >= 0:
      self.manager.reward += -100
      self.manager.running = False

    for enemy in self.enemies:
      enemy.on_update(delta, self.player.center)
      enemy_collison = enemy.rect.collidelist([arrow.rect for arrow in self.arrows])
      if enemy_collison >= 0:
        self.enemies.pop(self.enemies.index(enemy))
        self.arrows.pop(enemy_collison)
        self.manager.reward += 10
        self.enemy_score += 100

    for arrow in self.arrows:
      if time.time() - arrow.start <= arrow.lifespan:
        arrow.on_update(delta)
      else:
        self.arrows.pop(self.arrows.index(arrow))

    self.enemy_spawn_timer()

  def check_ray_cast(self):
    return [any(enemy.rect.clipline(line) for enemy in self.enemies) for line in self.player.lines]

def main():
  manager = MarbleGameManager()
  manager.scene = MarbleGame(manager)
  manager.running = True
  
  while manager.running:
    manager.game_step()

  print("Score: " + str(manager.scene.time_score + manager.scene.enemy_score))

pygame.init()
  
if __name__ == "__main__":
  main()