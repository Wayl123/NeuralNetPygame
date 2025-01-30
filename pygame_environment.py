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
  def __init__(self, width = 640, height = 480, center = True):
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

  def mainloop(self):
    self.running = True

    while self.running:
      # 1. collect user input
      for event in pygame.event.get():
        self.scene.on_event(event)
        
      self.scene.on_update(self.delta)
      self.scene.on_draw(self.surface)
      pygame.display.flip()
      self.delta = self.clock.tick(self.fps)

    self.score = self.scene.time_score + self.scene.enemy_score

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

  def on_keydown(self, keys_press, delta, spawn_func):
    if keys_press[pygame.K_UP] or keys_press[pygame.K_w]:
      self.center += pygame.Vector2(0, -1) * delta * self.speed
    if keys_press[pygame.K_DOWN] or keys_press[pygame.K_s]:
      self.center += pygame.Vector2(0, 1) * delta * self.speed
    if keys_press[pygame.K_LEFT] or keys_press[pygame.K_a]:
      self.center += pygame.Vector2(-1, 0) * delta * self.speed
    if keys_press[pygame.K_RIGHT] or keys_press[pygame.K_d]:
      self.center += pygame.Vector2(1, 0) * delta * self.speed
    self.rect.center = self.center

    if keys_press[pygame.K_q]:
      self.angle = (self.angle + self.rotation_speed * delta) % 360
    if keys_press[pygame.K_e]:
      self.angle = (self.angle - self.rotation_speed * delta) % 360
    self.do_rotate()

    if keys_press[pygame.K_SPACE]:
      if time.time() - self.cooldownStart > 0.1:
        self.cooldownStart = time.time()
        spawn_func()

class EnemyEntity(BaseEntity):
  def __init__(self, image, position, anchor = "topleft", speed = 0.02):
    BaseEntity.__init__(self, image, position, anchor)
    self.speed = speed

  def on_update(self, delta, player_pos = None):
    if player_pos:
      current_pos = self.center
      self.angle = math.degrees(math.atan2(current_pos[1] - player_pos[1], player_pos[0] - current_pos[0])) - 90
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
    self.player_image = pygame.Surface((20, 20), pygame.SRCALPHA)
    pygame.draw.circle(self.player_image, (0, 0, 127), (10, 10), 10)
    pygame.draw.circle(self.player_image, (255, 255, 255), (10, 3), 3)

  def create_enemy_image(self):
    self.enemy_image = pygame.Surface((10, 10), pygame.SRCALPHA)
    pygame.draw.circle(self.enemy_image, (127, 0, 0), (5, 5), 5)
    pygame.draw.circle(self.enemy_image, (255, 255, 255), (5, 2), 2)

  def create_arrow_image(self):
    self.arrow_image = pygame.Surface((5, 10), pygame.SRCALPHA)
    self.arrow_image.fill(pygame.Color(255, 255, 0))

  # Spawn entity
  def spawn_enemy(self):
    gameSize = self.manager.rect.size
    randomEdge = random.randrange(4)
    randomPos = random.randrange(gameSize[randomEdge % 2])
    randomCoord = tuple(randomPos if randomEdge % 2 == i else gameSize[i] * (randomEdge // 2) for i in range(2))
    self.enemies.append(EnemyEntity(self.enemy_image, randomCoord, "center"))
    
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

  def on_update(self, delta):
    keys = pygame.key.get_pressed()

    self.player.on_keydown(keys, delta, self.spawn_player_arrow)
    player_collison = self.player.rect.collidelist([enemy.rect for enemy in self.enemies])
    if player_collison >= 0:
      self.manager.running = False

    for enemy in self.enemies:
      enemy.on_update(delta, self.player.center)
      enemy_collison = enemy.rect.collidelist([arrow.rect for arrow in self.arrows])
      if enemy_collison >= 0:
        self.enemies.pop(self.enemies.index(enemy))
        self.arrows.pop(enemy_collison)
        self.enemy_score += 100

    for arrow in self.arrows:
      if time.time() - arrow.start <= arrow.lifespan:
        arrow.on_update(delta)
      else:
        self.arrows.pop(self.arrows.index(arrow))

    self.enemy_spawn_timer()

def main():
  pygame.init()
  manager = MarbleGameManager(800, 600)
  manager.scene = MarbleGame(manager)
  manager.mainloop()
  print("Score: " + str(manager.score))
  
if __name__ == "__main__":
  main()