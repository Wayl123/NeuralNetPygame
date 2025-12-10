import gymnasium
import gymnasium_env
import numpy as np
import pygame
import math

env = gymnasium.make('gymnasium_env/MarbleGame-v0', render_mode="human")
env.reset()

done = False

while not done:
  action = np.random.randint(2, size = 7)
  obs, reward, done, _, info = env.step(action)
  # print(obs["ray_cast"])
  # print(info)

# pygame.init()
# pygame.display.init()
# window = pygame.display.set_mode((16, 16))

# canvas = pygame.Surface((16, 16))
# canvas.fill((255, 255, 255))

# enemy_image = pygame.Surface((8, 8), pygame.SRCALPHA)
# pygame.draw.circle(enemy_image, (127, 0, 0), (4, 4), 4)
# pygame.draw.circle(enemy_image, (255, 255, 255), (4, 2), 2)

# enemy_rect = enemy_image.get_rect()

# enemy_angle = math.atan2(16 - 8, 0 - 8) # x, y

# print(enemy_angle)
# print(math.sin(enemy_angle), math.cos(enemy_angle))

# test_angle = math.pi/2
# print(math.sin(test_angle), math.cos(test_angle))

# enemy_rotated = pygame.transform.rotate(enemy_image, math.degrees(enemy_angle + math.pi))
# enemy_rect.center = (8, 8)

# canvas.blit(enemy_rotated, enemy_rect)

# window.blit(canvas, canvas.get_rect())
# pygame.event.pump()
# pygame.display.update()

# while True:
#   pass