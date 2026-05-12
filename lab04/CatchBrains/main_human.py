import gymnasium as gym
import pygame

import CatchBrains  # noqa: F401


def main():
    env = gym.make("CatchBrains/CatchBrains", render_mode="human")
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = [-1.0]
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = [1.0]
        else:
            action = [0.0]

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            running = False

    env.close()


if __name__ == "__main__":
    main()
