from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class CatchBrainsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 600
        low = np.array([0, 0, 0], dtype=np.float32)
        high = np.array([100, 100, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = 50.0
        self.food_pos = np.array([np.random.uniform(0, 100), 100.0])
        
        observation = np.array([self.agent_pos, self.food_pos[0], self.food_pos[1]], dtype=np.float32)
        return observation, {}

    def step(self, action):
        self.agent_pos = np.clip(self.agent_pos + action[0] * 5, 0, 100)
        self.food_pos[1] -= 2.0 # jak szybko jedzenie spada
        
        reward = 0
        terminated = False
        
        # Check if caught
        if self.food_pos[1] <= 0:
            terminated = True
            distance = abs(self.agent_pos - self.food_pos[0])
            if distance < 5:
                reward = 10  # Success!
            else:
                reward = -10 # Missed
        
        observation = np.array([self.agent_pos, self.food_pos[0], self.food_pos[1]], dtype=np.float32)
        return observation, reward, terminated, False, {}

    import pygame

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            else:
                self.screen = pygame.Surface((self.window_size, self.window_size))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        scale = self.window_size / 100
        
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(self.agent_pos * scale - 10, self.window_size - 30, 20, 20)
        )

        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (int(self.food_pos[0] * scale), int((100 - self.food_pos[1]) * scale)),
            10
        )

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(60)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
