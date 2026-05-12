import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

IMGS_DIR = Path(__file__).parent.parent / "imgs"
BODYPARTS_DIR = IMGS_DIR / "bodyparts&brains"

BASE_SPEED = 6.0
SLOWDOWN = 0.02
ITEM_SIZE = 60          # base size; actual items are randomly scaled
ITEM_SIZE_MIN = 0.7
ITEM_SIZE_MAX = 1.4
ITEM_SPEED_MIN = 1.5
ITEM_SPEED_MAX = 5.0
ZOMBIE_W, ZOMBIE_H = 120, 180
HUD_HEIGHT = 60
SPAWN_INTERVAL = 40
MAX_ITEMS_ON_SCREEN = 6
BRAIN_CHANCE = 0.65

RISING_FRAMES = 10
WALKING_FRAMES = 10
DYING_FRAMES = 9
WALK_IDLE_FRAME = WALKING_FRAMES - 2   # klatka 8 — zombie stoi


class SpriteSheet:
    def __init__(self, path: Path, num_frames: int):
        sheet = pygame.image.load(str(path)).convert_alpha()
        w = sheet.get_width() // num_frames
        h = sheet.get_height()
        self.frames = [
            pygame.transform.scale(
                sheet.subsurface(pygame.Rect(i * w, 0, w, h)),
                (ZOMBIE_W, ZOMBIE_H),
            )
            for i in range(num_frames)
        ]
        self._flipped = [pygame.transform.flip(f, True, False) for f in self.frames]

    def get_frame(self, i: int, flipped: bool = False) -> pygame.Surface:
        pool = self._flipped if flipped else self.frames
        return pool[i % len(pool)]

    def __len__(self):
        return len(self.frames)


@dataclass
class FallingItem:
    x: float
    y: float
    item_type: str      # "brain" | "bodypart"
    images: list        # pygame.Surface frames already scaled to `size`
    speed: float        # pixels per step
    size: int           # display/collision size in pixels
    anim_frame: int = 0
    anim_timer: int = 0


def _load_base_item_images() -> tuple[list, list]:
    """Load all item images at base ITEM_SIZE. Scaled per-item at spawn time."""
    bp_imgs = [
        pygame.transform.scale(
            pygame.image.load(str(BODYPARTS_DIR / f"bodyparts_{i:02d}.png")).convert_alpha(),
            (ITEM_SIZE, ITEM_SIZE),
        )
        for i in range(9)
    ]
    bp_imgs += [
        pygame.transform.scale(
            pygame.image.load(str(BODYPARTS_DIR / f"mixed_{i:02d}.png")).convert_alpha(),
            (ITEM_SIZE, ITEM_SIZE),
        )
        for i in range(15)
    ]
    brain_sets = [
        [
            pygame.transform.scale(
                pygame.image.load(str(BODYPARTS_DIR / f"brains4_{i:02d}.png")).convert_alpha(),
                (ITEM_SIZE, ITEM_SIZE),
            )
            for i in range(4)
        ],
        [
            pygame.transform.scale(
                pygame.image.load(str(BODYPARTS_DIR / f"brains6_{i:02d}.png")).convert_alpha(),
                (ITEM_SIZE, ITEM_SIZE),
            )
            for i in range(6)
        ],
    ]
    return bp_imgs, brain_sets


class CatchBrainsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, lives=5, window_w=600, window_h=700):
        self.window_w = window_w
        self.window_h = window_h
        self.game_h = window_h - HUD_HEIGHT
        self.max_lives = lives
        self.render_mode = render_mode

        obs_low = np.zeros(12, dtype=np.float32)
        obs_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._assets_loaded = False
        self._first_reset = True

        self.zombie_x = float(window_w // 2)
        self.items: list[FallingItem] = []
        self.score = 0.0
        self.lives = lives
        self._steps = 0
        self._spawn_counter = 0
        self._is_moving = False
        self._facing_right = False
        self._walk_frame = WALK_IDLE_FRAME
        self._walk_timer = 0

    # ── asset loading ────────────────────────────────────────────────────────

    def _load_assets(self):
        if self._assets_loaded:
            return
        self._bg = pygame.transform.scale(
            pygame.image.load(str(IMGS_DIR / "board-bg.jpg")).convert(),
            (self.window_w, self.window_h),
        )
        self._rising = SpriteSheet(IMGS_DIR / "rising.png", RISING_FRAMES)
        self._walking = SpriteSheet(IMGS_DIR / "walking.png", WALKING_FRAMES)
        self._dying = SpriteSheet(IMGS_DIR / "dying.png", DYING_FRAMES)
        self._bp_base, self._brain_base_sets = _load_base_item_images()

        self._heart_full = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.polygon(
            self._heart_full, (220, 40, 60),
            [(12, 22), (0, 10), (0, 5), (6, 0), (12, 5), (18, 0), (24, 5), (24, 10)],
        )
        self._heart_empty = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.polygon(
            self._heart_empty, (80, 80, 80),
            [(12, 22), (0, 10), (0, 5), (6, 0), (12, 5), (18, 0), (24, 5), (24, 10)],
        )
        self._font = pygame.font.SysFont("monospace", 22, bold=True)
        self._assets_loaded = True

    def _ensure_pygame(self):
        if self._window is not None:
            return
        pygame.init()
        if self.render_mode == "human":
            self._window = pygame.display.set_mode((self.window_w, self.window_h))
            pygame.display.set_caption("CatchBrains")
        else:
            self._window = pygame.Surface((self.window_w, self.window_h))
        self._clock = pygame.time.Clock()
        self._load_assets()

    # ── intro / death sequences ──────────────────────────────────────────────

    def _play_video_intro(self):
        import cv2
        cap = cv2.VideoCapture(str(IMGS_DIR / "animacja_poczatkowa.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (self.window_w, self.window_h))
            self._window.blit(surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(fps)
        cap.release()

    def _play_rising(self):
        for i in range(len(self._rising)):
            self._window.blit(self._bg, (0, 0))
            self._window.blit(
                self._rising.get_frame(i),
                (self.window_w // 2 - ZOMBIE_W // 2, self.game_h - ZOMBIE_H),
            )
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(8)

    def _play_death(self):
        for i in range(len(self._dying)):
            self._window.blit(self._bg, (0, 0))
            self._window.blit(
                self._dying.get_frame(i, self._facing_right),
                (int(self.zombie_x) - ZOMBIE_W // 2, self.game_h - ZOMBIE_H),
            )
            self._draw_hud()
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(8)

        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self._window.blit(overlay, (0, 0))

        font_big = pygame.font.SysFont("monospace", 52, bold=True)
        font_sub = pygame.font.SysFont("monospace", 30)
        go_surf = font_big.render("GAME OVER", True, (220, 40, 60))
        sc_surf = font_sub.render(f"Score: {int(self.score)}", True, (230, 230, 60))
        cx, cy = self.window_w // 2, self.window_h // 2
        self._window.blit(go_surf, (cx - go_surf.get_width() // 2, cy - 60))
        self._window.blit(sc_surf, (cx - sc_surf.get_width() // 2, cy + 10))
        pygame.display.flip()

        deadline = pygame.time.get_ticks() + 10_000
        while pygame.time.get_ticks() < deadline:
            pygame.event.pump()
            self._clock.tick(10)

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(self):
        pygame.draw.rect(self._window, (20, 20, 20),
                         pygame.Rect(0, self.game_h, self.window_w, HUD_HEIGHT))
        for i in range(self.max_lives):
            surf = self._heart_full if i < self.lives else self._heart_empty
            self._window.blit(surf, (10 + i * 30, self.game_h + (HUD_HEIGHT - 24) // 2))
        sc = self._font.render(f"SCORE: {int(self.score)}", True, (230, 230, 60))
        self._window.blit(sc, (self.window_w - sc.get_width() - 10, self.game_h + (HUD_HEIGHT - 22) // 2))

    # ── observation ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(12, dtype=np.float32)
        obs[0] = self.zombie_x / self.window_w
        for i, item in enumerate(sorted(self.items, key=lambda it: it.y, reverse=True)[:3]):
            base = 1 + i * 3
            obs[base]     = item.x / self.window_w
            obs[base + 1] = item.y / self.game_h
            obs[base + 2] = 1.0 if item.item_type == "brain" else 0.0
        obs[10] = self.lives / self.max_lives
        obs[11] = self.score / 100.0
        return obs

    # ── spawning ─────────────────────────────────────────────────────────────

    def _spawn_item(self):
        is_brain = self.np_random.random() < BRAIN_CHANCE
        x = float(self.np_random.integers(ITEM_SIZE // 2, self.window_w - ITEM_SIZE // 2))
        size = int(ITEM_SIZE * (ITEM_SIZE_MIN + self.np_random.random() * (ITEM_SIZE_MAX - ITEM_SIZE_MIN)))
        speed = float(ITEM_SPEED_MIN + self.np_random.random() * (ITEM_SPEED_MAX - ITEM_SPEED_MIN))

        if self._assets_loaded:
            if is_brain:
                base_imgs = random.choice(self._brain_base_sets)
            else:
                base_imgs = [random.choice(self._bp_base)]
            imgs = [pygame.transform.scale(img, (size, size)) for img in base_imgs]
        else:
            imgs = []

        self.items.append(FallingItem(
            x=x, y=0.0,
            item_type="brain" if is_brain else "bodypart",
            images=imgs, speed=speed, size=size,
        ))

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.zombie_x = float(self.window_w // 2)
        self.items = []
        self.score = 0.0
        self.lives = self.max_lives
        self._steps = 0
        self._spawn_counter = 0
        self._is_moving = False
        self._facing_right = False
        self._walk_frame = WALK_IDLE_FRAME
        self._walk_timer = 0

        if self.render_mode == "human":
            self._ensure_pygame()
            if self._first_reset:
                self._play_video_intro()
                self._play_rising()
            self._first_reset = False

        return self._get_obs(), {}

    def step(self, action):
        act = float(action[0])
        if act > 0.01:
            self._facing_right = True
            self._is_moving = True
        elif act < -0.01:
            self._facing_right = False
            self._is_moving = True
        else:
            self._is_moving = False

        speed = BASE_SPEED / (1.0 + self.score * SLOWDOWN)
        self.zombie_x = float(np.clip(self.zombie_x + act * speed, 0, self.window_w))

        # advance items
        for item in self.items:
            item.y += item.speed
            if item.item_type == "brain" and item.images:
                item.anim_timer += 1
                if item.anim_timer >= 6:
                    item.anim_frame = (item.anim_frame + 1) % len(item.images)
                    item.anim_timer = 0

        # spawn
        self._spawn_counter += 1
        if self._spawn_counter >= SPAWN_INTERVAL and len(self.items) < MAX_ITEMS_ON_SCREEN:
            self._spawn_counter = 0
            self._spawn_item()

        # catch zone: top half of zombie body
        zombie_top_y = self.game_h - ZOMBIE_H
        catch_zone_bottom = self.game_h - ZOMBIE_H // 2

        reward = 0.0
        terminated = False
        still_falling = []

        for item in self.items:
            item_center_y = item.y + item.size // 2
            item_center_x = item.x

            if item_center_y < zombie_top_y:
                # still above zombie
                still_falling.append(item)
                continue

            if item_center_y <= catch_zone_bottom:
                # inside catch zone — check horizontal overlap
                horizontal_ok = abs(item_center_x - self.zombie_x) <= ZOMBIE_W // 2 + item.size // 2
                if horizontal_ok:
                    if item.item_type == "brain":
                        self.score += 10
                        reward += 10
                    else:
                        self.lives -= 1
                        reward -= 15
                    # caught — remove from list
                else:
                    still_falling.append(item)  # still in zone, not aligned yet
            else:
                # passed below catch zone without being caught
                if item.item_type == "brain":
                    reward -= 5
                # bodypart miss: no penalty

        self.items = still_falling

        if self.lives <= 0:
            reward -= 50
            terminated = True

        self._steps += 1

        if self.render_mode == "human":
            self._render_frame()
            if terminated:
                self._play_death()

        return self._get_obs(), reward, terminated, False, {"score": self.score, "lives": self.lives}

    # ── rendering ────────────────────────────────────────────────────────────

    def _render_frame(self):
        self._ensure_pygame()
        self._window.blit(self._bg, (0, 0))

        for item in self.items:
            if item.images:
                img = item.images[item.anim_frame % len(item.images)]
                self._window.blit(img, (int(item.x) - item.size // 2, int(item.y)))

        if self._is_moving:
            self._walk_timer += 1
            if self._walk_timer >= 5:
                self._walk_frame = (self._walk_frame + 1) % WALKING_FRAMES
                self._walk_timer = 0
        else:
            self._walk_frame = WALK_IDLE_FRAME
            self._walk_timer = 0

        self._window.blit(
            self._walking.get_frame(self._walk_frame, self._facing_right),
            (int(self.zombie_x) - ZOMBIE_W // 2, self.game_h - ZOMBIE_H),
        )
        self._draw_hud()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._window)), axes=(1, 0, 2)
            )

    def render(self):
        if self.render_mode == "rgb_array":
            self._ensure_pygame()
            return self._render_frame()

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
