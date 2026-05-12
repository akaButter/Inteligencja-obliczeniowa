import gymnasium as gym

import CatchBrains  # noqa: F401


def main():
    env = gym.make("CatchBrains/CatchBrains", render_mode=None)

    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = env.action_space.sample()   # losowe ruchy — zastąp tu wytrenowaną polityką
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1:2d}  reward={total_reward:8.1f}  score={info['score']:.0f}  lives={info['lives']}")

    env.close()


if __name__ == "__main__":
    main()
