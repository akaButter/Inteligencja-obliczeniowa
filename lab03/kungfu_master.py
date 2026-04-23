import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np

training_period = 250           # Record video every 250 episodes
num_training_episodes = 10_000

def run_basic(env):
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        # Losowy wybór akcji
        action = env.action_space.sample()

        # Wykonanie kroku w środowisku
        observation, reward, terminated, truncated, info = env.step(action)

        # Jeśli epizod się skończył, zresetuj grę
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

def run_collect_reward(env):
    num_eval_episodes = 4
    # Add video recording for every episode
    # env = RecordVideo(
    #     env,
    #     video_folder="cartpole-agent",  # Folder to save videos
    #     name_prefix="eval",  # Prefix for video filenames
    #     episode_trigger=lambda x: True  # Record every episode
    # )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    print(f"Starting evaluation for {num_eval_episodes} episodes...")
    print(f"Videos will be saved to: cartpole-agent/")

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0

        episode_over = False
        while not episode_over:
            # Replace this with your trained agent's policy
            action = env.action_space.sample()  # Random policy for demonstration

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            episode_over = terminated or truncated

        print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

    env.close()

    # Print summary statistics
    print(f'\nEvaluation Summary:')
    print(f'Episode durations: {list(env.time_queue)}')
    print(f'Episode rewards: {list(env.return_queue)}')
    print(f'Episode lengths: {list(env.length_queue)}')

    # Calculate some useful metrics
    avg_reward = np.sum(env.return_queue)
    avg_length = np.sum(env.length_queue)
    std_reward = np.std(env.return_queue)

    print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
    print(f'Average episode length: {avg_length:.1f} steps')
    print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')

def print_info(env):
    # Discrete action space (button presses)
    env = gym.make("CartPole-v1")
    print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
    print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

    # Box observation space (continuous values)
    print(f"Observation space: {env.observation_space}")  # Box with 4 values
    # Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
    print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation

if __name__ == "__main__":
    # Ręczna rejestracja środowisk Atari (wymagane w Gymnasium >= 1.0.0)
    gym.register_envs(ale_py)

    env = gym.make("ALE/KungFuMaster-v5", render_mode="human")

    run_collect_reward(env)