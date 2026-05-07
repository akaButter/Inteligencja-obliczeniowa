import gymnasium as gym
import CatchBrains
import time

env = gym.make("CatchBrains/CatchBrains", render_mode="human")

observation, info = env.reset()

print("Environment initialized!")
print(f"Initial Observation: {observation}")

for episode in range(5):
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    observation, info = env.reset()

    while not (terminated or truncated):
        action = env.action_space.sample() 
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # time.sleep(0.01) 

    print(f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward}")

env.close()
print("done")