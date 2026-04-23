import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(log_folder, title='Krzywa uczenia Lunar Lander'):
    # Wczytujemy plik z logami (SB3 nazywa go monitor.csv)
    file_path = os.path.join(log_folder, "monitor.csv")
    data = pd.read_csv(file_path, skiprows=1)
    
    # Wyciągamy nagrody (column 'r') i wygładzamy je
    rewards = data['r']
    # Średnia krocząca (wygładzanie), żeby wykres był czytelny
    window = 50
    smoothed_rewards = rewards.rolling(window=window).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Nagroda surowa', color='skyblue')
    plt.plot(smoothed_rewards, label=f'Średnia krocząca ({window} ep.)', color='darkblue', linewidth=2)
    
    plt.xlabel('Epizody')
    plt.ylabel('Nagroda (Reward)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
log_dir = "./logs/"
plot_results(log_dir)