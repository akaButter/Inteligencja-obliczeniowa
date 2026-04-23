import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os
from tqdm import tqdm  # <-- NOWY IMPORT Paska postępu

# ==========================================
# 1. Konfiguracja Hiperparametrów
# ==========================================
NUM_EPISODES = 1000  # Całkowita liczba epizodów (zwiększ np. do 1000, gdy upewnisz się że działa)
RECORD_INTERVAL = max(1, NUM_EPISODES // 4)  # Nagrywaj wideo co 1/4 epizodów

ALPHA = 0.1  # Learning Rate - "stabilny balans"
GAMMA = 0.9  # Współczynnik dyskontowy - "złoty środek"
EPSILON_START = 1.0  # 100% eksploracji na początku
EPSILON_END = 0.1  # Zawsze zostawiamy minimum 10% na losowe akcje
EPSILON_DECAY = 0.995  # Szybkość "zanikania" eksploracji


# ==========================================
# 2. Pomocnicza funkcja dyskretyzacji stanu
# ==========================================
def discretize_state(ram_obs):
    """
    Pamięć RAM Atari to 128 bajtów. Dzielimy wartości przez 32.
    Aby uniknąć problemów z 'unhashable type: numpy.ndarray',
    wymuszamy spłaszczenie i całkowitą konwersję do standardowych intów Pythona.
    """
    binned_obs = (np.array(ram_obs).flatten() // 32).astype(int)
    return tuple(binned_obs.tolist())


# ==========================================
# 3. Główna funkcja trenująca Q-Learning
# ==========================================
def train_q_learning():
    # Ręczna rejestracja środowisk Atari
    gym.register_envs(ale_py)

    # POPRAWKA: Przywrócono wersję "-ram-v5", aby tabela Q miała szansę działać!
    env = gym.make("ALE/KungFuMaster-v5", obs_type="ram", render_mode="rgb_array")

    # Wrapper do nagrywania wideo
    video_folder = "kungfu_qlearning_videos"
    os.makedirs(video_folder, exist_ok=True)

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="qlearn-training",
        episode_trigger=lambda ep: ep % RECORD_INTERVAL == 0,
        disable_logger=True
    )

    action_size = env.action_space.n

    # Inicjalizacja Pustej Tabeli Q
    q_table = defaultdict(lambda: np.ones(action_size))

    epsilon = EPSILON_START
    rewards_history = []
    lengths_history = []

    print(f"Rozpoczynam trening Q-Learning na {NUM_EPISODES} epizodów.")
    print(f"Wideo będzie nagrywane co {RECORD_INTERVAL} epizodów do folderu '{video_folder}'.\n")

    # Pasek postępu
    pbar = tqdm(range(NUM_EPISODES), desc="Trening Q-Learning", unit="epizod")

    for episode in pbar:
        obs, info = env.reset()
        state = discretize_state(obs)

        episode_reward = 0
        steps = 0
        episode_over = False

        while not episode_over:
            # -----------------------------------
            # KROK 1: Wybór akcji (Epsilon-Greedy)
            # -----------------------------------
            if random.random() < epsilon:
                # EKSPLORACJA: Wybierz całkowicie losową akcję
                action = env.action_space.sample()
            else:
                # EKSPLOATACJA Z ŁAMANIEM REMISÓW:
                # Zamiast action = np.argmax(q_table[state])
                q_values = q_table[state]
                max_q = np.max(q_values)
                # Znajdź wszystkie akcje, które mają ten najwyższy wynik
                best_actions = np.where(q_values == max_q)[0]
                # Wylosuj jedną z nich (dzięki temu omija "zacinanie się" na akcji 0)
                action = random.choice(best_actions)

            # -----------------------------------
            # KROK 2: Wykonanie akcji w środowisku
            # -----------------------------------
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs)
            episode_over = terminated or truncated

            # -----------------------------------
            # KROK 3: Aktualizacja Tabeli Q (Równanie Q-Learningu)
            # -----------------------------------
            best_next_q = np.max(q_table[next_state]) if not episode_over else 0.0

            td_target = reward + GAMMA * best_next_q
            td_error = td_target - q_table[state][action]

            q_table[state][action] += ALPHA * td_error

            # Przejście do nowego stanu
            state = next_state
            episode_reward += reward
            steps += 1

        # Aktualizacja historii i zmniejszenie epsilona na koniec epizodu
        rewards_history.append(episode_reward)
        lengths_history.append(steps)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Aktualizacja danych na pasku postępu (co 10 epizodów)
        if (episode + 1) % 10 == 0:
            avg_rew = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            pbar.set_postfix({
                "Śr. nagroda (ost. 50)": f"{avg_rew:.1f}",
                "Epsilon": f"{epsilon:.3f}",
                "Zbadane stany": len(q_table)
            })

    env.close()

    # ==========================================
    # 4. Statystyki Końcowe
    # ==========================================
    print("\n" + "=" * 40)
    print("PODSUMOWANIE TRENINGU")
    print("=" * 40)
    print(f"Rozmiar Tabeli Q (zbadane unikalne stany): {len(q_table)}")
    print(f"Najwyższa nagroda w pojedynczym epizodzie: {np.max(rewards_history):.1f}")
    print(f"Średnia długość epizodu: {np.mean(lengths_history):.1f} kroków")

    plot_learning_curves(rewards_history)


# ==========================================
# 5. Generowanie Wykresów
# ==========================================
def plot_learning_curves(rewards):
    print("Generowanie krzywej uczenia...")
    episodes = range(len(rewards))

    plt.figure(figsize=(12, 6))

    # Surowe nagrody (tło)
    plt.plot(episodes, rewards, alpha=0.3, label='Nagroda (surowa)', color='#1f77b4')

    # Średnia krocząca
    window = max(10, len(rewards) // 20)  # Automatyczne okno w zależności od liczby epizodów
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg,
                 label=f'Średnia krocząca ({window} epizodów)', color='red', linewidth=2)

    plt.xlabel('Numer epizodu')
    plt.ylabel('Suma nagród')
    plt.title(f'Krzywa Uczenia Q-Learning (Gamma={0.9}, Alpha={0.1})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_q_learning()