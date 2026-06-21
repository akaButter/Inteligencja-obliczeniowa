import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium.spaces import Box, Discrete

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "W", "D", "K", "A"]
SUITS = ["Trefl", "Karo", "Kier", "Pik"]

RANK_2, RANK_3, RANK_4 = 0, 1, 2
RANK_JACK, RANK_QUEEN, RANK_KING, RANK_ACE = 9, 10, 11, 12
SUIT_HEARTS, SUIT_SPADES = 2, 3

ACTION_DRAW = 52
ACTION_PASS = 53
MAX_STEPS = 800
OBS_SIZE = 67


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def card_name(card: int) -> str:
    return f"{RANKS[card_rank(card)]}{SUITS[card_suit(card)][0]}"


class MakaoEnv(AECEnv):
    metadata = {"name": "makao_v0", "is_parallelizable": False}

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.render_mode = render_mode

        self.action_spaces = {a: Discrete(54) for a in self.possible_agents}
        self.observation_spaces = {
            a: Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for a in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        rng = np.random.default_rng(seed)
        deck = list(range(52))
        rng.shuffle(deck)

        self._hands: dict[str, set[int]] = {}
        idx = 0
        for a in self.agents:
            self._hands[a] = set(deck[idx : idx + 5])
            idx += 5

        special_ranks = {RANK_2, RANK_3, RANK_4, RANK_JACK, RANK_ACE}
        while True:
            start = deck[idx]
            idx += 1
            if card_rank(start) not in special_ranks:
                break

        self._discard: list[int] = [start]
        self._draw_pile: list[int] = deck[idx:]

        self._pending_draw = 0
        self._skip_next = False
        self._requested_suit: int | None = None
        self._requested_rank: int | None = None
        self._step_count = 0

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def observe(self, agent: str) -> np.ndarray:
        return self._build_obs(agent)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def action_mask(self) -> np.ndarray:
        agent = self.agent_selection
        mask = np.zeros(54, dtype=np.int8)
        hand = self._hands.get(agent, set())
        top = self._discard[-1]
        top_rank = card_rank(top)
        top_suit = card_suit(top)

        if self._skip_next:
            mask[ACTION_PASS] = 1
            return mask

        if self._pending_draw > 0:
            for c in hand:
                r = card_rank(c)
                if top_rank in (RANK_2, RANK_3) and r in (RANK_2, RANK_3):
                    mask[c] = 1
                elif top_rank == RANK_KING and r == RANK_KING and card_suit(c) == SUIT_SPADES:
                    mask[c] = 1
            if mask[:52].sum() == 0:
                mask[ACTION_DRAW] = 1
            return mask

        for c in hand:
            r = card_rank(c)
            s = card_suit(c)
            if r == RANK_QUEEN:
                mask[c] = 1
                continue
            if self._requested_suit is not None:
                if s == self._requested_suit or r == RANK_ACE:
                    mask[c] = 1
            elif self._requested_rank is not None:
                if r == self._requested_rank or r == RANK_JACK:
                    mask[c] = 1
            else:
                if s == top_suit or r == top_rank:
                    mask[c] = 1

        mask[ACTION_DRAW] = 1
        return mask

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        for a in self.possible_agents:
            self.rewards[a] = 0.0

        self._step_count += 1
        reward = 0.0

        if self._skip_next and action == ACTION_PASS:
            self._skip_next = False
            reward = -0.05

        elif self._pending_draw > 0 and action == ACTION_DRAW:
            for _ in range(self._pending_draw):
                self._draw_one(agent)
            reward = -0.05 * self._pending_draw
            self._pending_draw = 0

        elif action == ACTION_DRAW:
            can_play = self.action_mask()[:52].sum() > 0
            self._draw_one(agent)
            reward = -0.4 if can_play else -0.05

        else:
            card = int(action)
            self._hands[agent].discard(card)
            self._discard.append(card)
            self._requested_suit = None
            self._requested_rank = None
            r = card_rank(card)
            s = card_suit(card)

            hand_size = len(self._hands[agent])

            reward = 0.4
            if hand_size == 0:
                pass
            elif hand_size == 1:
                reward += 1.5
            elif hand_size <= 3:
                reward += 0.5

            if r in (RANK_2, RANK_3, RANK_4, RANK_JACK, RANK_QUEEN, RANK_ACE) or (
                r == RANK_KING and s in (SUIT_HEARTS, SUIT_SPADES)
            ):
                reward += 0.5

            if r == RANK_2:
                self._pending_draw += 2
                reward += 0.3 * 2
            elif r == RANK_3:
                self._pending_draw += 3
                reward += 0.3 * 3
            elif r == RANK_4:
                self._skip_next = True
                reward += 0.4
            elif r == RANK_JACK:
                self._requested_rank = self._best_rank_in_hand(agent)
            elif r == RANK_ACE:
                self._requested_suit = self._best_suit_in_hand(agent)
            elif r == RANK_KING:
                if s == SUIT_SPADES:
                    self._pending_draw += 5
                    reward += 0.3 * 5
                elif s == SUIT_HEARTS:
                    prev = self._prev_agent(agent)
                    for _ in range(5):
                        self._draw_one(prev)
                    reward += 0.3 * 5

            if len(self._hands[agent]) == 0:
                reward = 20.0
                self.terminations[agent] = True
                for other in self.agents:
                    if other != agent:
                        self.rewards[other] = -2.0
                        self.terminations[other] = True

        if not self.terminations.get(agent, False):
            own = len(self._hands.get(agent, set()))
            others_avg = sum(
                len(self._hands.get(a, set()))
                for a in self.possible_agents if a != agent
            ) / 3.0
            reward += (others_avg - own) * 0.04

        reward -= 0.003

        if self._step_count >= MAX_STEPS:
            for a in self.possible_agents:
                self.truncations[a] = True

        self.rewards[agent] = reward
        self._accumulate_rewards()

        self.agents = [a for a in self.possible_agents if not (self.terminations[a] or self.truncations[a])]

        if self.agents:
            next_agent = self._agent_selector.next()
            while next_agent not in self.agents:
                next_agent = self._agent_selector.next()
            self.agent_selection = next_agent

        if self.render_mode == "human":
            self.render()

    def render(self):
        agent = self.agent_selection
        top = self._discard[-1]
        hand = sorted(self._hands.get(agent, []))
        print(f"[{agent}] Wierzchnia: {card_name(top)} | Ręka: {[card_name(c) for c in hand]}")

    def close(self):
        pass


    def _build_obs(self, agent: str) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        hand = self._hands.get(agent, set())

        for c in hand:
            obs[c] = 1.0

        top = self._discard[-1]
        obs[52] = card_rank(top) / 12.0
        obs[53] = card_suit(top) / 3.0

        obs[54] = min(self._pending_draw / 10.0, 1.0)

        obs[55] = float(self._skip_next)

        obs[56] = self._requested_suit / 3.0 if self._requested_suit is not None else -1.0

        obs[57] = self._requested_rank / 12.0 if self._requested_rank is not None else -1.0

        others = [a for a in self.possible_agents if a != agent]
        for i, other in enumerate(others[:3]):
            obs[58 + i] = len(self._hands.get(other, set())) / 52.0

        obs[61] = len(self._draw_pile) / 52.0
        obs[62] = len(hand) / 52.0

        active = [a for a in self.possible_agents if not self.terminations.get(a, False)]
        if agent in active:
            pos = active.index(agent)
            if pos < 4:
                obs[63 + pos] = 1.0

        return obs

    def _draw_one(self, agent: str):
        if not self._draw_pile:
            if len(self._discard) > 1:
                top = self._discard[-1]
                self._draw_pile = self._discard[:-1]
                np.random.shuffle(self._draw_pile)
                self._discard = [top]
        if self._draw_pile:
            self._hands[agent].add(self._draw_pile.pop())

    def _best_rank_in_hand(self, agent: str) -> int:
        hand = self._hands.get(agent, set())
        playable_ranks = [r for r in range(9) if r not in (RANK_2, RANK_3, RANK_4)]
        counts = {r: 0 for r in range(13)}
        for c in hand:
            counts[card_rank(c)] += 1
        best = max(playable_ranks, key=lambda r: counts[r], default=5)
        return best

    def _best_suit_in_hand(self, agent: str) -> int:
        hand = self._hands.get(agent, set())
        counts = [0, 0, 0, 0]
        for c in hand:
            counts[card_suit(c)] += 1
        return int(np.argmax(counts))

    def _prev_agent(self, agent: str) -> str:
        active = [a for a in self.possible_agents if not self.terminations.get(a, False)]
        if not active:
            return agent
        idx = active.index(agent) if agent in active else 0
        return active[(idx - 1) % len(active)]
