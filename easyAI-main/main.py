import random
from dataclasses import dataclass
from typing import Callable

from easyAI import AI_Player

from expectiminimax import ExpectiMiniMax
from hexapawn import Hexapawn
from mynegamax import Negamax


SCORING = lambda game: -100 if game.lose() else 0

# ============================
# Manual configuration section
# ============================
# Edit these values directly in Python.
DEPTH_LOW = 3
DEPTH_HIGH = 4
N_GAMES = 100
TRACE_EACH_MOVE = False
MAX_MOVES_PER_GAME = 200
DETERMINISTIC_CHANCE = 0.0
PROBABILISTIC_CHANCE = 0.1


@dataclass(frozen=True)
class AgentConfig:
	label: str
	short_label: str
	depth: int
	builder: Callable[[int], object]


def print_game_header(game_index, chance, left_label, right_label, swapped):
	print("\n" + "=" * 90)
	print(
		f"GAME {game_index} | chance={chance:.1f} | "
		f"left={left_label} vs right={right_label} | swapped={swapped}"
	)
	print("=" * 90)


def print_game_summary(game_index, winner_label, started_won, moves_played, reason):
	print("-" * 90)
	print(
		f"Summary game {game_index}: winner={winner_label}, started_won={started_won}, "
		f"moves={moves_played}, reason={reason}"
	)
	print("-" * 90)


def play_single_game(ai_p1, ai_p2, chance, seed, game_index, trace_each_move=False, title=""):
	random.seed(seed)
	game = Hexapawn([AI_Player(ai_p1), AI_Player(ai_p2)], size=(4, 4), chance=chance)

	winner = 0
	moves_played = 0
	reason = "max_moves_limit"

	if trace_each_move:
		print_game_header(game_index, chance, title, title, False)
		print("Initial board:")
		game.show()

	for _ in range(MAX_MOVES_PER_GAME):
		if game.is_over():
			winner = game.opponent_index
			reason = "terminal_before_next_move"
			break

		current_player_before = game.current_player
		move = game.get_move()
		game.play_move(move)
		moves_played += 1

		if trace_each_move:
			print(f"Move {moves_played}: player {current_player_before} -> {move}")
			game.show()

	if winner == 0 and game.is_over():
		winner = game.opponent_index
		reason = "terminal_after_last_move"

	return winner, moves_played, list(ai_p1.time), list(ai_p2.time), reason


def avg(values):
	return sum(values) / len(values) if values else 0.0


def run_selfplay(config, chance, n_games, seed_base=1000, trace_each_move=False):
	stats = {
		"p1_wins": 0,
		"p2_wins": 0,
		"draws": 0,
		"starter_wins": 0,
		"second_wins": 0,
		"times": [],
	}

	for game_id in range(n_games):
		game_index = game_id + 1
		ai_p1 = config.builder(config.depth)
		ai_p2 = config.builder(config.depth)
		winner, moves_played, t1, t2, reason = play_single_game(
			ai_p1,
			ai_p2,
			chance,
			seed=seed_base + game_id,
			game_index=game_index,
			trace_each_move=trace_each_move,
			title=config.short_label,
		)

		stats["times"].extend(t1)
		stats["times"].extend(t2)

		if winner == 1:
			stats["p1_wins"] += 1
			stats["starter_wins"] += 1
			winner_label = "P1"
		elif winner == 2:
			stats["p2_wins"] += 1
			stats["second_wins"] += 1
			winner_label = "P2"
		else:
			stats["draws"] += 1
			winner_label = "draw"

		if trace_each_move:
			print_game_summary(
				game_index=game_index,
				winner_label=winner_label,
				started_won=(winner == 1),
				moves_played=moves_played,
				reason=reason,
			)

	return {
		"label": config.label,
		"chance": chance,
		"p1_wins": stats["p1_wins"],
		"p2_wins": stats["p2_wins"],
		"draws": stats["draws"],
		"starter_wins": stats["starter_wins"],
		"second_wins": stats["second_wins"],
		"avg_time": avg(stats["times"]),
	}


def run_head_to_head(left_config, right_config, chance, n_games, seed_base=9000, trace_each_move=False):
	stats = {
		"left_wins": 0,
		"right_wins": 0,
		"draws": 0,
		"starter_wins": 0,
		"second_wins": 0,
	}

	for game_id in range(n_games):
		game_index = game_id + 1
		swap = game_id % 2 == 1

		if not swap:
			ai_p1 = left_config.builder(left_config.depth)
			ai_p2 = right_config.builder(right_config.depth)
			owner_p1 = "left"
		else:
			ai_p1 = right_config.builder(right_config.depth)
			ai_p2 = left_config.builder(left_config.depth)
			owner_p1 = "right"

		winner, moves_played, _, _, reason = play_single_game(
			ai_p1,
			ai_p2,
			chance,
			seed=seed_base + game_id,
			game_index=game_index,
			trace_each_move=trace_each_move,
			title=f"{left_config.short_label} vs {right_config.short_label}",
		)

		if winner == 0:
			stats["draws"] += 1
			winner_label = "draw"
		elif winner == 1:
			stats["starter_wins"] += 1
			if owner_p1 == "left":
				stats["left_wins"] += 1
				winner_label = "P1"
			else:
				stats["right_wins"] += 1
				winner_label = "P2"
		else:
			stats["second_wins"] += 1
			if owner_p1 == "left":
				stats["right_wins"] += 1
				winner_label = "P2"
			else:
				stats["left_wins"] += 1
				winner_label = "P1"

		if trace_each_move:
			print_game_summary(
				game_index=game_index,
				winner_label=winner_label,
				started_won=(winner == 1),
				moves_played=moves_played,
				reason=reason,
			)

	return {
		"pair": f"{left_config.short_label} vs {right_config.short_label}",
		"p1_wins": stats["left_wins"],
		"p2_wins": stats["right_wins"],
		"draws": stats["draws"],
		"starter_wins": stats["starter_wins"],
		"second_wins": stats["second_wins"],
	}


def print_separator():
	print("-" * 98)


def print_table_1(rows):
	print("\nWyniki eksperymentow")
	print("Tabela 1: Wyniki rozegranych eksperymentow")
	print_separator()
	print(f"{'Algorytm i konfiguracja':<52} {'P1':>4} {'P2':>4} {'Zaczynajacy':>12} {'Drugi':>7}")
	print_separator()
	for row in rows:
		print(
			f"{row['label']:<52} {row['p1_wins']:>4} {row['p2_wins']:>4} "
			f"{row['starter_wins']:>12} {row['second_wins']:>7}"
		)
	print_separator()
	print("Kolumny Zaczynajacy/Drugi odnosza sie do kolejnosci pierwszego ruchu w danej partii.")


def print_table_2(rows):
	print("\nTabela 2: Bezposrednie starcia algorytmow (prawdopodobienstwo=0.1, 100 gier)")
	print_separator()
	print(f"{'P1 vs P2':<52} {'P1':>4} {'P2':>4} {'Zaczynajacy':>12} {'Drugi':>7}")
	print_separator()
	for row in rows:
		print(
			f"{row['pair']:<52} {row['p1_wins']:>4} {row['p2_wins']:>4} "
			f"{row['starter_wins']:>12} {row['second_wins']:>7}"
		)
	print_separator()


def print_table_3(rows):
	print("\nTabela 3: Sredni czas decyzji")
	print_separator()
	print(f"{'Algorytm i konfiguracja':<42} {'Deterministyczna [s]':>22} {'Probabilistyczna [s]':>22}")
	print_separator()
	for row in rows:
		deterministic = row["det"] if row["det"] is not None else "-"
		if isinstance(deterministic, float):
			deterministic = f"{deterministic:.4f}"
		probabilistic = row["prob"] if row["prob"] is not None else "-"
		if isinstance(probabilistic, float):
			probabilistic = f"{probabilistic:.4f}"
		print(f"{row['label']:<42} {deterministic:>22} {probabilistic:>22}")
	print_separator()


def main():
	negamax_noab_low = AgentConfig(
		label=f"Negamax, depth={DEPTH_LOW}, bez alfa-beta, prob=0.1",
		short_label=f"Negamax(d={DEPTH_LOW},noab)",
		depth=DEPTH_LOW,
		builder=lambda depth: Negamax(depth=depth, scoring=SCORING, pruning=False),
	)
	negamax_ab_low = AgentConfig(
		label=f"Negamax, depth={DEPTH_LOW}, alfa-beta, prob=0.1",
		short_label=f"Negamax(d={DEPTH_LOW},ab)",
		depth=DEPTH_LOW,
		builder=lambda depth: Negamax(depth=depth, scoring=SCORING, pruning=True),
	)
	negamax_noab_high = AgentConfig(
		label=f"Negamax, depth={DEPTH_HIGH}, bez alfa-beta, prob=0.1",
		short_label=f"Negamax(d={DEPTH_HIGH},noab)",
		depth=DEPTH_HIGH,
		builder=lambda depth: Negamax(depth=depth, scoring=SCORING, pruning=False),
	)
	negamax_ab_high = AgentConfig(
		label=f"Negamax, depth={DEPTH_HIGH}, alfa-beta, prob=0.1",
		short_label=f"Negamax(d={DEPTH_HIGH},ab)",
		depth=DEPTH_HIGH,
		builder=lambda depth: Negamax(depth=depth, scoring=SCORING, pruning=True),
	)
	expecti_low = AgentConfig(
		label=f"ExpectiMiniMax, depth={DEPTH_LOW}, prob=0.1",
		short_label=f"ExpectiMiniMax(d={DEPTH_LOW})",
		depth=DEPTH_LOW,
		builder=lambda depth: ExpectiMiniMax(depth=depth, scoring=SCORING, pruning=True),
	)
	expecti_high = AgentConfig(
		label=f"ExpectiMiniMax, depth={DEPTH_HIGH}, prob=0.1",
		short_label=f"ExpectiMiniMax(d={DEPTH_HIGH})",
		depth=DEPTH_HIGH,
		builder=lambda depth: ExpectiMiniMax(depth=depth, scoring=SCORING, pruning=True),
	)

	print("Octospawn experiments (4x4)")
	print(f"Depths: {DEPTH_LOW} and {DEPTH_HIGH}")
	print(f"Games per experiment: {N_GAMES}")
	print(f"Board trace after each move: {TRACE_EACH_MOVE}")

	table1_rows = []

	prob_rows = [
		run_selfplay(negamax_ab_low, PROBABILISTIC_CHANCE, N_GAMES, seed_base=5000, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(negamax_noab_low, PROBABILISTIC_CHANCE, N_GAMES, seed_base=5200, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(negamax_ab_high, PROBABILISTIC_CHANCE, N_GAMES, seed_base=5400, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(negamax_noab_high, PROBABILISTIC_CHANCE, N_GAMES, seed_base=5600, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(expecti_low, PROBABILISTIC_CHANCE, N_GAMES, seed_base=5800, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(expecti_high, PROBABILISTIC_CHANCE, N_GAMES, seed_base=6000, trace_each_move=TRACE_EACH_MOVE),
	]

	det_rows = [
		run_selfplay(negamax_ab_low, DETERMINISTIC_CHANCE, N_GAMES, seed_base=1000, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(negamax_noab_low, DETERMINISTIC_CHANCE, N_GAMES, seed_base=1200, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(negamax_ab_high, DETERMINISTIC_CHANCE, N_GAMES, seed_base=1400, trace_each_move=TRACE_EACH_MOVE),
		run_selfplay(negamax_noab_high, DETERMINISTIC_CHANCE, N_GAMES, seed_base=1600, trace_each_move=TRACE_EACH_MOVE),
	]

	table1_rows.extend(prob_rows)
	table1_rows.extend(det_rows)

	# Use labels matching chance in each row.
	for row in table1_rows:
		if row["chance"] == DETERMINISTIC_CHANCE:
			row["label"] = row["label"].replace("prob=0.1", "prob=0.0")

	table2_rows = [
		run_head_to_head(
			left_config=negamax_ab_low,
			right_config=negamax_ab_high,
			chance=PROBABILISTIC_CHANCE,
			n_games=N_GAMES,
			seed_base=7000,
			trace_each_move=TRACE_EACH_MOVE,
		),
		run_head_to_head(
			left_config=negamax_ab_low,
			right_config=expecti_low,
			chance=PROBABILISTIC_CHANCE,
			n_games=N_GAMES,
			seed_base=7300,
			trace_each_move=TRACE_EACH_MOVE,
		),
		run_head_to_head(
			left_config=negamax_ab_high,
			right_config=expecti_high,
			chance=PROBABILISTIC_CHANCE,
			n_games=N_GAMES,
			seed_base=7600,
			trace_each_move=TRACE_EACH_MOVE,
		),
	]

	def find_time(rows, label_prefix):
		for row in rows:
			if row["label"].startswith(label_prefix):
				return row["avg_time"]
		return None

	table3_rows = [
		{
			"label": f"Negamax, depth={DEPTH_LOW}, alfa-beta",
			"det": find_time(det_rows, f"Negamax, depth={DEPTH_LOW}, alfa-beta"),
			"prob": find_time(prob_rows, f"Negamax, depth={DEPTH_LOW}, alfa-beta"),
		},
		{
			"label": f"Negamax, depth={DEPTH_LOW}, bez alfa-beta",
			"det": find_time(det_rows, f"Negamax, depth={DEPTH_LOW}, bez alfa-beta"),
			"prob": find_time(prob_rows, f"Negamax, depth={DEPTH_LOW}, bez alfa-beta"),
		},
		{
			"label": f"Negamax, depth={DEPTH_HIGH}, alfa-beta",
			"det": find_time(det_rows, f"Negamax, depth={DEPTH_HIGH}, alfa-beta"),
			"prob": find_time(prob_rows, f"Negamax, depth={DEPTH_HIGH}, alfa-beta"),
		},
		{
			"label": f"Negamax, depth={DEPTH_HIGH}, bez alfa-beta",
			"det": find_time(det_rows, f"Negamax, depth={DEPTH_HIGH}, bez alfa-beta"),
			"prob": find_time(prob_rows, f"Negamax, depth={DEPTH_HIGH}, bez alfa-beta"),
		},
		{
			"label": f"ExpectiMiniMax, depth={DEPTH_LOW}",
			"det": None,
			"prob": find_time(prob_rows, f"ExpectiMiniMax, depth={DEPTH_LOW}"),
		},
		{
			"label": f"ExpectiMiniMax, depth={DEPTH_HIGH}",
			"det": None,
			"prob": find_time(prob_rows, f"ExpectiMiniMax, depth={DEPTH_HIGH}"),
		},
	]

	print_table_1(table1_rows)
	print_table_2(table2_rows)
	print_table_3(table3_rows)



if __name__ == "__main__":
	main()
