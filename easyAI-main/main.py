import os
import random
from dataclasses import dataclass

from easyAI import AI_Player

from expectiminimax import ExpectiMiniMax
from hexapawn import Hexapawn
from mynegamax import Negamax


SCORING = lambda game: -100 if game.lose() else 0


def env_bool(name, default):
	value = os.environ.get(name)
	if value is None:
		return default
	return value.lower() in ("1", "true", "yes", "y", "on")


TRACE_EACH_MOVE = env_bool("OCTOSPAWN_TRACE", True)
MAX_MOVES_PER_GAME = int(os.environ.get("OCTOSPAWN_MAX_MOVES", "200"))


@dataclass(frozen=True)
class AlgoVariant:
	name: str
	builder: callable


def print_game_header(game_index, variant_name, chance, depth_a, depth_b, swapped):
	print("\n" + "=" * 90)
	print(
		f"GAME {game_index} | {variant_name} | chance={chance:.1f} | "
		f"A(depth={depth_a}) vs B(depth={depth_b}) | swapped_start={swapped}"
	)
	print("=" * 90)


def print_game_summary(game_index, winner_owner, moves_played, reason):
	print("-" * 90)
	print(
		f"Summary game {game_index}: winner={winner_owner}, "
		f"moves={moves_played}, reason={reason}"
	)
	print("-" * 90)


def play_single_game(
	ai_p1,
	ai_p2,
	chance,
	seed,
	game_index,
	variant_name,
	depth_a,
	depth_b,
	swapped,
	trace_each_move,
):
	random.seed(seed)
	game = Hexapawn([AI_Player(ai_p1), AI_Player(ai_p2)], size=(4, 4), chance=chance)

	winner = 0
	moves_played = 0
	reason = "max_moves_limit"

	if trace_each_move:
		print_game_header(game_index, variant_name, chance, depth_a, depth_b, swapped)
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


def run_matchup(variant, depth_a, depth_b, chance, n_games, seed_base=1000, trace_each_move=True):
	stats = {
		"A_wins": 0,
		"B_wins": 0,
		"draws": 0,
		"A_times": [],
		"B_times": [],
		"moves": [],
	}

	for game_id in range(n_games):
		game_index = game_id + 1
		swap = game_id % 2 == 1

		if swap:
			ai_b = variant.builder(depth_b)
			ai_a = variant.builder(depth_a)
			winner, moves_played, t_p1, t_p2, reason = play_single_game(
				ai_b,
				ai_a,
				chance,
				seed=seed_base + game_id,
				game_index=game_index,
				variant_name=variant.name,
				depth_a=depth_a,
				depth_b=depth_b,
				swapped=swap,
				trace_each_move=trace_each_move,
			)
			owner_p1, owner_p2 = "B", "A"
		else:
			ai_a = variant.builder(depth_a)
			ai_b = variant.builder(depth_b)
			winner, moves_played, t_p1, t_p2, reason = play_single_game(
				ai_a,
				ai_b,
				chance,
				seed=seed_base + game_id,
				game_index=game_index,
				variant_name=variant.name,
				depth_a=depth_a,
				depth_b=depth_b,
				swapped=swap,
				trace_each_move=trace_each_move,
			)
			owner_p1, owner_p2 = "A", "B"

		if owner_p1 == "A":
			stats["A_times"].extend(t_p1)
			stats["B_times"].extend(t_p2)
		else:
			stats["A_times"].extend(t_p2)
			stats["B_times"].extend(t_p1)

		stats["moves"].append(moves_played)

		if winner == 0:
			stats["draws"] += 1
			winner_owner = "draw"
		elif (winner == 1 and owner_p1 == "A") or (winner == 2 and owner_p2 == "A"):
			stats["A_wins"] += 1
			winner_owner = "A"
		else:
			stats["B_wins"] += 1
			winner_owner = "B"

		if trace_each_move:
			print_game_summary(game_index, winner_owner, moves_played, reason)

	def avg(values):
		return sum(values) / len(values) if values else 0.0

	return {
		"variant": variant.name,
		"chance": chance,
		"depth_a": depth_a,
		"depth_b": depth_b,
		"games": n_games,
		"A_wins": stats["A_wins"],
		"B_wins": stats["B_wins"],
		"draws": stats["draws"],
		"A_avg_move_time_s": avg(stats["A_times"]),
		"B_avg_move_time_s": avg(stats["B_times"]),
		"avg_game_length_moves": avg(stats["moves"]),
	}


def print_results_table(results, title):
	print(f"\n{title}")
	print(
		"variant | chance | dA | dB | games | A_wins | B_wins | draws | "
		"A_win_rate | B_win_rate | A_avg_time[s] | B_avg_time[s] | avg_len[moves]"
	)
	print("-" * 145)

	for row in results:
		a_win_rate = row["A_wins"] / row["games"] if row["games"] else 0.0
		b_win_rate = row["B_wins"] / row["games"] if row["games"] else 0.0
		print(
			f"{row['variant']} | {row['chance']:.1f} | {row['depth_a']} | {row['depth_b']} | "
			f"{row['games']} | {row['A_wins']} | {row['B_wins']} | {row['draws']} | "
			f"{a_win_rate:.2%} | {b_win_rate:.2%} | "
			f"{row['A_avg_move_time_s']:.6f} | {row['B_avg_move_time_s']:.6f} | "
			f"{row['avg_game_length_moves']:.2f}"
		)


def main():
	depth_a = int(os.environ.get("OCTOSPAWN_DEPTH_A", "3"))
	depth_b = int(os.environ.get("OCTOSPAWN_DEPTH_B", "4"))
	n_games = int(os.environ.get("OCTOSPAWN_GAMES", "10"))

	variants = [
		AlgoVariant(
			"Negamax_noAB",
			lambda depth: Negamax(depth=depth, scoring=SCORING, pruning=False),
		),
		AlgoVariant(
			"Negamax_AB",
			lambda depth: Negamax(depth=depth, scoring=SCORING, pruning=True),
		),
		AlgoVariant(
			"ExpectiMiniMax_noAB",
			lambda depth: ExpectiMiniMax(depth=depth, scoring=SCORING, pruning=False),
		),
		AlgoVariant(
			"ExpectiMiniMax_AB",
			lambda depth: ExpectiMiniMax(depth=depth, scoring=SCORING, pruning=True),
		),
	]

	print("Octospawn experiments (4x4)")
	print(f"Compared depths: A={depth_a} vs B={depth_b}")
	print(f"Games per matchup: {n_games}")
	print(f"Board trace after each move: {TRACE_EACH_MOVE}")
	print("A and B alternate the starting player every game.")

	all_results = []
	for chance in (0.0, 0.1):
		for variant in variants:
			result = run_matchup(
				variant=variant,
				depth_a=depth_a,
				depth_b=depth_b,
				chance=chance,
				n_games=n_games,
				seed_base=1000 if chance == 0.0 else 5000,
				trace_each_move=TRACE_EACH_MOVE,
			)
			all_results.append(result)

	deterministic = [r for r in all_results if r["chance"] == 0.0]
	probabilistic = [r for r in all_results if r["chance"] == 0.1]

	print_results_table(deterministic, "Deterministic variant (chance=0.0)")
	print_results_table(probabilistic, "Probabilistic variant (chance=0.1)")


if __name__ == "__main__":
	main()
