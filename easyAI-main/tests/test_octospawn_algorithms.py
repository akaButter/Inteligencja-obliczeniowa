import random

from easyAI import Human_Player

from expectiminimax import ExpectiMiniMax
from hexapawn import Hexapawn
from mynegamax import Negamax


SCORING = lambda game: -100 if game.lose() else 0


def build_empty_octospawn(chance=0.0):
    game = Hexapawn([Human_Player("p1"), Human_Player("p2")], size=(4, 4), chance=chance)
    game.players[0].pawns = []
    game.players[1].pawns = []
    game.players[0].lost_pawns = []
    game.players[1].lost_pawns = []
    game.current_player = 1
    return game


def test_octospawn_initial_setup_and_possible_moves():
    game = Hexapawn([Human_Player("p1"), Human_Player("p2")], size=(4, 4), chance=0.1)

    assert len(game.players[0].pawns) == 4
    assert len(game.players[1].pawns) == 4

    expected = {"A1 B1", "A2 B2", "A3 B3", "A4 B4"}
    assert set(game.possible_moves()) == expected


def test_resurrection_forced_path_puts_pawn_back_on_origin():
    game = build_empty_octospawn(chance=1.0)
    game.players[0].pawns = [((1, 0), (1, 0))]
    game.players[0].lost_pawns = [((0, 0), (2, 2))]
    game.players[1].pawns = [((3, 3), (3, 3))]

    game.make_move("B1 C1", force_resurrect=True)

    current_positions = [curr for _, curr in game.players[0].pawns]
    assert (2, 0) in current_positions
    assert (0, 0) in current_positions
    assert game.players[0].lost_pawns == []


def test_negamax_picks_forcing_capture_move():
    game = build_empty_octospawn(chance=0.0)
    game.players[0].pawns = [((0, 0), (0, 0))]
    game.players[1].pawns = [((3, 1), (1, 1))]

    ai = Negamax(depth=3, scoring=SCORING, pruning=True)
    move = ai(game)

    assert move == "A1 B2"


def test_expectiminimax_picks_forcing_capture_move_with_and_without_ab():
    for pruning in (False, True):
        game = build_empty_octospawn(chance=0.0)
        game.players[0].pawns = [((0, 0), (0, 0))]
        game.players[1].pawns = [((3, 1), (1, 1))]

        ai = ExpectiMiniMax(depth=3, scoring=SCORING, pruning=pruning)
        move = ai(game)

        assert move == "A1 B2"


def test_expectiminimax_returns_legal_move_when_resurrection_is_possible():
    random.seed(123)
    game = Hexapawn([Human_Player("p1"), Human_Player("p2")], size=(4, 4), chance=0.1)
    game.players[0].lost_pawns = [((0, 0), (2, 2))]

    ai = ExpectiMiniMax(depth=2, scoring=SCORING, pruning=True)
    move = ai(game)

    assert move in game.possible_moves()
