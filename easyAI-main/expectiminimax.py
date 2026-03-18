import time


LOWERBOUND, EXACT, UPPERBOUND = -1, 0, 1
inf = float("infinity")


def expectiminimax(
    game,
    depth,
    origDepth,
    scoring,
    alpha=-inf,
    beta=+inf,
    pruning=True,
):
    if (depth == 0) or game.is_over():
        return scoring(game) * (1 + 0.001 * depth)

    possible_moves = game.possible_moves()
    state = game
    best_move = possible_moves[0]
    if depth == origDepth:
        state.ai_move = possible_moves[0]

    bestValue = -inf
    unmake_move = hasattr(state, "unmake_move")

    for move in possible_moves:
        if not unmake_move:
            game = state.copy()

        game.make_move(move)
        game.switch_player()

        if pruning:
            move_alpha_no = -expectiminimax(
                game,
                depth - 1,
                origDepth,
                scoring,
                -beta,
                -alpha,
                pruning=pruning,
            )
        else:
            move_alpha_no = -expectiminimax(
                game,
                depth - 1,
                origDepth,
                scoring,
                pruning=pruning,
            )

        # After switch_player(), game.opponent is the player that just moved.
        moving_player_has_lost_pawns = bool(game.opponent.lost_pawns)

        if unmake_move:
            game.switch_player()
            game.unmake_move(move)

        if moving_player_has_lost_pawns:
            if not unmake_move:
                game = state.copy()

            game.make_move(move, force_resurrect=True)
            game.switch_player()

            if pruning:
                move_alpha_yes = -expectiminimax(
                    game,
                    depth - 1,
                    origDepth,
                    scoring,
                    -beta,
                    -alpha,
                    pruning=pruning,
                )
            else:
                move_alpha_yes = -expectiminimax(
                    game,
                    depth - 1,
                    origDepth,
                    scoring,
                    pruning=pruning,
                )

            if unmake_move:
                game.switch_player()
                game.unmake_move(move)

            move_alpha = (1 - game.chance) * move_alpha_no + (game.chance) * move_alpha_yes
        else:
            move_alpha = move_alpha_no

        if bestValue < move_alpha:
            bestValue = move_alpha
            best_move = move
            if depth == origDepth:
                state.ai_move = move

        if alpha < move_alpha and pruning:
            alpha = move_alpha
            if alpha >= beta:
                break

    return bestValue


class ExpectiMiniMax:
    def __init__(self, depth, scoring=None, win_score=+inf, pruning=True):
        self.scoring = scoring
        self.depth = depth
        self.win_score = win_score
        self.time = []
        self.pruning = pruning

    def __call__(self, game):
        scoring = self.scoring if self.scoring else (lambda g: g.scoring())

        start = time.time()
        self.alpha = expectiminimax(
            game,
            self.depth,
            self.depth,
            scoring,
            -self.win_score,
            +self.win_score,
            pruning=self.pruning,
        )
        end = time.time()
        self.time.append(end - start)
        return game.ai_move