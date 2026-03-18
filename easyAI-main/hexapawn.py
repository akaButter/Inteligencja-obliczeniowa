from easyAI import TwoPlayerGame
import random


# Convert D7 to (3,6) and back...
to_string = lambda move: " ".join(
    ["ABCDEFGHIJ"[move[i][0]] + str(move[i][1] + 1) for i in (0, 1)]
)
to_tuple = lambda s: ("ABCDEFGHIJ".index(s[0]), int(s[1:]) - 1)


class Hexapawn(TwoPlayerGame):

    def __init__(self, players, size=(4, 4), chance=0.1):
        self.chance = chance
        self.size = M, N = size
        p = [[(i, j) for j in range(N)] for i in [0, M - 1]]

        for i, d, goal, pawns in [(0, 1, M - 1, p[0]), (1, -1, 0, p[1])]:
            players[i].direction = d
            players[i].goal_line = goal
            players[i].pawns = [(p, p) for p in pawns]

            players[i].lost_pawns = []
        self.players = players
        self.current_player = 1
        self.history = []


    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        pos = [curr for _, curr in self.player.pawns]
        opp_pos = [curr for _, curr in self.opponent.pawns]
        d = self.player.direction
        for orig,(i, j) in self.player.pawns:
            if (i + d, j) not in opp_pos:
                moves.append(((i, j), (i + d, j)))
            if (i + d, j + 1) in opp_pos:
                moves.append(((i, j), (i + d, j + 1)))
            if (i + d, j - 1) in opp_pos:
                moves.append(((i, j), (i + d, j - 1)))

        return list(map(to_string, [(i, j) for i, j in moves]))

    def make_move(self, move, force_resurrect=False):
        move = list(map(to_tuple, move.split(" ")))
        # ind = self.player.pawns.index(move[0])
        # self.player.pawns[ind] = move[1]
        move_data = {'captured': None, 'resurrected': None, 'pawn_idx': None}

        for idx, (orig, curr) in enumerate(self.player.pawns):
            if curr == move[0]:
                move_data['pawn_idx'] = idx
                self.player.pawns[idx] = (orig, move[1])
                break
        for idx, (orig, curr) in enumerate(self.opponent.pawns):
            if curr == move[1]:
                captured_pawn = self.opponent.pawns.pop(idx)
                self.opponent.lost_pawns.append(captured_pawn)
                move_data['captured'] = captured_pawn
                break
        if (random.random() < self.chance or force_resurrect) and self.player.lost_pawns:
            pawn = random.choice(self.player.lost_pawns)
            all = [p[1] for p in self.player.pawns + self.opponent.pawns]
            if pawn[0] not in all:
                self.player.lost_pawns.remove(pawn)
                self.player.pawns.append((pawn[0], pawn[0]))
                move_data["resurrected"] = pawn
                # print(f'Pawn {pawn} is resurected')
    # def unmake_move(self, move):
    #     if len(self.history) == 0:
    #         return
    #     move_data = self.history.pop()
    #     move_tuples = list(map(to_tuple, move.split(" ")))
    #     start_p, _ = move_tuples

    #     if move_data['resurrected']:
    #         pawn = move_data['resurrected']
    #         self.player.pawns.pop() 
    #         self.player.lost_pawns.append(pawn)

    #     if move_data['captured']:
    #         pawn = move_data['captured']
    #         self.opponent.lost_pawns.remove(pawn)
    #         self.opponent.pawns.append(pawn)

    #     idx = move_data['pawn_idx']
    #     orig, _ = self.player.pawns[idx]
    #     self.player.pawns[idx] = (orig, start_p)

    def lose(self):
        opp = [p[1] for p in self.opponent.pawns]
        return any([i == self.opponent.goal_line for i, j in opp]) or (
            self.possible_moves() == []
        )

    def is_over(self):
        return self.lose()

    def show(self):
        p1_pos = [p[1] for p in self.players[0].pawns]
        p2_pos = [p[1] for p in self.players[1].pawns]
        f = (
            lambda x: "1"
            if x in p1_pos
            else ("2" if x in p2_pos else ".")
        )
        print(
            "\n".join(
                [
                    " ".join([f((i, j)) for j in range(self.size[1])])
                    for i in range(self.size[0])
                ]
            )
        )


