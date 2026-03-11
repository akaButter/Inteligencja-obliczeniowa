from hexapawn import Hexapawn

from easyAI import AI_Player, Human_Player, Negamax

scoring = lambda game: -100 if game.lose() else 0

ai1 = Negamax(10, scoring)
ai2 = Negamax(10, scoring)
game = Hexapawn([AI_Player(ai1), AI_Player(ai2)], size=(4,4), chance = 0.1)
game.play()
print("player %d wins after %d turns " % (game.opponent_index, game.nmove))