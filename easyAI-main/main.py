from hexapawn import Hexapawn

from easyAI import AI_Player, Human_Player
from mynegamax import Negamax
from expectiminimax import ExpectiMiniMax

scoring = lambda game: -100 if game.lose() else 0
pruning = False
ai1 = ExpectiMiniMax(4, scoring, pruning=pruning)
ai2 = ExpectiMiniMax(4, scoring, pruning=pruning)
game = Hexapawn([AI_Player(ai1), AI_Player(ai2)], size=(4,4), chance = 0.1)
game.play()
print("player %d wins after %d turns " % (game.opponent_index, game.nmove))

print(ai1.time)
print(ai2.time)
