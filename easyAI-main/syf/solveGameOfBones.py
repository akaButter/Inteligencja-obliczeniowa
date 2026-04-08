from easyAI import solve_with_iterative_deepening
from GameOfBones import GameOfBones

tt = TranspositionTable()
GameOfBones.ttentry = lambda game : game.pile # key for the table
r,d,m = solve_with_iterative_deepening(
    game=GameOfBones(),
    ai_depths=range(2,20),
    win_score=100,
    tt=tt
)