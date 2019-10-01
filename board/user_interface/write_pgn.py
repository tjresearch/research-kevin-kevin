import sys
from chess_convert import *

sample_FEN = [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',
]

for FEN in sample_FEN:
    print(FEN)
    board = FEN_to_board(FEN)
    display(board)
    # f = board_to_half_FEN(board)
    # print(f)
    # f_board = FEN_to_board(f)
    # display(f_board)
