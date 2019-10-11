import sys
from chess_convert import *

pgn_file = sys.argv[1]
num_move_list = clean_pgn(pgn_file)
for i in range(len(num_move_list)):
    print(i, end=". ")
    print(num_move_list[i])
move_list = flatten_move_list(num_move_list)
# print(move_list)

# move_list = ["e4", "e6", "Nc3", "Qh4", "Bb5", "Ke7", "a3", "Na6", "Ra2"]
# move_list = ["Nc3", "e6", "Nf3", "Ne7", "Nb5", "Nbc6", "Nbd4", "Nxd4", "Nxd4"]
board = FEN_to_board()
display(board)
for i in range(len(move_list)):
    board = make_move(board, move_list[i], (i+1)%2)
    print(move_list[i])
    display(board)
    # display(board, i%2+1)
