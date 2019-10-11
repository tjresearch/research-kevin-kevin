"""
just to test find_pgn_move(), which in reality will be called elsewhere
"""

import sys
from chess_convert import *

"""
sample_FEN = [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',
]

boards = [FEN_to_board(FEN) for FEN in sample_FEN]

for i in range(len(boards)-1):
    # display(boards[i])
    print(find_pgn_move(boards[i], boards[i+1]))
    display(boards[i+1])
"""
"""
test = [(7,0),(0,0),(7,7),(0,7)] #a1, a8, h1, h8
for t in test:
    print(coords_to_alg(t))
"""

#will only work if pgn file is in directory
pgn_file = sys.argv[1]
num_move_list = clean_pgn(pgn_file)
for i in range(len(num_move_list)):
    print(i, end=". ")
    print(num_move_list[i])
move_list = flatten_move_list(num_move_list)
print(move_list)

boards = [FEN_to_board()]
board = FEN_to_board()
for i in range(len(move_list)):
    board = make_move(board, move_list[i], (i+1)%2)
    # print(move_list[i])
    # display(board)
    boards.append(board)

for i in range(len(boards)):
    display(boards[i])

print("-"*64)
# my_move_list = [find_pgn_move(boards[i], boards[i+1]) for i in range(len(boards)-1)]
# print(my_move_list)
my_moves = []
for i in range(len(boards)-1):
    # print(i)
    my_moves.append(find_pgn_move(boards[i], boards[i+1]))
    print(my_moves)
    print("-" * 64)

print(my_moves)
print(move_list)
if my_moves != move_list:
    print("DIFFS")
    for i in range(len(my_moves)):
        if my_moves[i] != move_list[i]:
            print(my_moves[i], move_list[i])
else:
    print("lists match")

out_dir = "out_games"
#will only work if pgn file is in directory
f = open(out_dir+"/"+"my_"+pgn_file[pgn_file.find("/")+1:], "w+")
f.write("{COMPUTER WRITTEN GAME}\n\n")
for i in range(int(len(my_moves)/2)):
    f.write("{}. {} {}".format(i+1, my_moves[i*2], my_moves[i*2+1]) + "\n")
