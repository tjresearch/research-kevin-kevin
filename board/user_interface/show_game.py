import sys
from chess_convert import *
"""
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
    f = board_to_half_FEN(board)
    print(f)
    f_board = FEN_to_board(f)
    display(f_board)
exit(0)
"""
"""
takes pgn file input
returns list of white/black moves by num, strips other input
"""
def clean_pgn(filename):
    if not filename.endswith(".pgn"):
        print("input .pgn file")
        return
    lines = open(filename, "r").read().split("\n")
    # print(lines)

    game_info = []
    comments = []
    num_move_list = []
    no_com = []
    for l in range(len(lines)):
        line = lines[l]
        # print("g", game_info)
        # print("c", comments)
        print("l",line)
        if not line: continue
        if "[" in line and "]" in line:
            game_info.append(line[1:-1])
            continue

        bracket_ct = 0
        for ch in line:
            if ch == "{":
                bracket_ct += 1
            if ch == "}":
                bracket_ct -= 1
        if bracket_ct:
            print("invalid bracketing")
            return
        if "{" in line:
            comments.append(line[line.index("{")+1:line.index("}")])
            lines[l] = lines[l][:line.index("{")-1]

        no_com.append(lines[l].replace('\n', '')) #reformatted line
    move_txt = ' '.join(no_com)
    # print(move_txt)
    for c in range(len(move_txt)):
        if move_txt[c] == ".": #indicator of move #
            next = move_txt.find(".", c+1)
            if next == -1:
                next = len(move_txt)
            moves = move_txt[c+1:next].strip().split(" ")[:2]
            # print(moves)
            num_move_list.append(moves)
    return num_move_list

def flatten_move_list(num_move_list):
    flat = []
    for pair in num_move_list:
        flat += [*pair]
    return flat

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
    # print(move_list[i])
    display(board)
    # display(board, i%2+1)
