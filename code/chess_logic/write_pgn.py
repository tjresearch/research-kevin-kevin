"""
converts given PGN to board states and back again
to test find_pgn_move(), which in reality will be called elsewhere
"""
import sys
import pgn_reader as reader
import pgn_writer as writer
import pgn_helper

if len(sys.argv) < 3:
    print("\nUsage: write_pgn.py pgnfile.pgn result_ltr")
    print("->result_ltr: w[hite] b[lack] t[ie] i[ncomplete]")
    exit(0)

#read input file
pgn_file = sys.argv[1] #will only work if pgn file is in directory
num_move_list = reader.clean_pgn(pgn_file)
print("input pgn:")
for i in range(len(num_move_list)):
    print(i, end=". ")
    print(num_move_list[i])
move_list = reader.flatten_move_list(num_move_list)
# print(move_list)

#create list of board states
boards = [reader.FEN_to_board()]
board = reader.FEN_to_board()
for i in range(len(move_list)):
    board = reader.make_move(board, move_list[i], (i+1)%2)
    boards.append(board)

# for i in range(len(boards)):
#     display(boards[i])

print("-"*64+"\n")
# write list of moves from board states
my_moves = []
for i in range(len(boards)-1):
    # print(i)
    my_moves.append(writer.find_pgn_move(boards[i], boards[i+1]))
    # print(my_moves)
    # print("-" * 64)

#add final move
print(my_moves[-1])
if "{" in my_moves[-1] and "}" in my_moves[-1]:
    print("adding manually inputted game result")
    result = writer.end_handler(sys.argv[2])
    print(result)
    if result:
        my_moves[-1] = result

#display differences
print("my moves:", my_moves)
print("\ngiven moves:", move_list)
if my_moves != move_list:
    print("\nDIFFS")
    for i in range(len(my_moves)):
        if my_moves[i] != move_list[i]:
            print(my_moves[i], move_list[i])
else:
    print("\nlists match")
print()

#save to output directory
out_dir = "out_games"
f = open(out_dir+"/"+"my_"+pgn_file[pgn_file.find("/")+1:], "w+")
f.write("{COMPUTER WRITTEN GAME}\n\n")
for i in range(int(len(my_moves)/2)):
    f.write("{}. {} {}".format(i+1, my_moves[i*2], my_moves[i*2+1]) + "\n")
