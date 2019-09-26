import sys

COL_HEADS = {chr(i+97): i for i in range(8)} #ltrs a-h
#rows are 1-8
#assumed 8x8 2d array
#flipped means display w/ black pieces on bottom
def display(board, flipped=False):
    if flipped:
        board = board[::-1]

    left_margin = " "*3
    output = "\n"+left_margin
    for k in COL_HEADS:
        output += "  {} ".format(k)
    line = left_margin + '-'*33
    output += "\n" + line
    for r in range(8):
        row_head = " "+str(r+1)+" " if flipped else " "+str(8-r)+" "
        output += "\n{}| ".format(row_head)
        for c in range(7):
            output += board[r][c]+' | '
        output += board[r][-1]+ ' |{}\n'.format(row_head)
        output += line
    output += "\n"+left_margin
    for k in COL_HEADS:
        output += "  {} ".format(k)
    output += "\n"
    print(output)
    return output

def get_col(array, i):
    return [r[i] for r in array]

valid_pieces = {*'rnbqkpRNBQKP'}
valid_spaces = {*'12345678'}
"""
convert FEN to 2d array of board
black is lowercase, white is capitals, spaces are #
https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
"""
def FEN_to_board(f_str="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    args = f_str.split(" ")
    #board handling
    board = []
    for s_row in args[0].split("/"):
        board.append([*s_row])
    for r in range(8): #assumed 8x8 board
        for c in range(8):
            token = board[r][c]
            if token != '-': #dash is not FEN notation
                if token not in valid_pieces | valid_spaces:
                    print("invalid start board")
                    print(board)
                    return
            if token in valid_spaces:
                board[r] = board[r][:c]+["-" for i in range(int(token))]+board[r][c+1:]
                c += int(token)

    #other FEN info, currently untouched
    if len(args) > 1: #if more than half_FEN
        player = args[1]
        castling = args[2] #case sens
        en_passant = args[3] #alg notation
        halfmove_count = args[4] #fifty move rule
        fullmove_count = args[5]

    return board
"""
turns 2d array board to string FEN
(w/out adtl FEN tags)
assumed valid 2d array
"""
def board_to_half_FEN(board):
    half_FEN = ''
    for r in range(8):
        sp_ct = 0
        for c in range(8):
            token = board[r][c]
            if token in valid_pieces:
                if sp_ct:
                    half_FEN += str(sp_ct)
                    sp_ct = 0
                half_FEN += token
            if token == '-':
                sp_ct += 1
        if sp_ct:
            half_FEN += str(sp_ct)
        half_FEN += '/'
    return half_FEN[:-1]

"""
input: lowercase ltr+num from 1-8, eg: e4, c6
output: pos in (r, c)
black on top, white on bottom:
a1 -> (7, 0), a8 -> (0, 0), h1 -> (7, 7), h8 -> (0, 7)
"""
def alg_to_coords(alg):
    chars = [*alg]
    return (8-int(chars[1]), COL_HEADS[chars[0]]) #ord('a') = 97

#search smaller areas depending on piece
#eg, pawn, search 3x3 square with end pos being (0, 1) of square
#knight (N) search only the eight relative pos needed
#will need to hardcode Knight in
#other pieces can be hardcoded with lookup table

#where to look for REL_STARTS for each piece, relative to cur pos
REL_STARTS = {
    "N": {(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)},
    "R": ({(0,i) for i in range(-7, 8)} | {(i,0) for i in range(-7, 8)}) - {(0,0)},
    "B": ({(i,i) for i in range(-7, 8)} | {(-i,i) for i in range(-7, 8)}) - {(0,0)},
    "K": {(i,j) for i in range(-1, 2) for j in range(-1, 2)} - {(0,0)},
    "P": {(1,0),(2,0)},
}
REL_STARTS["Q"] = REL_STARTS["R"]|REL_STARTS["B"] #add in queen moves

pawn_caps = {(1,1),(1,-1)} #pawn captures diff from pawn moves

def find_start(board, piece, end, diff=None):
    starts = REL_STARTS[piece.upper()]
    search_sp = [(end[0]+p[0], end[1]+p[1]) for p in starts]
    # print("sp", search_sp)
    poss = set()
    # for dir in search_sp:
    for pos in search_sp:
        if pos[0]>=0 and pos[1]>=0 and pos[0]<8 and pos[1]<8:
            if board[pos[0]][pos[1]] == piece:
                poss.add(pos)
                # break
            # if board[pos[0]][pos[1]] != "-": break
    if not poss:
        print("no poss")
        return None

    if len(poss) == 1: return poss.pop()
    if not diff:
        print("two poss pieces", poss)
        return None

    col = COL_HEADS[diff]
    for pos in poss:
        if col == pos[1]:
            return pos

    print("piece not found")
    return None

"""
finds coords for halfmove
"""
def get_coords_of_move(board, move, wtm):
    move = move.replace("0", "O") #common error
    print("white" if wtm else "black")
    print(move)
    if move == "O-O":
        print("kingside")
        r = 7 if wtm else 0
        if board[r][-2] != "-" or board[r][-3] != "-":
            print("castling error")
            return
        board[r][-3] = board[r][-1]
        board[r][-1] = "-"
        return (r, 4), (r, 6) #treat as king move and handle rook move
    elif move == "O-O-O":
        print("queenside")
    if "O" in move:
        print("\n\ncastling not handled yet\n\n")
        return
    if move[-1] == "#" or move[-1] == "+":
        print("check")
        move = move[:-1] #ignore for now
    if "=" in move:
        print("pawn promo") #not handled

    if 'x' in move: #piece capture
        moves = move.split("x")
        end = alg_to_coords(moves[1])
        if len(moves[0]) == 1: #no collision
            piece = moves[0].upper() if wtm else moves[0].lower()
            return find_start(board, piece, end), end
        else: #with collision
            piece = moves[0][0].upper() if wtm else moves[0][0].lower()
            return find_start(board, piece, end, moves[0][1]), end

    if len(move) == 2: #pawn moving fwd
        end = alg_to_coords(move)
        pawn = "P"
        if not wtm: pawn = "p"
        start = (get_col(board, end[1]).index(pawn), end[1])
        return start, end

    if len(move) == 3: #piece moving, no collision
        end = alg_to_coords(move[-2:])
        piece = move[0].upper() if wtm else move[0].lower()
        return find_start(board, piece, end), end

    if len(move) == 4: #piece moving with collision
        end = alg_to_coords(move[-2:])
        piece = move[0].upper() if wtm else move[0].lower()
        return find_start(board, piece, end, move[1]), end

"""
applies halfmove in alg notation to 2d array of board
wtm = white_to_move
true if white to move, false if black to move
"""
def make_move(board, move, wtm):
    start, end = get_coords_of_move(board, move, wtm)
    # print(start, end)
    moved = board[start[0]][start[1]]
    capt = board[end[0]][end[1]]
    print(moved, capt)
    #error checking
    if (moved.isupper() and capt.isupper()) or (moved.islower() and capt.islower()): print("wrong color captured")
    if "x" not in move and capt != "-": print("piece capt when not supposed to")
    #make move
    board[end[0]][end[1]] = board[start[0]][start[1]]
    board[start[0]][start[1]] = "-"
    return board
