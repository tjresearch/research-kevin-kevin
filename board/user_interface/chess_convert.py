"""
methods for write_pgn.py and show_game.py
"""
#rows are 1-8
#assumed 8x8 2d array
#flipped means display w/ black pieces on bottom
COL_HEADS = {chr(i+97): i for i in range(8)} #ltrs a-h
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

#useful helpers
def get_col(array, i):
    print("C", i)
    return [r[i] for r in array]

def in_bounds(pos): #checking against negative coords or coords > 8
    return pos[0]>=0 and pos[1]>=0 and pos[0]<8 and pos[1]<8

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
        # print("l",line)
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

"""
useful for clean_pgn
"""
def flatten_move_list(num_move_list):
    flat = []
    for pair in num_move_list:
        flat += [*pair]
    return flat

#for FEN methods
valid_pieces = {*'rnbqkpRNBQKP'}
valid_spaces = {*'12345678'}
"""
convert FEN to 2d array of board
black is lowercase, white is capitals, spaces are #
default is std chess start board
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
                    print(board)
                    exit("invalid start board")
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
turns 2d array board to string FEN (w/out adtl FEN tags)
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
turns lowercase SAN ltr+num (c, r) to pos in (r, c)
if board has black on top, white on bottom:
a1 -> (7, 0), a8 -> (0, 0), h1 -> (7, 7), h8 -> (0, 7)
"""
def alg_to_coords(alg):
    chars = [*alg]
    return (8-int(chars[1]), COL_HEADS[chars[0]]) #ord('a') = 97

"""
helper for get_coords_of_move()
finds start coords for all pieces besides pawns with REL_STARTS for each piece
pawns handled in get_coords_of_move() directly
"""
#lookup table for find_start()
REL_STARTS = {
    "N": [[(1,2),(2,1)],[(-1,2),(-2,1)],[(1,-2),(2,-1)],[(-1,-2),(-2,-1)]],
    "R": [[(0,i) for i in range(-1, -8, -1)], [(0,i) for i in range(1, 8)], [(i,0) for i in range(-1, -8, -1)], [(i,0) for i in range(1, 8)]],
    "B": [[(i,i) for i in range(-1, -8, -1)], [(i,i) for i in range(1, 8)], [(-i,i) for i in range(-1, -8, -1)], [(-i,i) for i in range(1, 8)]],
    "K": [[t] for t in ({(i,j) for i in range(-1, 2) for j in range(-1, 2)} - {(0,0)})],
}
REL_STARTS["Q"] = REL_STARTS["R"]+REL_STARTS["B"] #add in queen moves

def find_start(board, piece, end, diff=None):
    print(piece, end, diff)
    starts = REL_STARTS[piece.upper()]
    search_sp = [[(end[0]+p[0], end[1]+p[1]) for p in dir] for dir in starts]

    poss = set()
    for dir in search_sp:
        for pos in dir:
            if in_bounds(pos):
                print(pos, board[pos[0]][pos[1]])
                if board[pos[0]][pos[1]] == piece:
                    poss.add(pos)
                if piece.upper() != "N":
                    if board[pos[0]][pos[1]] != "-": break
    if not poss:
        exit("no poss")

    if len(poss) == 1: return poss.pop()
    if not diff:
        print(poss)
        exit("two poss start pieces")

    col = COL_HEADS[diff]
    for pos in poss:
        if col == pos[1]:
            return pos

    exit("start piece not found")

"""
finds coords for halfmove for make_move()
normal move logic largely in find_start()
special cases: end of game, castling, checks/checkmates, pawn promo, piece capt
returns (start, end) tuple
"""
def get_coords_of_move(board, move, wtm):
    #print move info
    # print("white" if wtm else "black")
    # print(move)

    #game end handler
    if move in {"1-0", "0-1", "1/2-1/2", "*"}:
        print("GAME END")
        if move == "1-0":
            print("white wins")
        elif move == "0-1":
            print("black wins")
        elif move == "1/2-1/2":
            print("tie")
        else:
            print("undecided/ongoing")
        return (-1, -1), (-1, -1)

    move = move.replace("0", "O") #common error

    #castle check doesn't check if rook or king has moved
    if move == "O-O":
        print("kingside")
        r = 7 if wtm else 0
        if {board[r][-2], board[r][-3]}-{"-"}:
            exit("kcastling error")
        board[r][-3], board[r][-1] = board[r][-1], "-"
        return (r, 4), (r, 6) #treat as king move and handle rook move
    elif move == "O-O-O":
        print("queenside")
        r = 7 if wtm else 0
        if {board[r][1], board[r][2], board[r][3]}-{"-"}:
            exit("qcastling error")
        board[r][3], board[r][0] = board[r][0], "-"
        return (r, 4), (r, 2) #like kingside

    #check/checkmate
    if move[-1] == "#" or move[-1] == "+":
        print("check")
        move = move[:-1] #ignore for now

    #pawn promotion
    if "=" in move:
        print("PAWN PROMOTION")
        temp = move.split('=')
        move, promo = temp[0], temp[1]
        # print(move, promo)
        if "x" in move:
            if len(move) != 4: #no clue when this would arise
                print(move, end="")
                exit("???")
            end = alg_to_coords(move[-2:])
            start_col = COL_HEADS[move[0]]
        else:
            end = alg_to_coords(move)
            start_col = end[1]

        start_row = 1 if wtm else 6 #second-to-back rows for each player
        start = (start_row, start_col)
        board[start[0]][start[1]] = promo.upper() if wtm else promo.lower()
        return start, end

    #piece capture
    if 'x' in move:
        moves = move.split("x")
        end = alg_to_coords(moves[1])

        if len(moves[0]) == 1: #no collision
            piece = moves[0].upper() if wtm else moves[0].lower()
            print(board[end[0]][end[1]])
            if (moves[0] != 'b') and (piece.upper() in REL_STARTS): #doesn't handle b pawn correctly
                return find_start(board, piece, end), end
            pawn = "P" if wtm else "p"
            bck = 1 if wtm else -1
            start = (end[0]+bck, ord(piece.upper())-65)

            if board[end[0]][end[1]] == "-": #en passant handler
                opp_pawn = "p" if wtm else "P"
                r = 3 if wtm else 4 #if wtm, black en passant pawn should be in rank 5 (row 3), else rank 4 (row 4)
                print(end)
                print(start)
                if board[r][end[1]] == opp_pawn:
                    print("EN PASSANT CAPTURE")
                    #move pawn back a square for make_move()
                    board[r][end[1]] = "-"
                    board[end[0]][end[1]] = opp_pawn
            return start, end
        else: #with collision
            piece = moves[0][0].upper() if wtm else moves[0][0].lower()
            return find_start(board, piece, end, moves[0][1]), end

    #pawn moving fwd
    if len(move) == 2:
        end = alg_to_coords(move)
        pawn = "P" if wtm else "p"
        bck = 1 if wtm else -1
        start = (end[0]+bck, end[1])
        if board[start[0]][start[1]] != pawn:
            start = (start[0]+bck, start[1])
        print(end)
        print(start)
        return start, end

    #non-pawn moving, no collision
    if len(move) == 3:
        end = alg_to_coords(move[-2:])
        piece = move[0].upper() if wtm else move[0].lower()
        return find_start(board, piece, end), end

    #non-pawn moving, with collision
    if len(move) == 4:
        end = alg_to_coords(move[-2:])
        piece = move[0].upper() if wtm else move[0].lower()
        return find_start(board, piece, end, move[1]), end

"""
applies halfmove in alg notation to 2d array of board
wtm = white_to_move bool (true if white's turn, false if black's turn)
"""
def make_move(b, move, wtm):
    board = [[r[c] for c in range(8)] for r in b] #python passes 2d arr by ref
    start, end = get_coords_of_move(board, move, wtm)
    if start == end: #end of game: start = end = (-1, -1)
        if start != (-1, -1): print(start, end)
        return board
    moved = board[start[0]][start[1]]
    capt = board[end[0]][end[1]]
    #error checking
    if (moved.isupper() and capt.isupper()) or (moved.islower() and capt.islower()):
        print(moved, capt)
        exit("wrong color captured")
    if "x" not in move and capt != "-":
        print(moved, capt)
        exit("false piece capture")
    if "x" in move and capt == "-":
        print(moved, capt)
        exit("no piece captured")
    #make move
    board[end[0]][end[1]] = board[start[0]][start[1]]
    board[start[0]][start[1]] = "-"
    return board

"""
boards to pgn
"""
"""
reverse of alg_to_coords, returns string from tuple
"""
def coords_to_alg(coords): #ord('a') = 97
     return str(chr(coords[1]+97))+str(8-coords[0])

def find_pgn_move(st_board, board):
    diffs = set()
    for r in range(8): #assumed 8x8
        for c in range(8):
            if st_board[r][c] != board[r][c]:
                diffs.add((r,c))

    if not diffs: #could tell game win off mate
        return "{game end, manual input required}"

    #special cases
    #castling
    if len(diffs) == 4:
        print("4 diffs = castling")
        print(diffs)
        if (0, 6) in diffs:
            if "k" == board[0][6]: #black
                return "O-O"
        elif (7, 6) in diffs:
            if "K" == board[7][6]: #white
                return "O-O"
        elif (0, 2) in diffs:
            if "k" == board[0][2]:
                return "O-O-O"
        elif (7, 2) in diffs:
            if "K" == board[7][2]:
                return "O-O-O"

    #en passant
    if len(diffs) == 3:
        print("3 diffs = en passant")
        # display(st_board)
        # print(diffs)
        # display(board)
        st_pieces = {st_board[d[0]][d[1]] for d in diffs}
        end_pieces = {board[d[0]][d[1]] for d in diffs}
        if (st_pieces | end_pieces) != {"P", "p", "-"}:
            print(st_pieces|end_pieces)
            exit("three diffs not en passant")
        cap_pawn = (st_pieces-end_pieces).pop() #this will give capt pawn
        end = ""
        st_col = ""
        if cap_pawn == "p": #white captured
            for d in diffs:
                if board[d[0]][d[1]] == "P":
                    end = d
                if st_board[d[0]][d[1]] == "P":
                    st_col = str(chr(d[1]+97))
            return st_col+"x"+coords_to_alg(end)
        elif cap_pawn == "P": #black captured
            for d in diffs:
                if board[d[0]][d[1]] == "p":
                    end = d
                if st_board[d[0]][d[1]] == "p":
                    st_col = str(chr(d[1]+97))
            return st_col+"x"+coords_to_alg(end)
        else:
            exit(cap_pawn)
        exit("multiple diffs, en passant")

    #error
    if len(diffs) == 1:
        display(st_board)
        print(diffs)
        display(board)
        for d in diffs:
            print(board[d[0]][d[1]])
        exit("too few diffs")

    print(diffs)
    #values to be filled
    start = ()
    end = ()
    piece = ""
    capt = False
    diff_col = "" #filled if neccessary to diff
    pawn = ""
    promo = ""
    for d in diffs: #def two diffs by this pt
        if board[d[0]][d[1]] == "-":
            start = d
            piece = st_board[d[0]][d[1]]
            print(piece, start)
            display(board)
        else:
            end = d
            capt = st_board[d[0]][d[1]] != "-" #if end sq wasn't blank, capt happened

    print(end, piece)

    if piece.upper() == "P": #pawns are different, only need ltr if capt
        pawn = str(chr(start[1]+97)) if capt else ""
        piece = ""
        if board[end[0]][end[1]].upper() != "P":
            promo = "="+board[end[0]][end[1]].upper()
    else:
        starts = REL_STARTS[piece.upper()]
        search_sp = [[(end[0]+p[0], end[1]+p[1]) for p in dir] for dir in starts]
        print(search_sp)

        st_poss = set()
        for dir in search_sp:
            for pos in dir:
                if in_bounds(pos):
                    if st_board[pos[0]][pos[1]] == piece:
                        print(st_board[pos[0]][pos[1]], pos)
                        st_poss.add(pos)
                    if piece.upper() != "N": #not sure if in {"N", "n"} is faster
                        if board[pos[0]][pos[1]] != "-": break

        print(st_poss)
        if len(st_poss) > 1:
            print("COLLISION")
            print(st_poss)
            #figure out which st piece is correct one
            for pos in st_poss:
                if board[pos[0]][pos[1]] == "-":
                    print("st_poss",pos)
                    diff_col = coords_to_alg(pos)[0]
                    break #maybe not needed

    print(pawn, piece, diff_col, capt, start, end, promo)
    pgn_move = pawn+piece.upper()+diff_col+"x"+coords_to_alg(end)+promo
    if not capt:
        pgn_move = pgn_move.replace("x","")
        # if pgn_move[0] in COL_HEADS:
        #     pgn_move = pgn_move[1:]
    print(pgn_move)

    #check handler
    if promo: piece = promo[-1] #update promoted pawns
    if piece: #not pawn
        starts = REL_STARTS[piece.upper()]
        search_sp = [[(end[0]+p[0], end[1]+p[1]) for p in dir] for dir in starts]
        for dir in search_sp:
            for pos in dir:
                if in_bounds(pos):
                    if (piece.islower() and board[pos[0]][pos[1]] == "K") or (piece.isupper() and board[pos[0]][pos[1]] == "k"):
                        pgn_move += "+" #check
                    if piece.upper() != "N": #not sure if in {"N", "n"} is faster
                        if board[pos[0]][pos[1]] != "-": break
    else: #pawn
        if board[end[0]][end[1]].isupper(): #white moved
            for c in {(end[0]-1, end[1]-1), (end[0]-1, end[1]+1)}:
                if in_bounds(c):
                    if board[c[0]][c[1]] == "k":
                        pgn_move += "+"
        else: #black moved
            for c in {(end[0]+1, end[1]-1), (end[0]+1, end[1]+1)}:
                if in_bounds(c):
                    if board[c[0]][c[1]] == "K":
                        pgn_move += "+"
    return pgn_move
