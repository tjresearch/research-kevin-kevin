import pgn_helper as ph
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
        # print("4 diffs = castling")
        # print(diffs)
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
        # print("3 diffs = en passant")
        # ph.display(st_board)
        # print(diffs)
        # ph.display(board)
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
        ph.display(st_board)
        print(diffs)
        ph.display(board)
        for d in diffs:
            print(board[d[0]][d[1]])
        exit("too few diffs")

    # print(diffs)
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
            # print(piece, start)
            # ph.display(board)
        else:
            end = d
            capt = st_board[d[0]][d[1]] != "-" #if end sq wasn't blank, capt happened

    # print(end, piece)

    if piece.upper() == "P": #pawns are different, only need ltr if capt
        pawn = str(chr(start[1]+97)) if capt else ""
        piece = ""
        if board[end[0]][end[1]].upper() != "P":
            promo = "="+board[end[0]][end[1]].upper()
    else:
        starts = ph.REL_STARTS[piece.upper()]
        search_sp = [[(end[0]+p[0], end[1]+p[1]) for p in dir] for dir in starts]

        st_poss = set()
        for dir in search_sp:
            for pos in dir:
                if ph.in_bounds(pos):
                    if st_board[pos[0]][pos[1]] == piece:
                        # print(st_board[pos[0]][pos[1]], pos)
                        st_poss.add(pos)
                    if piece.upper() != "N": #not sure if in {"N", "n"} is faster
                        if board[pos[0]][pos[1]] != "-": break

        # print(st_poss)
        if len(st_poss) > 1:
            # print("COLLISION")
            # print(st_poss)
            #figure out which st piece is correct one
            for pos in st_poss:
                if board[pos[0]][pos[1]] == "-":
                    # print("st_poss",pos)
                    diff_col = coords_to_alg(pos)[0]
                    break #maybe not needed

    # print(pawn, piece, diff_col, capt, start, end, promo)
    pgn_move = pawn+piece.upper()+diff_col+"x"+coords_to_alg(end)+promo
    if not capt:
        pgn_move = pgn_move.replace("x","")
        # if pgn_move[0] in ph.COL_HEADS:
        #     pgn_move = pgn_move[1:]
    # print(pgn_move)

    #check handler
    if promo: piece = promo[-1] #update promoted pawns
    if piece: #not pawn
        starts = ph.REL_STARTS[piece.upper()]
        search_sp = [[(end[0]+p[0], end[1]+p[1]) for p in dir] for dir in starts]
        for dir in search_sp:
            for pos in dir:
                if ph.in_bounds(pos):
                    if (piece.islower() and board[pos[0]][pos[1]] == "K") or (piece.isupper() and board[pos[0]][pos[1]] == "k"):
                        pgn_move += "+" #check
                    if piece.upper() != "N": #not sure if in {"N", "n"} is faster
                        if board[pos[0]][pos[1]] != "-": break
    else: #pawn
        if board[end[0]][end[1]].isupper(): #white moved
            for c in {(end[0]-1, end[1]-1), (end[0]-1, end[1]+1)}:
                if ph.in_bounds(c):
                    if board[c[0]][c[1]] == "k":
                        pgn_move += "+"
        else: #black moved
            for c in {(end[0]+1, end[1]-1), (end[0]+1, end[1]+1)}:
                if ph.in_bounds(c):
                    if board[c[0]][c[1]] == "K":
                        pgn_move += "+"
    return pgn_move

def end_handler(ch):
    if ch == "w": #white win
        return "1-0"
    if ch == "b": #black win
        return "0-1"
    if ch == "t": #tie
        return "1/2-1/2"
    if ch == "i": #incomplete
        return "*"
    print("incorrect key entered")
    return ""
