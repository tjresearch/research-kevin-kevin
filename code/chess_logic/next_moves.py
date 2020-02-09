import sys
import pgn_reader as reader
import pgn_helper as ph

#given one piece, figure out all next poss positions for that piece
def next_pos_from_st(board, st_pos):
    next_pos = set()
    piece = board[st_pos[0]][st_pos[1]]
    if piece == "-": return set()

    upper_piece = piece.upper()
    if upper_piece == "P":
        """
        pawn logic similar to pgn_reader.py
        en passant not handled
        """
        fwd = 1 #black pawn assumed
        if piece.isupper(): #white pawn
            fwd = -1

        #move fwd 2 if unmoved
        if st_pos[0] in {1, 6}:
            poss = [(st_pos[0]+fwd, st_pos[1]), (st_pos[0]+(2*fwd), st_pos[1])]
            for p in poss:
                if not ph.in_bounds(p): continue
                if board[p[0]][p[1]] != "-": break
                next_pos.add(p)
        #en passant
        #actually not possible to figure out en passant w/out prev board state
        #but assuming en passant is possible if the current board appears to allow it
        elif (st_pos[0] == 3 and piece.isupper()) or (st_pos[0] == 4 and piece.islower()):
            for shift in {-1, 1}:
                p = (st_pos[0], st_pos[1]+shift)
                if not ph.in_bounds(p): continue
                opp_team = (board[p[0]][p[1]].isupper() and piece.islower()) or (piece.isupper() and board[p[0]][p[1]].islower())
                if board[p[0]][p[1]].upper() == "P" and opp_team:
                    if board[p[0]+(2*fwd)][p[1]] == "-":
                        next_pos.add((p[0]+fwd, p[1]))

        #move fwd 1 otherwise (could overlap w/ above, but set fixes that)
        p = (st_pos[0]+fwd, st_pos[1])
        if ph.in_bounds(p):
            if board[p[0]][p[1]] == "-":
                next_pos.add(p)

        #pawn captures
        for shift in {-1, 1}:
            p = (st_pos[0]+fwd, st_pos[1]+shift)
            if ph.in_bounds(p):
                if board[p[0]][p[1]] != "-":
                    opp_team = (board[p[0]][p[1]].isupper() and piece.islower()) or (piece.isupper() and board[p[0]][p[1]].islower())
                    if opp_team:
                        next_pos.add(p)
    else:
        """
        doesn't handle checks, probably will do when full stack considered
        """
        #all pieces besides pawns
        starts = ph.REL_STARTS[upper_piece]
        search_sp = [[(st_pos[0]+p[0], st_pos[1]+p[1]) for p in dir] for dir in starts]

        st_poss = set()
        for dir in search_sp:
            for pos in dir:
                if ph.in_bounds(pos):
                    #capture handler
                    if board[pos[0]][pos[1]] != "-":
                        pc = board[pos[0]][pos[1]]
                        opp_team = (pc.isupper() and piece.islower()) or (piece.isupper() and pc.islower())
                        # print(opp_team, pos)
                        if opp_team:
                            next_pos.add(pos)
                        if upper_piece != "N": #not sure if in {"N", "n"} is faster
                            break
                    else:
                        next_pos.add(pos)

        #castling check
        if upper_piece in {"K", "R"}:
            #always in king, rook order
            pairs_to_check = set()
            if st_pos[0] in {0, 7}:
                if board[st_pos[0]][st_pos[1]].upper() == "K" and st_pos[1] == 4:
                    for col in [0, 7]:
                        if board[st_pos[0]][col].upper() == "R":
                            pairs_to_check.add((st_pos, (st_pos[0], col)))
                elif board[st_pos[0]][st_pos[1]].upper() == "R" and st_pos[1] in {0, 7}:
                    if board[st_pos[0]][4].upper() == "K":
                        pairs_to_check.add(((st_pos[0], 4), st_pos))
            print()
            print(pairs_to_check)
            print()

            for p1, p2 in pairs_to_check:
                low = min(p1[1], p2[1])
                high = max(p1[1], p2[1])

                print(low, high)
                good_castle = True
                for c in range(low+1, high):
                    if board[p1[0]][c] != "-":
                        good_castle = False
                        break
                print(p1, p2, good_castle)
                # TODO: add the correct positions (rn adding where the pieces are, not where they will castle to
                if good_castle:
                    if upper_piece == "K":
                        next_pos.add(p1)
                    else:
                        next_pos.add(p2)

    bitboard = [[0 for j in range(8)] for i in range(8)]
    for r, c in next_pos:
        bitboard[r][c] = 1
    return piece, bitboard

#take board state, get all possible next piece positions
#return as a 3D stacked_board
#where every sqr has set of every piece that could be there
#(includes possibility of pieces not moving)
def get_stacked_board(board):
    piece_bitboards = []
    for i in range(8):
        for j in range(8):
            pc = board[i][j]
            if pc == "-": continue
            piece, bitboard = next_pos_from_st(board, (i, j))
            piece_bitboards.append((piece, bitboard))

    # for piece, bitboard in piece_bitboards:
    #     if piece not in {"R", "K"}: continue
    #     print(piece)
    #     ph.display(bitboard)

    stacked_board = [[{board[i][j]} for j in range(8)] for i in range(8)]
    for piece, bitboard in piece_bitboards:
        for i in range(8):
            for j in range(8):
                if bitboard[i][j]:
                    stacked_board[i][j].add(piece)
    return stacked_board

def main():
    pgn_file = sys.argv[1]
    num_move_list = reader.clean_pgn(pgn_file)
    for i in range(len(num_move_list)):
        print(i, end=". ")
        print(num_move_list[i])
    move_list = reader.flatten_move_list(num_move_list)

    board = reader.FEN_to_board()
    ph.display(board)

    for i in range(8):
        board = reader.make_move(board, move_list[i], (i+1)%2)
        print(move_list[i])
        # ph.display(board)

    # board[4][5] = "q"
    # board[1][5] = "-"
    # board[3][5] = "p"
    # board[3][6] = "P"
    ph.display(board)
    # board = reader.make_move(board, "e4", True)
    # board[4][3] = 'p'
    # ph.display(board)
    #
    # next_pos = next_pos_from_st(board, (4, 3))
    # print(next_pos)
    #
    # for np in next_pos:
    #     board[np[0]][np[1]] = "*"
    # ph.display(board)

    stacked_board = get_stacked_board(board)
    for row in stacked_board:
        print(row)

if __name__ == '__main__':
    main()
